"""Bounded, offline validation and warm-start training for the four-relation task."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
import threading
import time


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('witness', 'baseline', 'train'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--embedding-dir', type=Path, required=True)
    parser.add_argument('--initial-checkpoint', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=1, help='Physical GPU index; becomes CUDA device 0 in this process.')
    parser.add_argument('--memory-fraction', type=float, default=.25)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--input-max-length', type=int, default=512)
    parser.add_argument('--gradient-accumulation', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args(argv)
    if not 0 < args.memory_fraction <= 1 or min(args.threads, args.batch_size, args.eval_batch_size, args.gradient_accumulation, args.epochs, args.input_max_length) < 1:
        parser.error('Resource fractions and integer sizes must be positive; memory fraction must be <= 1.')
    if args.gpu < 0 or args.learning_rate <= 0:
        parser.error('GPU index must be nonnegative and learning rate must be positive.')
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        parser.error('run-id must be a single directory name.')
    return args


class ResourceSampler:
    """Sample only this process. CUDA context overhead is outside the allocator cap."""

    def __init__(self):
        self.stop = threading.Event()
        self.peak_process_mib = 0
        self.last_sample_error = None
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10, check=True)
            for line in result.stdout.splitlines():
                pid, memory = (x.strip() for x in line.split(',', 1))
                if pid == str(os.getpid()):
                    self.peak_process_mib = max(self.peak_process_mib, int(memory))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.last_sample_error = str(error)

    def _loop(self):
        self.sample()
        while not self.stop.wait(20):
            self.sample()


def main(argv=None):
    args = arguments(argv)
    # These must be set before importing torch/numpy/transformers.
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS=str(args.threads),
                      MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from .common import ROOT, read_jsonl, seed_all, sha256, utc_now, write_json, write_jsonl
    from .metrics import score_rows
    from .neural import Collator, GraphData, GraphPrefixModel, QADataset, move, train, validation_outputs

    run = ROOT / 'runs' / args.run_id
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f'Refusing to overwrite run: {run}')
    run.mkdir(parents=True, exist_ok=True)
    # train() also guards its own directory. Keep wrapper evidence alongside it.
    evidence = ROOT / '.work' / 'freeform_launches' / args.run_id
    if evidence.exists() and any(evidence.iterdir()):
        raise FileExistsError(f'Refusing to overwrite launch evidence: {evidence}')
    evidence.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(evidence / 'launch.log', encoding='utf-8')])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; refusing a silent CPU fallback')
    torch.set_num_threads(args.threads)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    seed_all(42)
    device = torch.device('cuda:0')
    selected_data, embedding, checkpoint = (ROOT / x for x in (args.data_dir, args.embedding_dir, args.initial_checkpoint))
    bindings = {
        'dataset': {name: sha256(selected_data / name) for name in ('manifest.json', 'qa.jsonl', 'entities.json', 'splits.json', 'triples.tsv')},
        'embedding': {name: sha256(embedding / name) for name in ('embeddings.npz', 'mapping.json')},
        'checkpoint': {p.name: sha256(p) for p in sorted(checkpoint.iterdir()) if p.is_file()},
        'source': {name: sha256(ROOT / name) for name in ('kgqa/neural.py', 'kgqa/freeform_experiment.py', 'kgqa/metrics.py', 'kgqa/linker.py')},
    }
    started = time.perf_counter()
    sampler = ResourceSampler()
    sampler.thread.start()
    write_json(evidence / 'launch.json', {**vars(args), 'data_dir': str(args.data_dir), 'embedding_dir': str(args.embedding_dir),
               'initial_checkpoint': str(args.initial_checkpoint), 'bindings': bindings,
               'pid': os.getpid(), 'created_at': utc_now(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
               'torch_threads': torch.get_num_threads(),
               'gpu_name': torch.cuda.get_device_name(0), 'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory,
               'allocator_cap_bytes': int(torch.cuda.get_device_properties(0).total_memory * args.memory_fraction),
               'cap_scope': 'PyTorch allocator only; sampled process peak includes CUDA overhead.'})
    status = 'failed'
    try:
        if args.mode == 'train':
            train(args.run_id, args.learning_rate, True, args.epochs, False,
                  batch_size=args.batch_size, gradient_accumulation=args.gradient_accumulation,
                  input_max_length=args.input_max_length, eval_batch_size=args.eval_batch_size,
                  data_dir=args.data_dir, embedding_dir=args.embedding_dir, initial_checkpoint=args.initial_checkpoint)
        else:
            graph = GraphData(True, args.data_dir, args.embedding_dir)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
            model = GraphPrefixModel(str(checkpoint), True).to(device)
            prefix = torch.load(checkpoint / 'graph_prefix.pt', map_location=device, weights_only=True)
            model.projection.load_state_dict(prefix['projection'])
            model.norm.load_state_dict(prefix['norm'])
            rows = read_jsonl(selected_data / 'qa.jsonl')
            if args.mode == 'baseline':
                rows = [row for row in rows if row['split'] == 'valid']
                if not rows:
                    raise ValueError('No validation questions')
                loader = DataLoader(QADataset(rows, graph, link_questions=True), batch_size=args.eval_batch_size,
                                    collate_fn=Collator(tokenizer, True, args.input_max_length))
                predictions, loss = validation_outputs(model, loader, tokenizer, graph.entities, device)
                write_jsonl(run / 'predictions_valid.jsonl', predictions)
                write_json(run / 'metrics_valid.json', {**score_rows(rows, predictions, graph.entities), 'validation_loss': loss})
                write_json(run / 'config.json', {'data_dir': str(args.data_dir), 'embedding_dir': str(args.embedding_dir),
                           'initial_checkpoint': str(args.initial_checkpoint), 'mode': 'baseline_valid_only',
                           'test_used': False, 'batch_size': args.eval_batch_size, 'input_max_length': args.input_max_length, 'seed': 42,
                           'metric_note': 'Raw generation; main AnswerEntityIndex decoder with retrieved fact context; no name recovery/adapter.'})
            else:
                # Real CUDA dispatch, gradients and optimizer states, then actual T5 generation.
                x = torch.randn(128, 128, device=device, requires_grad=True)
                product = x @ x.T
                product.square().mean().backward()
                assert product.shape == (128, 128) and torch.isfinite(x.grad).all()
                logging.info('KERNEL_WITNESS shape=%s device=%s finite=True', tuple(product.shape), torch.cuda.get_device_name(0))
                train_rows = [row for row in rows if row['split'] == 'train']
                # Exercise long fact lists and targets rather than a convenient tiny example.
                fact_counts = {qid: sum(len(tails) for tails in entity.get('facts', {}).values())
                               for qid, entity in graph.entities.items()}
                train_rows.sort(key=lambda row: (fact_counts[row['movie_id']], len(row['reference_answers'][0])), reverse=True)
                batch = Collator(tokenizer, True, args.input_max_length)([QADataset(train_rows[:args.batch_size], graph)[i] for i in range(min(args.batch_size, len(train_rows)))])
                batch.pop('rows')
                batch = move(batch, device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
                losses = []
                for _ in range(2):
                    model.train()
                    optimizer.zero_grad()
                    loss = model(**batch).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite witness loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(loss.item())
                model.eval()
                batch.pop('labels')
                decoded = tokenizer.batch_decode(model.generate(**batch), skip_special_tokens=True)
                assert len(decoded) == args.batch_size and all(text.strip() for text in decoded)
                write_json(run / 'witness.json', {'seed': 42, 'losses': losses, 'precision': 'float32',
                           'batch': args.batch_size, 'max_input_tokens': batch['input_ids'].shape[1],
                           'graph_prefix_count': batch['graph_vectors'].shape[1], 'generated': decoded})
                logging.info('MODEL_WITNESS batch=%s precision=float32 finite=True nonempty=%s', args.batch_size, len(decoded))
        status = 'complete'
        return 0
    finally:
        sampler.sample()
        sampler.stop.set()
        sampler.thread.join(timeout=12)
        write_json(evidence / 'resources.json', {'status': status, 'elapsed_seconds': time.perf_counter() - started,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                   'sampled_peak_process_mib': sampler.peak_process_mib, 'sampling_interval_seconds': 20,
                   'sampling_error': sampler.last_sample_error, 'completed_at': utc_now()})
        logging.info('FINISHED status=%s seconds=%.1f peak_reserved_mib=%.1f process_peak_mib=%s', status,
                     time.perf_counter() - started, torch.cuda.max_memory_reserved() / 2**20, sampler.peak_process_mib)


if __name__ == '__main__':
    raise SystemExit(main())
