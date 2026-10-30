from __future__ import annotations

import argparse
from contextlib import nullcontext
import logging
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from .common import RELATIONS, ROOT, read_json, read_jsonl, seed_all, setup_logging, sha256, utc_now, write_json, write_jsonl
from .metrics import AnswerEntityIndex, entity_ids_from_text, score_rows
from .linker import EntityLinker
from .neural_preflight import _facts, FACT_LIMIT


class GraphData:
    def __init__(self, use_embeddings=True, data_dir=None, embedding_dir=None):
        self.data_dir = ROOT / (data_dir or 'data/processed')
        self.embedding_dir = ROOT / (embedding_dir or 'models/transe')
        self.entities = read_json(self.data_dir / 'entities.json')
        self.use_embeddings = use_embeddings
        if use_embeddings:
            arrays = np.load(self.embedding_dir / 'embeddings.npz')
            self.entity, self.relation = arrays['entity'], arrays['relation']
            mapping = read_json(self.embedding_dir / 'mapping.json')
            self.entity_to_id, self.relation_to_id = mapping['entity_to_id'], mapping['relation_to_id']
            needed_entities, needed_relations = set(), set()
            for head, entity in self.entities.items():
                for relation, tails in entity.get('facts', {}).items():
                    if relation in RELATIONS:
                        for tail in tails:
                            if tail in self.entities and self.entities[tail].get('label'):
                                needed_entities.update((head, tail)); needed_relations.add(relation)
            if needed_entities - self.entity_to_id.keys() or needed_relations - self.relation_to_id.keys():
                raise ValueError('Graph embedding mapping does not cover this dataset; use its matching embedding directory')

    def _retained_facts(self, movie_id):
        mapping = {'entity_to_id': self.entity_to_id, 'relation_to_id': self.relation_to_id} if self.use_embeddings else None
        return _facts(movie_id, self.entities, mapping)[:FACT_LIMIT]

    def fact_tail_ids(self, movie_id):
        return [tail for _, tail in self._retained_facts(movie_id)]

    def facts(self, movie_id):
        entity = self.entities[movie_id]; rows = []
        for relation in RELATIONS:
            for tail in entity.get('facts', {}).get(relation, []):
                if tail not in self.entities or not self.entities[tail].get('label'): continue
                text = f"{entity['label']} | {RELATIONS[relation]} | {self.entities[tail]['label']}"
                vector = None
                if self.use_embeddings:
                    if movie_id not in self.entity_to_id or tail not in self.entity_to_id or relation not in self.relation_to_id:
                        continue
                    vector = np.concatenate([self.entity[self.entity_to_id[movie_id]], self.relation[self.relation_to_id[relation]], self.entity[self.entity_to_id[tail]]])
                rows.append((text, vector))
        return rows[:FACT_LIMIT]


class QADataset(Dataset):
    def __init__(self, rows, graph, link_questions=False):
        self.rows, self.graph = rows, graph
        if link_questions:
            splits = read_json(graph.data_dir / 'splits.json')
            linker = EntityLinker(sum(splits.values(), []), graph.entities)
            self.linked_movie_ids = [linker.link(row['question'])[0] for row in rows]
        else:
            self.linked_movie_ids = [row['movie_id'] for row in rows]
        self.name_context = {row['question_id']: graph.fact_tail_ids(movie_id)
                             for row, movie_id in zip(rows, self.linked_movie_ids)}
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]; movie_id = self.linked_movie_ids[index]
        facts = self.graph.facts(movie_id) if movie_id else []
        return {'row': row, 'text': f"question: {row['question']} facts: " + '; '.join(item[0] for item in facts),
                'target': row['reference_answers'][0], 'vectors': [item[1] for item in facts if item[1] is not None]}


class Collator:
    def __init__(self, tokenizer, use_embeddings, input_max_length=384):
        if not isinstance(input_max_length, int) or isinstance(input_max_length, bool) or input_max_length < 1:
            raise ValueError("input_max_length must be a positive integer")
        self.tokenizer, self.use_embeddings = tokenizer, use_embeddings
        self.input_max_length = input_max_length
    def __call__(self, items):
        tokens = self.tokenizer([x['text'] for x in items], padding=True, truncation=True, max_length=self.input_max_length, return_tensors='pt')
        labels = self.tokenizer(text_target=[x['target'] for x in items], padding=True, truncation=True, max_length=96, return_tensors='pt')['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch = {'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask'], 'labels': labels,
                 'rows': [x['row'] for x in items]}
        if self.use_embeddings:
            dim = len(items[0]['vectors'][0]) if items[0]['vectors'] else 300
            count = max(max(len(x['vectors']), 1) for x in items)
            vectors = torch.zeros(len(items), count, dim); mask = torch.zeros(len(items), count, dtype=torch.long)
            for i, item in enumerate(items):
                if item['vectors']:
                    vectors[i, :len(item['vectors'])] = torch.tensor(np.asarray(item['vectors']), dtype=torch.float32); mask[i, :len(item['vectors'])] = 1
            batch.update(graph_vectors=vectors, graph_mask=mask)
        return batch


class GraphPrefixModel(nn.Module):
    def __init__(self, model_name, use_embeddings=True):
        super().__init__(); self.base = AutoModelForSeq2SeqLM.from_pretrained(model_name); self.use_embeddings = use_embeddings
        self.projection = nn.Linear(300, self.base.config.d_model) if use_embeddings else None
        self.norm = nn.LayerNorm(self.base.config.d_model) if use_embeddings else None

    def embed(self, input_ids, attention_mask, graph_vectors=None, graph_mask=None):
        text = self.base.get_input_embeddings()(input_ids)
        if not self.use_embeddings: return text, attention_mask
        prefix = self.norm(self.projection(graph_vectors)); return torch.cat([prefix, text], 1), torch.cat([graph_mask, attention_mask], 1)

    def forward(self, input_ids, attention_mask, labels, graph_vectors=None, graph_mask=None):
        embeddings, mask = self.embed(input_ids, attention_mask, graph_vectors, graph_mask)
        return self.base(inputs_embeds=embeddings, attention_mask=mask, labels=labels)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, graph_vectors=None, graph_mask=None):
        embeddings, mask = self.embed(input_ids, attention_mask, graph_vectors, graph_mask)
        return self.base.generate(inputs_embeds=embeddings, attention_mask=mask, max_new_tokens=96, num_beams=1, do_sample=False)


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def predictions(model, loader, tokenizer, entities, device):
    model.eval(); output = []
    name_index = AnswerEntityIndex(entities)
    for batch in loader:
        rows = batch.pop('rows'); batch = move(batch, device); batch.pop('labels', None)
        if device.type == 'cuda': torch.cuda.synchronize()
        started = time.perf_counter(); generated = model.generate(**batch)
        if device.type == 'cuda': torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000 / len(rows)
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for row, text in zip(rows, texts):
            output.append({'question_id': row['question_id'], 'question': row['question'], 'answer': text,
                           'answer_ids': entity_ids_from_text(text, entities, row['movie_id'], index=name_index,
                               preferred_ids=loader.dataset.name_context[row['question_id']]), 'latency_ms': elapsed})
    return output


@torch.no_grad()
def validation_outputs(model, loader, tokenizer, entities, device):
    model.eval(); output, losses = [], []
    name_index = AnswerEntityIndex(entities)
    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    for batch in loader:
        rows = batch.pop('rows'); batch = move(batch, device)
        context = torch.autocast('cuda', dtype=torch.bfloat16) if use_bf16 else nullcontext()
        with context:
            losses.append(float(model(**batch).loss.item()))
        labels = batch.pop('labels', None)
        if device.type == 'cuda': torch.cuda.synchronize()
        started = time.perf_counter(); generated = model.generate(**batch)
        if device.type == 'cuda': torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000 / len(rows)
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for row, text in zip(rows, texts):
            output.append({'question_id': row['question_id'], 'question': row['question'], 'answer': text,
                           'answer_ids': entity_ids_from_text(text, entities, row['movie_id'], index=name_index,
                               preferred_ids=loader.dataset.name_context[row['question_id']]), 'latency_ms': elapsed})
    return output, float(np.mean(losses))


def train(run_id, learning_rate, use_embeddings=True, epochs=5, smoke=False,
          batch_size=16, gradient_accumulation=1, input_max_length=384, eval_batch_size=None,
          model_name="google/flan-t5-small", data_dir=None, embedding_dir=None, initial_checkpoint=None):
    eval_batch_size = batch_size if eval_batch_size is None else eval_batch_size
    for name, value in (('epochs', epochs), ('batch_size', batch_size), ('gradient_accumulation', gradient_accumulation),
                        ('input_max_length', input_max_length), ('eval_batch_size', eval_batch_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    run = ROOT / 'runs' / run_id
    if run.exists() and (not run.is_dir() or any(run.iterdir())):
        raise FileExistsError(f'Refusing to overwrite a nonempty run: {run}')
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / 'status.json', {'status': 'running', 'stage': 'initializing', 'created_at': utc_now()})
    epoch, completed_epochs, optimizer_updates = 0, 0, 0
    try:
        seed_all(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        initial = (ROOT / initial_checkpoint) if initial_checkpoint is not None else None
        if initial is not None and (not initial.is_dir() or use_embeddings and not (initial / 'graph_prefix.pt').is_file()):
            raise ValueError('Initial checkpoint is missing or has no matching graph prefix; select the intended checkpoint')
        model_name = str(initial) if initial is not None else model_name
        selected_data = ROOT / (data_dir or 'data/processed')
        selected_embeddings = ROOT / (embedding_dir or 'models/transe')
        all_rows = read_jsonl(selected_data / 'qa.jsonl')
        if smoke:
            all_rows = [row for split in ('train', 'valid', 'test')
                        for row in [x for x in all_rows if x['split'] == split][:32]]
            epochs = 1
        data = {split: [row for row in all_rows if row['split'] == split] for split in ('train', 'valid', 'test')}
        if any(not rows for rows in data.values()):
            raise ValueError('Training, validation and test splits must all be nonempty')
        config = {'run_id': run_id, 'model': Path(model_name).name if Path(model_name).is_dir() else model_name,
                  'model_source_kind': 'local_directory' if Path(model_name).is_dir() else 'model_id',
                  'input_max_length': input_max_length, 'target_max_length': 96, 'eval_batch_size': eval_batch_size, 'learning_rate': learning_rate, 'epochs': epochs,
                  'batch_size': batch_size, 'gradient_accumulation': gradient_accumulation,
                  'effective_batch': batch_size * gradient_accumulation, 'use_embeddings': use_embeddings,
                  'seed': 42, 'shuffle_seed': 42, 'device': str(device), 'smoke': smoke,
                  'torch_threads': torch.get_num_threads(), 'generation_precision': 'float32',
                  'data_dir': str(data_dir or 'data/processed'), 'embedding_dir': str(embedding_dir or 'models/transe'),
                  'initial_checkpoint': str(initial_checkpoint) if initial_checkpoint is not None else None,
                  'optimizer_state_resumed': False,
                  'input_hashes': {name: sha256(selected_data / name) for name in ('entities.json', 'splits.json', 'qa.jsonl')},
                  'validation_entity_linking': True, 'selection_metric': 'validation entity_macro_f1',
                  'qa_by_split': {split: len(rows) for split, rows in data.items()},
                  'effective_batch_note': 'Maximum examples per optimizer update; the final accumulation window may be smaller.'}
        config['input_sha256'] = dict(config['input_hashes'])
        if (selected_data / 'triples.tsv').is_file():
            config['input_sha256']['triples.tsv'] = sha256(selected_data / 'triples.tsv')
        if use_embeddings:
            config['transe_sha256'] = {name: sha256(selected_embeddings / name)
                                      for name in ('mapping.json', 'embeddings.npz')}
        if Path(model_name).is_dir():
            config['pretrained_sha256'] = {name: sha256(Path(model_name) / name)
                                          for name in ('config.json', 'model.safetensors', 'tokenizer.json', 'graph_prefix.pt')
                                          if (Path(model_name) / name).is_file()}
        write_json(run / 'config.json', config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        graph = GraphData(use_embeddings, data_dir, embedding_dir)
        collator = Collator(tokenizer, use_embeddings, input_max_length)
        loaders = {split: DataLoader(QADataset(rows, graph, link_questions=split != 'train'),
                                    batch_size=batch_size if split == 'train' else eval_batch_size, shuffle=split == 'train', collate_fn=collator,
                                    generator=torch.Generator().manual_seed(42) if split == 'train' else None)
                   for split, rows in data.items()}
        model = GraphPrefixModel(model_name, use_embeddings).to(device)
        if initial is not None and use_embeddings:
            state = torch.load(initial / 'graph_prefix.pt', map_location=device, weights_only=True)
            model.projection.load_state_dict(state['projection']); model.norm.load_state_dict(state['norm'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
        batches_per_epoch = len(loaders['train'])
        updates_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation)
        updates = epochs * updates_per_epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(.1 * updates)), updates)
        config.update(training_batches_per_epoch=batches_per_epoch, optimizer_updates_per_epoch=updates_per_epoch,
                      planned_optimizer_updates=updates)
        write_json(run / 'config.json', config)
        use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
        config['training_precision'] = 'bf16 autocast' if use_bf16 else 'float32'
        write_json(run / 'config.json', config)
        scaler = torch.amp.GradScaler('cuda', enabled=False); history, best = [], (-1, None)
        training_started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter(); epoch_updates = 0
            model.train(); optimizer.zero_grad(); losses = []
            for step, batch in enumerate(loaders['train'], 1):
                batch.pop('rows'); batch = move(batch, device)
                context = torch.autocast('cuda', dtype=torch.bfloat16) if use_bf16 else nullcontext()
                with context: loss = model(**batch).loss
                if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss at epoch={epoch} step={step}')
                # Scale by the actual window length, including its partial tail.
                window_start = ((step - 1) // gradient_accumulation) * gradient_accumulation
                window_size = min(gradient_accumulation, batches_per_epoch - window_start)
                scaler.scale(loss / window_size).backward(); losses.append(float(loss.item()))
                if step % gradient_accumulation == 0 or step == batches_per_epoch:
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    previous_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                    if scaler.get_scale() >= previous_scale:
                        scheduler.step(); optimizer_updates += 1; epoch_updates += 1
                    optimizer.zero_grad()
                if step % 100 == 0 or step == batches_per_epoch:
                    write_json(run / 'status.json', {'status': 'running', 'stage': 'training',
                               'current_epoch': epoch, 'step': step, 'total_batches': batches_per_epoch,
                               'optimizer_updates': optimizer_updates, 'mean_loss': float(np.mean(losses)),
                               'updated_at': utc_now()})
            if device.type == 'cuda': torch.cuda.empty_cache()
            valid_predictions, valid_loss = validation_outputs(model, loaders['valid'], tokenizer, graph.entities, device)
            if device.type == 'cuda': torch.cuda.empty_cache()
            metrics = score_rows(data['valid'], valid_predictions, graph.entities)
            metrics.update(gold_total=len(data['valid']), prediction_coverage=100.0)
            record = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation_loss': valid_loss,
                      'epoch_seconds': time.perf_counter() - epoch_started, 'optimizer_updates': epoch_updates,
                      'optimizer_updates_total': optimizer_updates, **metrics}
            history.append(record); write_json(run / 'history.json', history)
            completed_epochs = epoch
            logging.info('run=%s epoch=%s train_loss=%.5f valid_loss=%.5f valid_f1=%.3f updates=%s seconds=%.1f',
                         run_id, epoch, record['train_loss'], record['validation_loss'], record['entity_macro_f1'],
                         epoch_updates, record['epoch_seconds'])
            if metrics['entity_macro_f1'] > best[0]:
                model.base.save_pretrained(run / 'checkpoint'); tokenizer.save_pretrained(run / 'checkpoint')
                if use_embeddings:
                    torch.save({'projection': model.projection.state_dict(), 'norm': model.norm.state_dict()},
                               run / 'checkpoint' / 'graph_prefix.pt')
                write_jsonl(run / 'predictions_valid.jsonl', valid_predictions)
                write_json(run / 'metrics_valid.json', metrics)
                best = (metrics['entity_macro_f1'], epoch)
            write_json(run / 'selection.json', {'best_valid_f1': best[0], 'best_epoch': best[1],
                                               'validation_entity_linking': True,
                                               'training_seconds': time.perf_counter() - training_started})
            write_json(run / 'status.json', {'status': 'running', 'stage': 'epoch_complete',
                                            'completed_epochs': epoch, 'optimizer_updates': optimizer_updates,
                                            'best_epoch': best[1], 'updated_at': utc_now()})
        # Restore the validation-selected checkpoint before the single test evaluation.
        # Release optimizer states before loading the checkpoint on a small GPU.
        del optimizer, scheduler
        model.base = AutoModelForSeq2SeqLM.from_pretrained(run / 'checkpoint').to(device)
        if use_embeddings:
            state = torch.load(run / 'checkpoint' / 'graph_prefix.pt', map_location=device, weights_only=True)
            model.projection.load_state_dict(state['projection']); model.norm.load_state_dict(state['norm'])
        write_jsonl(run / 'predictions_test.jsonl', predictions(model, loaders['test'], tokenizer, graph.entities, device))
        write_json(run / 'status.json', {'status': 'complete', 'stage': 'test_predictions_saved',
                                        'completed_epochs': epoch, 'optimizer_updates': optimizer_updates,
                                        'best_epoch': best[1], 'test_count': len(data['test']), 'completed_at': utc_now()})
    except Exception as error:
        write_json(run / 'status.json', {'status': 'failed', 'current_epoch': epoch, 'completed_epochs': completed_epochs,
                                        'optimizer_updates': optimizer_updates, 'error': str(error), 'failed_at': utc_now()})
        raise


def infer(run_id, input_path, output_name='predictions_robustness.jsonl'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); run = ROOT / 'runs' / run_id
    config = read_json(run / 'config.json'); use_embeddings = config['use_embeddings']
    graph = GraphData(use_embeddings, config.get('data_dir'), config.get('embedding_dir'))
    tokenizer = AutoTokenizer.from_pretrained(run / 'checkpoint'); model = GraphPrefixModel(str(run / 'checkpoint'), use_embeddings).to(device)
    if use_embeddings:
        state = torch.load(run / 'checkpoint' / 'graph_prefix.pt', map_location=device, weights_only=True)
        model.projection.load_state_dict(state['projection']); model.norm.load_state_dict(state['norm'])
    rows = read_jsonl(input_path); loader = DataLoader(QADataset(rows, graph, link_questions=True), batch_size=config.get('eval_batch_size', 16), collate_fn=Collator(tokenizer, use_embeddings, config.get('input_max_length', 384)))
    write_jsonl(run / output_name, predictions(model, loader, tokenizer, graph.entities, device))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True); parser.add_argument('--learning-rate', type=float, required=True)
    parser.add_argument('--no-embeddings', action='store_true'); parser.add_argument('--epochs', type=int, default=5); parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--batch-size', type=int, default=16, help='Training, validation and test batch size during training.')
    parser.add_argument('--gradient-accumulation', type=int, default=1, help='Training microbatches per optimizer update.')
    parser.add_argument('--input-max-length', type=int, default=384)
    parser.add_argument('--eval-batch-size', type=int)
    parser.add_argument('--model-name', default='google/flan-t5-small')
    parser.add_argument('--data-dir', type=Path, help='Independent frozen training data; does not modify the default dataset.')
    parser.add_argument('--embedding-dir', type=Path, help='TransE files matching the selected dataset.')
    parser.add_argument('--initial-checkpoint', type=Path, help='Warm-start model/tokenizer/prefix; creates a new optimizer, not an exact resume.')
    parser.add_argument('--infer-file'); parser.add_argument('--output-name', default='predictions_robustness.jsonl')
    args = parser.parse_args(); setup_logging(args.run_id)
    if args.infer_file and any(value is not None for value in (args.data_dir, args.embedding_dir, args.initial_checkpoint)):
        parser.error('Inference uses the dataset and embeddings recorded in the run config; path/checkpoint overrides are training-only')
    infer(args.run_id, args.infer_file, args.output_name) if args.infer_file else train(
        args.run_id, args.learning_rate, not args.no_embeddings, args.epochs, args.smoke,
        batch_size=args.batch_size, gradient_accumulation=args.gradient_accumulation,
        input_max_length=args.input_max_length, eval_batch_size=args.eval_batch_size, model_name=args.model_name,
        data_dir=args.data_dir, embedding_dir=args.embedding_dir, initial_checkpoint=args.initial_checkpoint)


