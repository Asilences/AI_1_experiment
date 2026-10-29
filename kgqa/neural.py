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

from .common import RELATIONS, ROOT, read_json, read_jsonl, seed_all, setup_logging, utc_now, write_json, write_jsonl
from .metrics import EntityNameIndex, entity_ids_from_text, score_rows
from .linker import EntityLinker


class GraphData:
    def __init__(self, use_embeddings=True):
        self.entities = read_json(ROOT / 'data' / 'processed' / 'entities.json')
        self.use_embeddings = use_embeddings
        if use_embeddings:
            arrays = np.load(ROOT / 'models' / 'transe' / 'embeddings.npz')
            self.entity, self.relation = arrays['entity'], arrays['relation']
            mapping = read_json(ROOT / 'models' / 'transe' / 'mapping.json')
            self.entity_to_id, self.relation_to_id = mapping['entity_to_id'], mapping['relation_to_id']

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
        return rows[:24]


class QADataset(Dataset):
    def __init__(self, rows, graph, link_questions=False):
        self.rows, self.graph = rows, graph
        if link_questions:
            splits = read_json(ROOT / 'data' / 'processed' / 'splits.json')
            linker = EntityLinker(sum(splits.values(), []), graph.entities)
            self.linked_movie_ids = [linker.link(row['question'])[0] for row in rows]
        else:
            self.linked_movie_ids = [row['movie_id'] for row in rows]
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]; movie_id = self.linked_movie_ids[index]
        facts = self.graph.facts(movie_id) if movie_id else []
        return {'row': row, 'text': f"question: {row['question']} facts: " + '; '.join(item[0] for item in facts),
                'target': row['reference_answers'][0], 'vectors': [item[1] for item in facts if item[1] is not None]}


class Collator:
    def __init__(self, tokenizer, use_embeddings): self.tokenizer, self.use_embeddings = tokenizer, use_embeddings
    def __call__(self, items):
        tokens = self.tokenizer([x['text'] for x in items], padding=True, truncation=True, max_length=384, return_tensors='pt')
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
    name_index = EntityNameIndex(entities)
    for batch in loader:
        rows = batch.pop('rows'); batch = move(batch, device); batch.pop('labels', None)
        if device.type == 'cuda': torch.cuda.synchronize()
        started = time.perf_counter(); generated = model.generate(**batch)
        if device.type == 'cuda': torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000 / len(rows)
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for row, text in zip(rows, texts):
            output.append({'question_id': row['question_id'], 'question': row['question'], 'answer': text,
                           'answer_ids': entity_ids_from_text(text, entities, row['movie_id'], index=name_index), 'latency_ms': elapsed})
    return output


@torch.no_grad()
def validation_outputs(model, loader, tokenizer, entities, device):
    model.eval(); output, losses = [], []
    name_index = EntityNameIndex(entities)
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
                           'answer_ids': entity_ids_from_text(text, entities, row['movie_id'], index=name_index), 'latency_ms': elapsed})
    return output, float(np.mean(losses))


def train(run_id, learning_rate, use_embeddings=True, epochs=5, smoke=False,
          batch_size=16, gradient_accumulation=1):
    for name, value in (('epochs', epochs), ('batch_size', batch_size), ('gradient_accumulation', gradient_accumulation)):
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
        model_name = 'google/flan-t5-small'
        all_rows = read_jsonl(ROOT / 'data' / 'processed' / 'qa.jsonl')
        if smoke:
            all_rows = [row for split in ('train', 'valid', 'test')
                        for row in [x for x in all_rows if x['split'] == split][:32]]
            epochs = 1
        data = {split: [row for row in all_rows if row['split'] == split] for split in ('train', 'valid', 'test')}
        if any(not rows for rows in data.values()):
            raise ValueError('Training, validation and test splits must all be nonempty')
        config = {'run_id': run_id, 'model': model_name, 'learning_rate': learning_rate, 'epochs': epochs,
                  'batch_size': batch_size, 'gradient_accumulation': gradient_accumulation,
                  'effective_batch': batch_size * gradient_accumulation, 'use_embeddings': use_embeddings,
                  'seed': 42, 'device': str(device), 'smoke': smoke,
                  'validation_entity_linking': True, 'selection_metric': 'validation entity_macro_f1',
                  'qa_by_split': {split: len(rows) for split, rows in data.items()},
                  'effective_batch_note': 'Maximum examples per optimizer update; the final accumulation window may be smaller.'}
        write_json(run / 'config.json', config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        graph = GraphData(use_embeddings); collator = Collator(tokenizer, use_embeddings)
        loaders = {split: DataLoader(QADataset(rows, graph, link_questions=split != 'train'),
                                    batch_size=batch_size, shuffle=split == 'train', collate_fn=collator)
                   for split, rows in data.items()}
        model = GraphPrefixModel(model_name, use_embeddings).to(device)
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
            valid_predictions, valid_loss = validation_outputs(model, loaders['valid'], tokenizer, graph.entities, device)
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
    config = read_json(run / 'config.json'); use_embeddings = config['use_embeddings']; graph = GraphData(use_embeddings)
    tokenizer = AutoTokenizer.from_pretrained(run / 'checkpoint'); model = GraphPrefixModel(str(run / 'checkpoint'), use_embeddings).to(device)
    if use_embeddings:
        state = torch.load(run / 'checkpoint' / 'graph_prefix.pt', map_location=device, weights_only=True)
        model.projection.load_state_dict(state['projection']); model.norm.load_state_dict(state['norm'])
    rows = read_jsonl(input_path); loader = DataLoader(QADataset(rows, graph, link_questions=True), batch_size=16, collate_fn=Collator(tokenizer, use_embeddings))
    write_jsonl(run / output_name, predictions(model, loader, tokenizer, graph.entities, device))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True); parser.add_argument('--learning-rate', type=float, required=True)
    parser.add_argument('--no-embeddings', action='store_true'); parser.add_argument('--epochs', type=int, default=5); parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--batch-size', type=int, default=16, help='Training, validation and test batch size during training.')
    parser.add_argument('--gradient-accumulation', type=int, default=1, help='Training microbatches per optimizer update.')
    parser.add_argument('--infer-file'); parser.add_argument('--output-name', default='predictions_robustness.jsonl')
    args = parser.parse_args(); setup_logging(args.run_id)
    infer(args.run_id, args.infer_file, args.output_name) if args.infer_file else train(
        args.run_id, args.learning_rate, not args.no_embeddings, args.epochs, args.smoke,
        args.batch_size, args.gradient_accumulation)


