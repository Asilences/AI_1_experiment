from __future__ import annotations

import argparse
from collections import defaultdict
import re
import statistics

from rouge_score import rouge_scorer
import sacrebleu

from .common import REFUSAL, ROOT, read_json, read_jsonl, write_json
from .linker import normalize

_ENTITY_NAME_CACHE = {}


def entity_ids_from_text(text, entities, excluded=None):
    cache_key = id(entities)
    name_map = _ENTITY_NAME_CACHE.get(cache_key)
    if name_map is None:
        name_map = {}
        for qid, entity in entities.items():
            label = normalize(entity.get('label', ''))
            if label:
                name_map.setdefault(label, {'labels': set(), 'aliases': set()})['labels'].add(qid)
            for raw_name in entity.get('aliases', []):
                name = normalize(raw_name)
                if name:
                    name_map.setdefault(name, {'labels': set(), 'aliases': set()})['aliases'].add(qid)
        _ENTITY_NAME_CACHE.clear()
        _ENTITY_NAME_CACHE[cache_key] = name_map
    padded = f' {normalize(text)} '
    found, occupied = set(), []
    for name in sorted(name_map, key=len, reverse=True):
        token = f' {name} '; offset = 0
        while True:
            position = padded.find(token, offset)
            if position < 0: break
            span = (position + 1, position + 1 + len(name))
            if not any(span[0] < other[1] and other[0] < span[1] for other in occupied):
                entry = name_map[name]
                preferred = entry['labels'] or entry['aliases']
                qids = {qid for qid in preferred if qid != excluded}
                if qids:
                    found.update(qids)
                occupied.append(span)
            offset = position + 1
    return sorted(found)


def score_rows(gold_rows, predictions, entities):
    by_id = {row['question_id']: row for row in predictions}
    hypotheses, references = [], []
    exact, f1s, refusal, latencies = [], [], [], []
    relation = defaultdict(list)
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_values = []
    for gold in gold_rows:
        pred = by_id[gold['question_id']]
        predicted_ids = pred.get('answer_ids')
        if predicted_ids is None:
            predicted_ids = entity_ids_from_text(pred['answer'], entities, gold['movie_id'])
        gold_set, predicted_set = set(gold['answer_ids']), set(predicted_ids)
        if not gold_set:
            correct_refusal = normalize(pred['answer']) == normalize(REFUSAL)
            item_exact = float(correct_refusal); item_f1 = float(correct_refusal); refusal.append(item_exact)
        else:
            item_exact = float(gold_set == predicted_set)
            precision = len(gold_set & predicted_set) / len(predicted_set) if predicted_set else 0
            recall = len(gold_set & predicted_set) / len(gold_set)
            item_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        exact.append(item_exact); f1s.append(item_f1); relation[gold['relation_id']].append((item_exact, item_f1))
        hypotheses.append(pred['answer']); references.append(gold['reference_answers'])
        rouge_values.append(max(rouge.score(reference, pred['answer'])['rougeL'].fmeasure for reference in gold['reference_answers']))
        if 'latency_ms' in pred: latencies.append(float(pred['latency_ms']))
    reference_streams = [[refs[min(i, len(refs)-1)] for refs in references] for i in range(max(map(len, references)))]
    result = {'count': len(gold_rows), 'entity_exact': 100 * statistics.mean(exact), 'entity_macro_f1': 100 * statistics.mean(f1s),
              'bleu': sacrebleu.corpus_bleu(hypotheses, reference_streams, tokenize='13a').score,
              'rouge_l': 100 * statistics.mean(rouge_values),
              'by_relation': {key: {'exact': 100 * statistics.mean(v[0] for v in values),
                                    'f1': 100 * statistics.mean(v[1] for v in values), 'count': len(values)} for key, values in relation.items()}}
    if refusal: result['refusal_accuracy'] = 100 * statistics.mean(refusal)
    if latencies:
        ordered = sorted(latencies); result['latency_mean_ms'] = statistics.mean(latencies); result['latency_p95_ms'] = ordered[min(len(ordered)-1, int(.95 * len(ordered)))]
    return result


def evaluate(run_id, split='test'):
    predictions = read_jsonl(ROOT / 'runs' / run_id / f'predictions_{split}.jsonl')
    predicted_ids = {row['question_id'] for row in predictions}
    all_gold = [row for row in read_jsonl(ROOT / 'data' / 'processed' / 'qa.jsonl') if row['split'] == split]
    gold = [row for row in all_gold if row['question_id'] in predicted_ids]
    if len(gold) != len(all_gold):
        config = read_json(ROOT / 'runs' / run_id / 'config.json')
        if not config.get('smoke'):
            raise ValueError(f'Prediction coverage is incomplete: {len(gold)}/{len(all_gold)}')
    result = score_rows(gold, predictions, read_json(ROOT / 'data' / 'processed' / 'entities.json'))
    result['gold_total'] = len(all_gold); result['prediction_coverage'] = 100 * len(gold) / max(1, len(all_gold))
    write_json(ROOT / 'runs' / run_id / f'metrics_{split}.json', result); return result


def evaluate_robustness(run_id):
    gold = read_jsonl(ROOT / 'data' / 'processed' / 'robustness.jsonl')
    predictions = read_jsonl(ROOT / 'runs' / run_id / 'predictions_robustness.jsonl')
    entities = read_json(ROOT / 'data' / 'processed' / 'entities.json')
    result = {}
    for variant in sorted({row['variant'] for row in gold}):
        subset = [row for row in gold if row['variant'] == variant]
        result[variant] = score_rows(subset, predictions, entities)
    write_json(ROOT / 'runs' / run_id / 'metrics_robustness.json', result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('run_id'); parser.add_argument('--split', default='test'); parser.add_argument('--robustness', action='store_true'); args = parser.parse_args()
    print(evaluate_robustness(args.run_id) if args.robustness else evaluate(args.run_id, args.split))


