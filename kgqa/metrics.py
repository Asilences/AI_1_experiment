from __future__ import annotations

import argparse
from collections import defaultdict
import statistics

from rouge_score import rouge_scorer
import sacrebleu

from .common import REFUSAL, ROOT, read_json, read_jsonl, write_json
from .linker import normalize

class EntityNameIndex:
    """An explicit snapshot of entity names for one inference/evaluation pass."""

    def __init__(self, entities):
        name_map = {}
        for qid, entity in entities.items():
            label = normalize(entity.get('label', ''))
            if label:
                name_map.setdefault(label, {'labels': set(), 'aliases': set()})['labels'].add(qid)
            for raw_name in entity.get('aliases', []):
                name = normalize(raw_name)
                if name:
                    name_map.setdefault(name, {'labels': set(), 'aliases': set()})['aliases'].add(qid)
        self.names = [(name, name_map[name]) for name in sorted(name_map, key=len, reverse=True)]

    def match(self, text, excluded=None):
        padded = f' {normalize(text)} '
        found, occupied = set(), []
        for name, entry in self.names:
            token = f' {name} '; offset = 0
            while True:
                position = padded.find(token, offset)
                if position < 0: break
                span = (position + 1, position + 1 + len(name))
                if not any(span[0] < other[1] and other[0] < span[1] for other in occupied):
                    preferred = entry['labels'] or entry['aliases']
                    qids = {qid for qid in preferred if qid != excluded}
                    if qids:
                        found.update(qids)
                    occupied.append(span)
                offset = position + 1
        return sorted(found)


def entity_ids_from_text(text, entities, excluded=None, *, index=None):
    # A fresh default index sees dictionary edits; repeated callers can explicitly
    # reuse a snapshot instead of relying on a process-global identity cache.
    if index is None:
        index = EntityNameIndex(entities)
    return index.match(text, excluded)


def _index_rows(rows, name):
    by_id = {}
    for position, row in enumerate(rows, 1):
        question_id = row.get('question_id')
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f'{name} row {position} must have a nonempty question_id')
        if question_id in by_id:
            raise ValueError(f'Duplicate {name.lower()} question_id: {question_id!r}')
        by_id[question_id] = row
    return by_id


def _validate_alignment(gold_rows, predictions, *, allow_missing=False, allow_extra=False):
    if not gold_rows:
        raise ValueError('Cannot evaluate an empty gold set')
    gold_by_id = _index_rows(gold_rows, 'Gold')
    by_id = _index_rows(predictions, 'Prediction')
    if not by_id:
        raise ValueError('Cannot evaluate empty predictions')
    unknown = by_id.keys() - gold_by_id.keys()
    if unknown and not allow_extra:
        raise ValueError(f'Unknown prediction question IDs: {sorted(unknown)!r}')
    missing = gold_by_id.keys() - by_id.keys()
    if missing and not allow_missing:
        raise ValueError(f'Missing predictions for question IDs: {sorted(missing)!r}')
    return by_id


def score_rows(gold_rows, predictions, entities, *, allow_extra_predictions=False):
    by_id = _validate_alignment(gold_rows, predictions, allow_extra=allow_extra_predictions)
    hypotheses, references = [], []
    exact, f1s, refusal, latencies = [], [], [], []
    answerable_exact, answerable_f1s = [], []
    relation = defaultdict(list)
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_values = []
    name_index = None
    for gold in gold_rows:
        pred = by_id[gold['question_id']]
        if not gold.get('reference_answers'):
            raise ValueError(f"Gold row {gold['question_id']!r} must contain at least one reference answer")
        predicted_ids = pred.get('answer_ids')
        if predicted_ids is None:
            if name_index is None:
                name_index = EntityNameIndex(entities)
            predicted_ids = name_index.match(pred['answer'], gold['movie_id'])
        gold_set, predicted_set = set(gold['answer_ids']), set(predicted_ids)
        if not gold_set:
            correct_refusal = normalize(pred['answer']) == normalize(REFUSAL)
            item_exact = float(correct_refusal); item_f1 = float(correct_refusal); refusal.append(item_exact)
        else:
            item_exact = float(gold_set == predicted_set)
            precision = len(gold_set & predicted_set) / len(predicted_set) if predicted_set else 0
            recall = len(gold_set & predicted_set) / len(gold_set)
            item_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
            answerable_exact.append(item_exact); answerable_f1s.append(item_f1)
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
    result['answerable_count'] = len(answerable_exact); result['refusal_count'] = len(refusal)
    if answerable_exact:
        result['answerable_entity_exact'] = 100 * statistics.mean(answerable_exact)
        result['answerable_entity_macro_f1'] = 100 * statistics.mean(answerable_f1s)
    if refusal: result['refusal_accuracy'] = 100 * statistics.mean(refusal)
    if latencies:
        ordered = sorted(latencies); result['latency_mean_ms'] = statistics.mean(latencies); result['latency_p95_ms'] = ordered[min(len(ordered)-1, int(.95 * len(ordered)))]
    return result


def evaluate(run_id, split='test'):
    config_path = ROOT / 'runs' / run_id / 'config.json'
    config = read_json(config_path) if config_path.exists() else {}
    data = ROOT / config.get('data_dir', 'data/processed')
    predictions = read_jsonl(ROOT / 'runs' / run_id / f'predictions_{split}.jsonl')
    all_gold = [row for row in read_jsonl(data / 'qa.jsonl') if row['split'] == split]
    # Validate against the complete split before selecting an explicit smoke
    # subset, so filtering cannot conceal duplicate or unknown predictions.
    predicted_ids = _validate_alignment(all_gold, predictions, allow_missing=True).keys()
    gold = [row for row in all_gold if row['question_id'] in predicted_ids]
    if len(gold) != len(all_gold):
        if not config.get('smoke'):
            raise ValueError(f'Prediction coverage is incomplete: {len(gold)}/{len(all_gold)}')
    result = score_rows(gold, predictions, read_json(data / 'entities.json'))
    result['gold_total'] = len(all_gold); result['prediction_coverage'] = 100 * len(gold) / max(1, len(all_gold))
    write_json(ROOT / 'runs' / run_id / f'metrics_{split}.json', result); return result


def evaluate_robustness(run_id):
    config_path = ROOT / 'runs' / run_id / 'config.json'
    config = read_json(config_path) if config_path.exists() else {}
    data = ROOT / config.get('data_dir', 'data/processed')
    gold = read_jsonl(data / 'robustness.jsonl')
    predictions = read_jsonl(ROOT / 'runs' / run_id / 'predictions_robustness.jsonl')
    entities = read_json(data / 'entities.json')
    # Extra rows are only legitimate within a variant after the entire file has
    # passed strict coverage and uniqueness validation.
    _validate_alignment(gold, predictions)
    result = {}
    for variant in sorted({row['variant'] for row in gold}):
        subset = [row for row in gold if row['variant'] == variant]
        result[variant] = score_rows(subset, predictions, entities, allow_extra_predictions=True)
    write_json(ROOT / 'runs' / run_id / 'metrics_robustness.json', result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('run_id'); parser.add_argument('--split', default='test'); parser.add_argument('--robustness', action='store_true'); args = parser.parse_args()
    print(evaluate_robustness(args.run_id) if args.robustness else evaluate(args.run_id, args.split))


