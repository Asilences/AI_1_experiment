from __future__ import annotations

from collections import Counter

from .common import REFUSAL, ROOT, read_json, read_jsonl, write_json
from .linker import EntityLinker, normalize


def analyze(run_id):
    entities = read_json(ROOT / 'data' / 'processed' / 'entities.json')
    splits = read_json(ROOT / 'data' / 'processed' / 'splits.json')
    linker = EntityLinker(sum(splits.values(), []), entities)
    gold_rows = [row for row in read_jsonl(ROOT / 'data' / 'processed' / 'qa.jsonl') if row['split'] == 'test']
    predictions = {row['question_id']: row for row in read_jsonl(ROOT / 'runs' / run_id / 'predictions_test.jsonl')}
    counts = Counter()
    examples = {}
    for gold in gold_rows:
        pred = predictions[gold['question_id']]
        gold_ids, pred_ids = set(gold['answer_ids']), set(pred.get('answer_ids', []))
        linked_id = pred.get('movie_id') if run_id == 'A0' else linker.link(gold['question'])[0]
        category = 'correct'
        if linked_id != gold['movie_id']:
            category = 'entity_linking_error'
        elif run_id == 'A0' and pred.get('relation_id') != gold['relation_id']:
            category = 'relation_error'
        elif not gold_ids and normalize(pred['answer']) != normalize(REFUSAL):
            category = 'failed_refusal'
        elif gold_ids and not pred_ids:
            category = 'missed_answer'
        elif pred_ids - gold_ids:
            category = 'unsupported_answer_entity'
        elif gold_ids - pred_ids:
            category = 'incomplete_answer'
        elif pred_ids != gold_ids:
            category = 'other_answer_error'
        counts[category] += 1
        if category != 'correct' and category not in examples:
            examples[category] = {'question_id': gold['question_id'], 'question': gold['question'], 'gold_answer_ids': sorted(gold_ids), 'predicted_answer_ids': sorted(pred_ids), 'answer': pred['answer']}
    total = len(gold_rows)
    return {'run_id': run_id, 'total': total, 'counts': dict(counts), 'percentages': {k: 100*v/total for k,v in counts.items()}, 'examples': examples}


def build(selected_run):
    result = {run: analyze(run) for run in ('A0', selected_run)}
    write_json(ROOT / 'output' / 'error_analysis.json', result)
    return result


if __name__ == '__main__':
    selected = read_json(ROOT / 'output' / 'selected_model.json')['selected_run']
    print(build(selected))
