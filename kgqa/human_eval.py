from __future__ import annotations

import argparse
import csv
import random

from .common import ROOT, read_jsonl


def build(neural_run, limit=100):
    gold = {row['question_id']: row for row in read_jsonl(ROOT / 'data' / 'processed' / 'qa.jsonl') if row['split'] == 'test' and row['is_answerable']}
    systems = {'X': read_jsonl(ROOT / 'runs' / 'A0' / 'predictions_test.jsonl'), 'Y': read_jsonl(ROOT / 'runs' / neural_run / 'predictions_test.jsonl')}
    ids = sorted(gold)[:limit]; rows = []
    for qid in ids:
        order = ['X', 'Y']; random.Random(qid).shuffle(order)
        for position, system in enumerate(order, 1):
            prediction = next(row for row in systems[system] if row['question_id'] == qid)
            rows.append({'question_id': qid, 'question': gold[qid]['question'], 'reference': ' || '.join(gold[qid]['reference_answers']),
                         'answer_position': position, 'answer': prediction['answer'], 'accuracy_0_2': '', 'completeness_0_2': '',
                         'fluency_1_5': '', 'unsupported_claim_0_1': '', 'notes': '', '_system_key': system})
    output = ROOT / 'output'; output.mkdir(parents=True, exist_ok=True)
    visible = [key for key in rows[0] if not key.startswith('_')]
    with (output / 'human_evaluation.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=visible); writer.writeheader(); writer.writerows({key: row[key] for key in visible} for row in rows)
    with (output / 'human_evaluation_key.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['question_id', 'answer_position', 'system']); writer.writeheader()
        writer.writerows({'question_id': row['question_id'], 'answer_position': row['answer_position'], 'system': row['_system_key']} for row in rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--neural-run', required=True); parser.add_argument('--limit', type=int, default=100); args = parser.parse_args(); build(args.neural_run, args.limit)
