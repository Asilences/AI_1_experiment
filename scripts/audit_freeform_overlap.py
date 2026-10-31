"""从已保存预测重算已见模板与其余测试题；不生成答案或重选模型。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kgqa.common import read_json, read_jsonl, sha256, write_json, write_jsonl
from kgqa.freeform_data import audit_template_overlap
from kgqa.metrics import score_rows


def run(data_dir, predictions_path, output_dir):
    data, output = Path(data_dir), Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new empty audit directory')
    rows, entities = read_jsonl(data / 'qa.jsonl'), read_json(data / 'entities.json')
    audit = audit_template_overlap(rows, entities)
    test = [r for r in rows if r['split'] == 'test']
    predictions = read_jsonl(predictions_path)
    # Full strict alignment must pass before selecting either subgroup.
    full = score_rows(test, predictions, entities)
    seen_ids = set(audit['splits']['test']['seen_question_ids'])
    grouped = {}
    for name, selected in [('seen_template', [r for r in test if r['question_id'] in seen_ids]),
                           ('remaining_templates', [r for r in test if r['question_id'] not in seen_ids])]:
        qids = {r['question_id'] for r in selected}
        predicted = [r for r in predictions if r['question_id'] in qids]
        grouped[name] = score_rows(selected, predicted, entities) if selected else {'count': 0}
        write_jsonl(output / f'{name}_question_ids.jsonl', [{'question_id': q['question_id']} for q in selected])
    result = {'evaluation': 'rescore_saved_predictions_no_generation_no_training_no_selection',
              'data_sha256': {n: sha256(data / n) for n in ['qa.jsonl', 'entities.json', 'splits.json']},
              'predictions_sha256': sha256(predictions_path), 'template_audit': audit,
              'full_mixed_test': full, 'groups': grouped,
              'scope': 'Remaining templates have no normalized exact training-template match; not an independent human benchmark.'}
    write_json(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data_dir, args.predictions, args.output_dir)
    print({k: v['entity_macro_f1'] for k, v in result['groups'].items() if v['count']})


if __name__ == '__main__':
    main()
