"""Recheck the frozen A0 release without modifying its data, model or scores.

Run from the repository root: python verification/2026-10-31/a0_audit/recompute.py
The standalone script adds only the repository root to sys.path.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from kgqa.common import REFUSAL, answer_text, read_json, read_jsonl, sha256, write_json
from kgqa.metrics import score_rows
from kgqa.pipeline import PipelineQA, load_model
from kgqa.prepare import QUESTIONS, UNKNOWN_QUESTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('recomputed.json'))
    parser.add_argument('--skip-live', action='store_true', help='Recompute saved predictions only.')
    args = parser.parse_args()
    data = ROOT / 'data/processed/movies_5000_20261029'
    model_dir = ROOT / 'models/pipeline_5000_20261029'
    paths = [*(data / name for name in ('qa.jsonl', 'entities.json', 'splits.json')),
             *(model_dir / name for name in ('model.pkl', 'config.json', 'predictions_test.jsonl', 'metrics_test.json'))]
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}
    qa, entities, splits = (read_jsonl(data / 'qa.jsonl'), read_json(data / 'entities.json'),
                            read_json(data / 'splits.json'))
    test = [r for r in qa if r['split'] == 'test']
    predictions = read_jsonl(model_dir / 'predictions_test.jsonl')
    by_id = {p['question_id']: p for p in predictions}
    recomputed = score_rows(test, predictions, entities)
    saved = read_json(model_dir / 'metrics_test.json')
    split_sets = {key: set(value) for key, value in splits.items()}
    patterns = []
    for relation, templates in {**QUESTIONS, 'UNKNOWN': UNKNOWN_QUESTIONS}.items():
        for template in templates:
            patterns.append((relation, template.replace('{title}', 'MOVIE_TITLE'),
                             re.compile(re.escape(template).replace(re.escape('{title}'), '.+'))))
    templates = {key: set() for key in splits}
    unmatched = []
    for row in qa:
        matched = [canonical for relation, canonical, pattern in patterns
                   if relation == row['relation_id'] and pattern.fullmatch(row['question'])]
        if matched:
            templates[row['split']].update(matched)
        else:
            unmatched.append(row['question_id'])
    answerable = [r for r in test if r['answer_ids']]
    gold_from_kb = sum(set(r['answer_ids']) == set(entities[r['movie_id']]['facts'].get(r['relation_id'], []))
                       for r in answerable)
    reference_from_template = sum(
        r['reference_answers'][0] == answer_text(
            entities[r['movie_id']]['label'], r['relation_id'], [entities[q]['label'] for q in r['answer_ids']])
        for r in answerable)
    thresholds = []
    for threshold in (.30, .70, .90, .99, .995, .998, .999):
        altered, false_refusals = [], 0
        for row in test:
            pred = deepcopy(by_id[row['question_id']])
            if pred.get('relation_confidence', 0) < threshold:
                pred.update(answer_ids=[], answer=REFUSAL)
                false_refusals += bool(row['answer_ids'])
            altered.append(pred)
        metric = score_rows(test, altered, entities)
        thresholds.append({'threshold': threshold, 'entity_exact': metric['entity_exact'],
                           'answerable_entity_exact': metric['answerable_entity_exact'],
                           'false_refusals': false_refusals, 'count': len(test),
                           'type': 'post_diagnostic_replay_of_saved_confidences_not_retraining'})
    live = {'executed': False}
    if not args.skip_live:
        classifier, threshold = load_model(model_dir / 'model.pkl')
        system = PipelineQA(entities, sum(splits.values(), []), classifier, threshold)
        replay = [dict(system.predict(row['question']), question_id=row['question_id']) for row in test]
        metrics = score_rows(test, replay, entities)
        live = {'executed': True, 'model_threshold': threshold, 'count': len(replay), 'metrics': metrics,
                'answer_id_differences_from_saved': sum(set(p['answer_ids']) != set(by_id[p['question_id']]['answer_ids']) for p in replay),
                'text_differences_from_saved': sum(p['answer'] != by_id[p['question_id']]['answer'] for p in replay)}
    values = [by_id[r['question_id']]['relation_confidence'] for r in answerable]
    result = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'scope': 'Published frozen 5000-movie V1 template QA; not V2 or a new held-out generalization experiment.',
        'input_sha256': hashes, 'metrics_from_saved_predictions': recomputed,
        'reported_metric_differences': {key: recomputed[key] - saved[key]
                                        for key in ('entity_exact', 'entity_macro_f1', 'bleu', 'rouge_l')},
        'unique_question_ids': len({r['question_id'] for r in qa}) == len(qa),
        'movie_split_sizes': {k: len(v) for k, v in split_sets.items()},
        'movie_split_intersections': {f'{a}/{b}': len(split_sets[a] & split_sets[b])
                                      for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]},
        'raw_question_train_test_overlap': len({r['question'] for r in qa if r['split'] == 'train'} & {r['question'] for r in test}),
        'templates_by_split': {k: sorted(v) for k, v in templates.items()},
        'test_templates_seen_in_train': len(templates['test'] & templates['train']),
        'unmatched_template_rows': unmatched, 'test_questions': len(test), 'answerable_questions': len(answerable),
        'gold_answer_sets_present_in_visible_kb': gold_from_kb,
        'gold_first_references_equal_pipeline_formatter': reference_from_template,
        'prediction_text_matches_a_reference': sum(p['answer'] in row['reference_answers']
                                                  for row in test for p in [by_id[row['question_id']]]),
        'answerable_confidence_min': min(values), 'answerable_confidence_max': max(values),
        'answerable_distinct_confidences': len(set(round(v, 12) for v in values)),
        'link_status_counts': dict(Counter(p['link_status'] for p in predictions)),
        'threshold_sensitivity_from_saved_predictions': thresholds, 'live_model_replay': live,
        'inputs_unchanged': all(sha256(ROOT / name) == value for name, value in hashes.items()),
        'note': 'Scores are percentages. A0 uses returned IDs; BLEU/ROUGE independently compare text to dataset references. High-threshold diagnostics use already-inspected test confidences and cannot select the default threshold.'}
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('test_questions', 'answerable_questions', 'test_templates_seen_in_train',
          'gold_answer_sets_present_in_visible_kb', 'prediction_text_matches_a_reference',
          'answerable_confidence_min', 'answerable_confidence_max', 'reported_metric_differences', 'inputs_unchanged')}, ensure_ascii=False))
    print(json.dumps({'thresholds': thresholds, 'live': live}, ensure_ascii=False))


if __name__ == '__main__':
    main()
