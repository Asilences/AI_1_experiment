"""Evaluate a frozen versioned A0 model without retraining or replacing its data."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .common import REFUSAL, read_json, read_jsonl, sha256, utc_now, write_json, write_jsonl
from .linker import normalize
from .metrics import score_rows
from .personal_experiment import validate_inputs, _validate_pairs
from .pipeline import PipelineQA, load_model
from .robustness import build


def paired_scores(rows, predictions, manifest, entities):
    pairs = _validate_pairs(rows, manifest)
    result = {}
    for variant, selected in pairs.items():
        if not selected:
            result[variant] = {'matched_count': 0}
            continue
        clean, changed = zip(*selected)
        result[variant] = {
            'matched_count': len(selected),
            'clean': score_rows(list(clean), predictions, entities, allow_extra_predictions=True),
            'variant': score_rows(list(changed), predictions, entities, allow_extra_predictions=True),
        }
    return result


def error_breakdown(rows, predictions):
    by_id = {row['question_id']: row for row in predictions}
    counts, examples = Counter(), []
    for gold in rows:
        pred = by_id[gold['question_id']]
        correct = (set(pred['answer_ids']) == set(gold['answer_ids']) if gold['answer_ids']
                   else normalize(pred['answer']) == normalize(REFUSAL))
        if correct:
            continue
        if not pred['movie_id']:
            reason = 'movie_not_linked'
        elif pred['movie_id'] != gold['movie_id']:
            reason = 'wrong_movie'
        elif pred['relation_id'] != gold['relation_id']:
            reason = 'wrong_relation'
        elif not pred['answer_ids'] and gold['answer_ids']:
            reason = 'false_refusal'
        else:
            reason = 'answer_mismatch'
        counts[reason] += 1
        if len(examples) < 5:
            examples.append({'question_id': gold['question_id'], 'question': gold['question'],
                             'reason': reason, 'gold_movie_id': gold['movie_id'],
                             'predicted_movie_id': pred['movie_id'], 'answer': pred['answer']})
    return {'incorrect_rows': sum(counts.values()), 'by_first_failing_stage': dict(counts), 'examples': examples}


def run(data_dir, model_path, output_dir, *, limit=100, seed=42, split='test'):
    if split not in {'valid', 'test'}:
        raise ValueError('Evaluation split must be valid or test')
    data, model_path, output = map(Path, (data_dir, model_path, output_dir))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite evaluation output: {output}')
    hashes = {name: sha256(data / name) for name in ('qa.jsonl', 'entities.json', 'splits.json')}
    model_hash = sha256(model_path)
    config = read_json(model_path.parent / 'config.json')
    if config.get('input_hashes') != hashes or config.get('model_sha256') != model_hash:
        raise ValueError('Model and data hashes do not match the frozen configuration')
    entities, splits = read_json(data / 'entities.json'), read_json(data / 'splits.json')
    qa = read_jsonl(data / 'qa.jsonl')
    grouped = validate_inputs(qa, entities, splits)
    classifier, threshold = load_model(model_path)
    if threshold != config['selected_threshold']:
        raise ValueError('Model threshold differs from its frozen configuration')
    system = PipelineQA(entities, sum(splits.values(), []), classifier, threshold)
    robust = build(limit, seed, data, output / 'questions', split=split)
    if split == 'test':
        validate_inputs(qa, entities, splits, robust)
    else:
        source_rows = {row['question_id']: row for row in grouped['valid']}
        for row in robust:
            source = source_rows.get(row['source_question_id'])
            if source is None or row['split'] != 'valid' or any(
                    row[key] != source[key] for key in ('movie_id', 'relation_id', 'answer_ids', 'reference_answers')):
                raise ValueError('Validation perturbations must preserve their validation source and gold fields')
    manifest = read_json(output / 'questions' / 'robustness_manifest.json')

    def predict(rows):
        return [{**system.predict(row['question']), 'question_id': row['question_id']} for row in rows]

    test_predictions = predict(grouped[split])
    robust_predictions = predict(robust)
    by_variant = {}
    for variant in sorted({row['variant'] for row in robust}):
        subset = [row for row in robust if row['variant'] == variant]
        by_variant[variant] = score_rows(subset, robust_predictions, entities, allow_extra_predictions=True)
    previous_path = model_path.parent / 'predictions_test.jsonl'
    regression = None
    if split == 'test' and previous_path.is_file():
        previous = {row['question_id']: row for row in read_jsonl(previous_path)}
        regression = {
            'archived_predictions_sha256': sha256(previous_path),
            'same_question_ids': set(previous) == {row['question_id'] for row in test_predictions},
            'same_answers_and_entity_sets': all(
                row['question_id'] in previous and row['answer'] == previous[row['question_id']]['answer']
                and set(row['answer_ids']) == set(previous[row['question_id']]['answer_ids'])
                for row in test_predictions),
        }
    if hashes != {name: sha256(data / name) for name in hashes} or model_hash != sha256(model_path):
        raise RuntimeError('Frozen data/model changed during evaluation')
    result = {
        'evaluated_at': utc_now(), 'scope': 'Frozen A0 re-evaluation and seeded matched robustness diagnostics; no retraining or test-based tuning.',
        'movie_count': len(sum(splits.values(), [])), 'entity_count': len(entities), 'qa_count': len(qa),
        'input_hashes': hashes, 'model_sha256': model_hash, 'selected_threshold': threshold,
        'seed': seed, 'requested_answerable_limit': limit,
        'evaluation_split': split, split: score_rows(grouped[split], test_predictions, entities),
        'robustness': by_variant, 'paired_comparisons': paired_scores(robust, robust_predictions, manifest, entities),
        'robustness_errors': {kind: error_breakdown([row for row in robust if row['variant'] == kind], robust_predictions)
                              for kind in by_variant},
        'archived_test_regression': regression, 'frozen_inputs_unchanged': True,
        'source_sha256': {name: sha256(Path(__file__).with_name(name))
                          for name in ('expanded_evaluation.py', 'robustness.py', 'pipeline.py', 'linker.py', 'metrics.py')},
        'interpretation': 'Fixed templates on a visible knowledge base are separate from perturbation performance. Alias comparisons use their matched clean subset.',
    }
    write_jsonl(output / 'predictions_robustness.jsonl', robust_predictions)
    write_json(output / 'summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--model-path', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', choices=('valid', 'test'), default='test')
    args = parser.parse_args(argv)
    result = run(args.data_dir, args.model_path, args.output_dir, limit=args.limit, seed=args.seed, split=args.split)
    print(json.dumps({'movies': result['movie_count'], args.split: result[args.split]['entity_exact'],
                      'robustness': {kind: {'count': metrics['count'], 'entity_exact': metrics['entity_exact'],
                                           'entity_macro_f1': metrics['entity_macro_f1']}
                                     for kind, metrics in result['robustness'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
