"""Reproduce the personal A0 experiment without replacing shared models or runs.

Only the confidence threshold is tuned, using validation questions. Test repeats
measure deterministic inference and timing; they are not independent training runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from importlib import metadata
import json
from pathlib import Path
import pickle
import platform
import re
import statistics
import sys
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from .common import RELATIONS, ROOT, read_json, read_jsonl, sha256, utc_now, write_json, write_jsonl
from .linker import EntityLinker
from .metrics import score_rows
from .pipeline import PipelineQA, mask_question

THRESHOLD_GRID = tuple(round(.30 + .05 * index, 2) for index in range(12))
# Declared before acquiring/reading the new test data. Stringent settings expose
# false refusals; these diagnostic results never alter validation selection.
FIXED_THRESHOLDS = (.30, .70, .90, .99)
REPEATS = 3
SPLITS = ('train', 'valid', 'test')


class FittedClassifier:
    """Adapter shared by threshold selection and PipelineQA inference."""

    def __init__(self, features, estimator):
        self.features, self.estimator = features, estimator
        self.classes_ = estimator.classes_

    def predict_proba(self, questions):
        return self.estimator.predict_proba(self.features.transform(questions))


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _same_gold(left, right):
    return (left['movie_id'] == right['movie_id'] and left['relation_id'] == right['relation_id']
            and set(left['answer_ids']) == set(right['answer_ids'])
            and left['is_answerable'] == right['is_answerable']
            and set(left['reference_answers']) == set(right['reference_answers']))


def _validate_rows(rows, entities, split_for, *, robustness=False, original_ids=()):
    seen = set()
    for position, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f'Question row {position} must be an object')
        question_id = row.get('question_id')
        if not _nonempty_string(question_id):
            raise ValueError(f'Question row {position} requires a nonempty question_id')
        if question_id in seen or question_id in original_ids:
            raise ValueError(f'Duplicate question_id: {question_id!r}')
        seen.add(question_id)
        movie_id, split = row.get('movie_id'), row.get('split')
        if movie_id not in split_for or split_for[movie_id] != split:
            raise ValueError(f'{question_id}: movie must belong to exactly its declared split')
        if robustness and (split != 'test' or not _nonempty_string(row.get('variant'))):
            raise ValueError(f'{question_id}: robustness rows require test split and a variant')
        if not _nonempty_string(row.get('question')):
            raise ValueError(f'{question_id}: question must be a nonempty string')
        relation = row.get('relation_id')
        if relation not in {*RELATIONS, 'UNKNOWN'}:
            raise ValueError(f'{question_id}: unsupported relation_id {relation!r}')
        answers = row.get('answer_ids')
        if not isinstance(answers, list) or any(not _nonempty_string(value) for value in answers):
            raise ValueError(f'{question_id}: answer_ids must be a list of nonempty strings')
        if len(answers) != len(set(answers)):
            raise ValueError(f'{question_id}: duplicate answer IDs')
        if not isinstance(row.get('is_answerable'), bool) or row['is_answerable'] != bool(answers):
            raise ValueError(f'{question_id}: is_answerable must agree with answer_ids')
        expected = [] if relation == 'UNKNOWN' else entities[movie_id]['facts'].get(relation, [])
        if set(answers) != set(expected):
            raise ValueError(f'{question_id}: gold answer IDs do not match movie facts')
        references = row.get('reference_answers')
        if not isinstance(references, list) or not references or any(not _nonempty_string(value) for value in references):
            raise ValueError(f'{question_id}: reference_answers must contain nonempty strings')
    return seen


def validate_inputs(qa, entities, splits, robustness=None):
    """Reject leakage, ambiguous IDs and incomplete facts before fitting anything."""
    if not isinstance(splits, dict) or set(splits) != set(SPLITS):
        raise ValueError('splits.json must contain exactly train, valid and test')
    if not isinstance(entities, dict) or not entities:
        raise ValueError('entities.json must contain a nonempty object')
    split_for = {}
    for split in SPLITS:
        movies = splits[split]
        if not isinstance(movies, list) or not movies:
            raise ValueError(f'{split}: movie split must be a nonempty list')
        for movie_id in movies:
            if not _nonempty_string(movie_id) or movie_id in split_for:
                raise ValueError(f'Movie splits must be disjoint with unique IDs: {movie_id!r}')
            split_for[movie_id] = split
            movie = entities.get(movie_id)
            if not isinstance(movie, dict) or not _nonempty_string(movie.get('label')):
                raise ValueError(f'{movie_id}: missing movie label')
            facts = movie.get('facts')
            if not isinstance(facts, dict) or not facts:
                raise ValueError(f'{movie_id}: missing facts object')
            aliases = movie.get('aliases', [])
            if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
                raise ValueError(f'{movie_id}: aliases must be a list of strings')
            for relation, tails in facts.items():
                if relation not in RELATIONS or not isinstance(tails, list):
                    raise ValueError(f'{movie_id}: invalid facts relation or answer list')
                if any(not _nonempty_string(tail) for tail in tails) or len(tails) != len(set(tails)):
                    raise ValueError(f'{movie_id}: invalid or duplicate fact tail IDs')
                for tail in tails:
                    entity = entities.get(tail)
                    if not isinstance(entity, dict) or not _nonempty_string(entity.get('label')):
                        raise ValueError(f'{movie_id}: missing fact tail label for {tail!r}')
    ids = _validate_rows(qa, entities, split_for)
    grouped = {split: [row for row in qa if row['split'] == split] for split in SPLITS}
    if any(not grouped[split] for split in SPLITS):
        raise ValueError('train, valid and test QA splits must each be nonempty')
    if len({row['relation_id'] for row in grouped['train']}) < 2:
        raise ValueError('Training split must contain at least two relation classes')
    if robustness is not None:
        if not robustness:
            raise ValueError('An existing robustness.jsonl must not be empty')
        _validate_rows(robustness, entities, split_for, robustness=True, original_ids=ids)
        test_by_id = {row['question_id']: row for row in grouped['test']}
        for row in robustness:
            if 'source_question_id' in row:
                source_id = row['source_question_id']
                if source_id not in test_by_id:
                    raise ValueError(f"{row['question_id']}: robustness source must be a test question")
                if not _same_gold(row, test_by_id[source_id]):
                    raise ValueError(f"{row['question_id']}: robustness source movie, relation, gold IDs and references must be preserved")
    return grouped


def _validate_pairs(robustness, manifest):
    """Resolve manifest pairs before running; refuse mismatched denominators."""
    if manifest is None:
        return {}
    if not isinstance(manifest, dict) or not isinstance(manifest.get('comparisons', {}), dict):
        raise ValueError('Robustness manifest comparisons must be an object')
    by_id = {row['question_id']: row for row in robustness or []}
    result = {}
    for variant, comparison in sorted(manifest.get('comparisons', {}).items()):
        if not isinstance(comparison, dict):
            raise ValueError(f'{variant}: robustness manifest comparison must be an object')
        if 'pairs' not in comparison:
            continue
        pairs = comparison['pairs']
        if not isinstance(pairs, list):
            raise ValueError(f'{variant}: robustness manifest pairs must be a list')
        selected, seen_clean, seen_variant = [], set(), set()
        for pair in pairs:
            if not isinstance(pair, dict):
                raise ValueError(f'{variant}: every robustness pair must be an object')
            clean_id, variant_id = pair.get('clean_question_id'), pair.get('variant_question_id')
            if not _nonempty_string(clean_id) or not _nonempty_string(variant_id):
                raise ValueError(f'{variant}: pair IDs must be nonempty strings')
            if clean_id not in by_id or variant_id not in by_id:
                raise ValueError(f'{variant}: pair references missing robustness rows')
            clean, changed = by_id[clean_id], by_id[variant_id]
            if clean['variant'] != 'clean' or changed['variant'] != variant:
                raise ValueError(f'{variant}: pair references the wrong variants')
            if clean_id in seen_clean or variant_id in seen_variant:
                raise ValueError(f'{variant}: duplicate robustness pair IDs')
            if not _same_gold(clean, changed):
                raise ValueError(f'{variant}: paired clean and variant gold must agree')
            if ('source_question_id' in clean and 'source_question_id' in changed
                    and clean['source_question_id'] != changed['source_question_id']):
                raise ValueError(f'{variant}: paired rows must have the same source question')
            if changed.get('paired_clean_question_id', clean_id) != clean_id:
                raise ValueError(f'{variant}: pair disagrees with paired_clean_question_id')
            selected.append((clean, changed)); seen_clean.add(clean_id); seen_variant.add(variant_id)
        expected = {row['question_id'] for row in robustness or [] if row['variant'] == variant}
        if seen_variant != expected:
            raise ValueError(f'{variant}: manifest pairs must cover the entire variant pool')
        if comparison.get('matched_count', len(selected)) != len(selected):
            raise ValueError(f'{variant}: manifest matched_count disagrees with pairs')
        result[variant] = selected
    return result


def select_threshold(validation_rows, unfiltered_predictions):
    """Choose exact-set accuracy on valid only; equal scores keep the lower value."""
    if not validation_rows or len(validation_rows) != len(unfiltered_predictions):
        raise ValueError('Threshold selection requires aligned nonempty validation predictions')
    records = []
    for threshold in THRESHOLD_GRID:
        correct = 0
        for gold, prediction in zip(validation_rows, unfiltered_predictions):
            if gold['split'] != 'valid' or gold['question_id'] != prediction['question_id']:
                raise ValueError('Threshold selection accepts aligned validation rows only')
            answer_ids = prediction['answer_ids']
            if prediction.get('relation_confidence', 0) < threshold:
                answer_ids = []
            correct += set(answer_ids) == set(gold['answer_ids'])
        records.append({'threshold': threshold, 'correct': correct, 'count': len(validation_rows),
                        'entity_exact': 100 * correct / len(validation_rows)})
    best = max(records, key=lambda row: row['entity_exact'])
    return best['threshold'], records


def _predict(system, rows):
    started = time.perf_counter()
    predictions = [{**system.predict(row['question']), 'question_id': row['question_id']} for row in rows]
    return predictions, time.perf_counter() - started


def _prediction_digest(predictions):
    stable = [{key: value for key, value in row.items() if key != 'latency_ms'} for row in predictions]
    return hashlib.sha256(json.dumps(stable, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def _environment():
    packages = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'sacrebleu', 'rouge-score'):
        packages[name] = metadata.version(name)
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'machine': platform.machine(), 'processor': platform.processor(), 'packages': packages}


def _summary(trials):
    numeric = [key for key, value in trials[0]['metrics'].items()
               if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return {'metric_means': {key: statistics.mean(trial['metrics'][key] for trial in trials) for key in numeric},
            'metric_population_std': {key: statistics.pstdev(trial['metrics'][key] for trial in trials) for key in numeric},
            'forward_duration_mean_seconds': statistics.mean(trial['forward_duration_seconds'] for trial in trials),
            'forward_duration_population_std_seconds': statistics.pstdev(trial['forward_duration_seconds'] for trial in trials),
            'deterministic_predictions': len({trial['prediction_sha256_without_latency'] for trial in trials}) == 1}


def _write_sensitivity_csv(directory, settings, config):
    rows = []
    for setting in settings:
        row = {key: setting[key] for key in ('setting', 'threshold', 'role', 'count_per_repeat', 'deterministic_predictions')}
        row.update(run_id=config['run_id'], repeats=REPEATS, total_test_forward_passes=len(settings) * REPEATS,
                   training_runs=1, threshold_selection_split='valid', selected_threshold=config['threshold'],
                   forward_duration_seconds_mean=setting['forward_duration_mean_seconds'],
                   forward_duration_seconds_population_std=setting['forward_duration_population_std_seconds'])
        row.update({f'{key}_mean': value for key, value in setting['metric_means'].items()})
        row.update({f'{key}_population_std': value for key, value in setting['metric_population_std'].items()})
        rows.append(row)
    fixed = ['run_id', 'setting', 'threshold', 'role', 'count_per_repeat', 'repeats', 'training_runs',
             'total_test_forward_passes', 'threshold_selection_split', 'selected_threshold', 'deterministic_predictions']
    fields = fixed + sorted(set().union(*(row.keys() for row in rows)) - set(fixed))
    with (directory / 'sensitivity_comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def run(data_dir=ROOT / 'data' / 'processed', output_dir=ROOT / 'runs', run_id='A0_personal_20261019'):
    """Fit one A0 model and save a new, self-contained validation/test record."""
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise ValueError('run_id must be a single safe directory name')
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    directory = output_dir / run_id
    if directory.exists():
        raise FileExistsError(f'Refusing to overwrite existing run: {directory}')
    paths = {name: data_dir / name for name in ('qa.jsonl', 'entities.json', 'splits.json')}
    for optional in ('manifest.json', 'triples.tsv', 'robustness.jsonl', 'robustness_manifest.json'):
        if (data_dir / optional).exists():
            paths[optional] = data_dir / optional
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'Required processed input is missing: {path}')
    input_files = {name: {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
                   for name, path in paths.items()}
    qa, entities, splits = read_jsonl(paths['qa.jsonl']), read_json(paths['entities.json']), read_json(paths['splits.json'])
    robustness = read_jsonl(paths['robustness.jsonl']) if 'robustness.jsonl' in paths else None
    grouped = validate_inputs(qa, entities, splits, robustness)
    robust_manifest = read_json(paths['robustness_manifest.json']) if 'robustness_manifest.json' in paths else None
    matched_pairs = _validate_pairs(robustness, robust_manifest)
    movies = [movie for split in SPLITS for movie in splits[split]]
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'status.json', {'status': 'running', 'created_at': utc_now()})
    manifest = {'input_files': input_files, 'entity_count': len(entities), 'question_count': len(qa),
                'movies_by_split': {key: len(value) for key, value in splits.items()},
                'questions_by_split': {key: len(value) for key, value in grouped.items()},
                'relation_counts_by_split': {key: dict(sorted(Counter(row['relation_id'] for row in value).items()))
                                             for key, value in grouped.items()},
                'movie_splits_disjoint': True, 'question_ids_unique': True,
                'robustness': {'available': robustness is not None, 'count': len(robustness or []),
                               'variant_counts': dict(sorted(Counter(row['variant'] for row in robustness or []).items()))}}
    write_json(directory / 'data_manifest.json', manifest)
    write_json(directory / 'environment.json', _environment())
    source_names = ('common.py', 'linker.py', 'pipeline.py', 'metrics.py', 'personal_experiment.py')
    write_json(directory / 'source_hashes.json', {name: sha256(Path(__file__).parent / name) for name in source_names})

    linker = EntityLinker(movies, entities)
    features = FeatureUnion([('word', TfidfVectorizer(ngram_range=(1, 2))),
                             ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5)))])
    started = time.perf_counter()
    train_rows = grouped['train']
    matrix = features.fit_transform([mask_question(row, entities, linker) for row in train_rows])
    estimator = LogisticRegression(max_iter=300, random_state=42, solver='liblinear').fit(
        matrix, [row['relation_id'] for row in train_rows])
    training_seconds = time.perf_counter() - started
    classifier = FittedClassifier(features, estimator)
    system = PipelineQA(entities, movies, classifier, threshold=0)
    validation_predictions, validation_seconds = _predict(system, grouped['valid'])
    threshold, validation_records = select_threshold(grouped['valid'], validation_predictions)
    write_jsonl(directory / 'validation_candidates.jsonl', validation_predictions)
    write_json(directory / 'validation_thresholds.json', {
        'selection_split': 'valid', 'objective': 'entity_exact', 'tie_break': 'lowest threshold',
        'selected_threshold': threshold, 'count': len(grouped['valid']), 'records': validation_records,
        'unfiltered_forward_duration_seconds': validation_seconds})
    with (directory / 'model.pkl').open('xb') as stream:
        pickle.dump({'features': features, 'estimator': estimator, 'threshold': threshold}, stream)
    config = {'system': 'pipeline', 'experiment': 'A0 personal reproduction', 'run_id': run_id,
              'created_at': utc_now(), 'data_dir': str(data_dir), 'threshold': threshold,
              'selection_split': 'valid', 'threshold_grid': list(THRESHOLD_GRID),
              'fixed_test_thresholds': list(FIXED_THRESHOLDS), 'repeats': REPEATS,
              'fixed_threshold_rationale': 'Prespecified before reading the new test data; .90 and .99 test the '
                                           'false-refusal tradeoff at stringent confidence thresholds. '
                                           'Fixed test diagnostics never choose the baseline threshold.',
              'training_runs': 1, 'repeat_type': 'deterministic forward passes of the same fitted model; no retraining',
              'feature_fit_split': 'train', 'random_state': 42, 'max_iter': 300, 'solver': 'liblinear',
              'features': {'word_ngram_range': [1, 2], 'char_wb_ngram_range': [3, 5]},
              'training_duration_seconds': training_seconds, 'smoke': False,
              'model_path': str(directory / 'model.pkl'), 'test_selection': False,
              'timing_scope': 'CPU end-to-end PipelineQA.predict per question; includes linking, classification and fact lookup',
              'timing_note': 'No warmup is excluded. Machine load and cache state may affect these repeated timings.'}
    write_json(directory / 'config.json', config)

    settings = [('baseline', threshold)] + [(f'threshold_{int(value * 100):03d}', value) for value in FIXED_THRESHOLDS]
    sensitivity = []
    for name, value in settings:
        system.threshold = value
        trials = []
        for repeat in range(1, REPEATS + 1):
            predictions, duration = _predict(system, grouped['test'])
            metrics = score_rows(grouped['test'], predictions, entities)
            metrics.update(gold_total=len(grouped['test']), prediction_coverage=100.0)
            relative = Path('sensitivity') / name / f'repeat_{repeat}'
            write_jsonl(directory / relative / 'predictions_test.jsonl', predictions)
            write_json(directory / relative / 'metrics_test.json', metrics)
            trials.append({'repeat': repeat, 'prediction_path': (relative / 'predictions_test.jsonl').as_posix(),
                           'metrics_path': (relative / 'metrics_test.json').as_posix(), 'metrics': metrics,
                           'forward_duration_seconds': duration,
                           'prediction_sha256_without_latency': _prediction_digest(predictions)})
            if name == 'baseline' and repeat == 1:
                write_jsonl(directory / 'predictions_test.jsonl', predictions)
                write_json(directory / 'metrics_test.json', metrics)
        sensitivity.append({'setting': name, 'threshold': value,
                            'role': 'validation_selected_baseline' if name == 'baseline' else 'prespecified_test_diagnostic',
                            'count_per_repeat': len(grouped['test']), 'repeats': trials, **_summary(trials)})
    write_json(directory / 'sensitivity_summary.json', {
        'selection_split': 'valid', 'selection_metric': 'entity_exact', 'selected_threshold': threshold,
        'training_runs': 1, 'repeat_type': config['repeat_type'],
        'note': 'All fixed thresholds are reported. Test metrics never choose or change the baseline threshold. '
                'Population standard deviations describe timing/repeat variation, not training uncertainty.',
        'settings': sensitivity})
    _write_sensitivity_csv(directory, sensitivity, config)

    if robustness is not None:
        system.threshold = threshold
        predictions, duration = _predict(system, robustness)
        write_jsonl(directory / 'predictions_robustness.jsonl', predictions)
        by_id = {row['question_id']: row for row in predictions}
        variant_metrics = {}
        for variant in sorted({row['variant'] for row in robustness}):
            gold = [row for row in robustness if row['variant'] == variant]
            metrics = score_rows(gold, [by_id[row['question_id']] for row in gold], entities)
            metrics.update(gold_total=len(gold), prediction_coverage=100.0)
            variant_metrics[variant] = metrics
        write_json(directory / 'metrics_robustness.json', variant_metrics)
        matched_metrics = {}
        for variant, pairs in matched_pairs.items():
            entry = {'matched_count': len(pairs), 'all_clean_count': variant_metrics.get('clean', {}).get('count', 0),
                     'variant_count': variant_metrics.get(variant, {}).get('count', 0),
                     'pairs': [{'clean_question_id': clean['question_id'], 'variant_question_id': changed['question_id']}
                               for clean, changed in pairs]}
            if pairs:
                clean_rows, variant_rows = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
                clean_score = score_rows(clean_rows, [by_id[row['question_id']] for row in clean_rows], entities)
                variant_score = score_rows(variant_rows, [by_id[row['question_id']] for row in variant_rows], entities)
                entry.update(matched_clean_metrics=clean_score, matched_variant_metrics=variant_score,
                             entity_exact_delta_percentage_points=variant_score['entity_exact'] - clean_score['entity_exact'],
                             entity_macro_f1_delta_percentage_points=variant_score['entity_macro_f1'] - clean_score['entity_macro_f1'])
            else:
                entry['status'] = 'unavailable: no matched pairs'
            matched_metrics[variant] = entry
        write_json(directory / 'metrics_robustness_matched.json', matched_metrics)
        write_json(directory / 'robustness_summary.json', {
            'threshold': threshold, 'total_count': len(robustness),
            'variant_counts': {key: value['count'] for key, value in variant_metrics.items()},
            'matched_comparison_counts': {key: len(value) for key, value in matched_pairs.items()},
            'forward_duration_seconds': duration,
            'note': 'Each variant has its own denominator; variant pools can differ in size.'})
    for name, path in paths.items():
        if sha256(path) != input_files[name]['sha256']:
            raise RuntimeError(f'Input changed during the experiment: {path}. Run is incomplete.')
    write_json(directory / 'status.json', {'status': 'complete', 'completed_at': utc_now(),
                                          'training_runs': 1, 'test_forward_passes': len(settings) * REPEATS})
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'processed')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs')
    parser.add_argument('--run-id', default='A0_personal_20261019')
    args = parser.parse_args()
    print(run(args.data_dir, args.output_dir, args.run_id))


if __name__ == '__main__':
    main()
