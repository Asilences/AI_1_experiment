"""Run a small, explicitly developmental acceptance comparison without training.

The questions are authored during development. Their factual references come
from a frozen KB, never from model answers. This is not a held-out benchmark or
human evaluation, and its scores must not be presented as generalization scores.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

from .common import REFUSAL, RELATIONS, ROOT, read_json, read_jsonl, sha256, utc_now, write_json, write_jsonl


DEFAULT_CHALLENGE = ROOT / 'benchmarks/trainfree_challenge_v1.jsonl'


def build_gold(cases, entities, movie_ids):
    """Resolve declared factual questions against the input KB before inference."""
    movies = set(movie_ids)
    seen, rows = set(), []
    for case in cases:
        item = dict(case)
        if not item.get('question_id') or item['question_id'] in seen:
            raise ValueError('Challenge question IDs must be present and unique')
        seen.add(item['question_id'])
        if not isinstance(item.get('question'), str) or not item['question'].strip():
            raise ValueError('Challenge questions must be nonempty strings')
        status = item['expected_status']
        if status not in {'ready', 'unanswerable', 'needs_clarification'}:
            raise ValueError(f'Unknown expected status: {status}')
        item['answer_ids'] = []
        if status == 'ready':
            movie, relation = item['movie_id'], item['relation_id']
            if movie not in movies or relation not in RELATIONS:
                raise ValueError(f'Unsupported reference binding: {item["question_id"]}')
            tails = entities[movie].get('facts', {}).get(relation, [])
            if not tails or any(qid not in entities for qid in tails):
                raise ValueError(f'Missing reference facts: {item["question_id"]}')
            item['answer_ids'] = sorted(set(tails))
            item['reference_facts'] = [[movie, relation, qid] for qid in item['answer_ids']]
        else:
            item['reference_facts'] = []
        item['reference_source'] = 'frozen KB entities.json facts; authored task/status labels'
        rows.append(item)
    return rows


def entity_f1(expected, predicted):
    expected, predicted = set(expected), set(predicted)
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0
    return 2 * len(expected & predicted) / (len(expected) + len(predicted))


def refusal_flag(result):
    # Raw model refusal may be repaired by a separately labelled tool fallback.
    # Score the final response, preserving the raw text only as evidence.
    if 'refused' in result:
        return bool(result['refused'])
    return any(
        result.get(key, '').strip() == REFUSAL for key in ('answer', 'original_answer')
        if isinstance(result.get(key, ''), str))


def summarize(rows):
    """Percentages use declared case counts, never prediction-dependent scaling."""
    supported = [row for row in rows if row['expected_status'] == 'ready']
    impossible = [row for row in rows if row['expected_status'] == 'unanswerable']
    clarification = [row for row in rows if row['expected_status'] == 'needs_clarification']
    bounded = [row for row in rows if row['expected_status'] != 'ready']

    def rate(group, key):
        return 100 * sum(bool(row[key]) for row in group) / len(group) if group else None

    latency = sorted(row['elapsed_ms'] for row in rows)
    return {
        'count': len(rows),
        'supported_count': len(supported),
        'unanswerable_count': len(impossible),
        'clarification_count': len(clarification),
        'supported_entity_f1': 100 * statistics.mean(row['entity_f1'] for row in supported) if supported else None,
        'supported_entity_exact': rate(supported, 'entity_exact'),
        'supported_movie_entity_exact': rate(supported, 'movie_entity_exact'),
        'supported_movie_relation_entity_exact': rate(supported, 'grounded_exact'),
        'supported_answer_coverage': rate(supported, 'answered'),
        'unanswerable_refusal_accuracy': rate(impossible, 'refused'),
        'clarification_status_accuracy': rate(clarification, 'status_exact'),
        'nonanswerable_unsafe_answer_rate': rate(bounded, 'answered'),
        'accepted_status_accuracy': rate(rows, 'status_exact'),
        'answer_coverage': rate(rows, 'answered'),
        'tool_fallback_rate': rate(rows, 'tool_fallback'),
        'mean_latency_ms': statistics.mean(latency) if latency else None,
        'p95_latency_ms': latency[math.ceil(.95 * len(latency)) - 1] if latency else None,
        'status_counts': dict(Counter(row['observed_status'] for row in rows)),
    }


def evaluate_cases(cases, predict, decode_ids):
    records = []
    for case in cases:
        started = time.perf_counter()
        response = predict(case['question'])
        elapsed_ms = (time.perf_counter() - started) * 1000
        result = response.get('result', response)
        interpretation = response.get('interpretation', {})
        refused = refusal_flag(result)
        status = interpretation.get('status') or ('unanswerable' if refused else 'ready')
        predicted_ids = sorted(set(result['answer_ids'] if 'answer_ids' in result else decode_ids(result)))
        actual_movie = result.get('movie_id')
        relation = result.get('relation', result.get('relation_id', result.get('input_relation')))
        exact = set(predicted_ids) == set(case['answer_ids'])
        action = result.get('action', result.get('guard', {}).get('action', '')
                            if isinstance(result.get('guard'), dict) else result.get('guard', ''))
        record = {
            **case, 'observed_status': status, 'predicted_movie_id': actual_movie,
            'predicted_relation_id': relation, 'predicted_answer_ids': predicted_ids,
            'refused': refused, 'answered': not refused and status == 'ready',
            'status_exact': status in case.get('acceptable_statuses', [case['expected_status']]),
            'entity_f1': entity_f1(case['answer_ids'], predicted_ids), 'entity_exact': exact,
            'movie_entity_exact': exact and actual_movie == case.get('movie_id'),
            'grounded_exact': exact and actual_movie == case.get('movie_id') and relation == case.get('relation_id'),
            'tool_fallback': bool(result.get('fallback_used')) or 'fallback' in str(action),
            'elapsed_ms': elapsed_ms, 'response': response,
        }
        records.append(record)
    by_category = defaultdict(list)
    for record in records:
        by_category[record['category']].append(record)
    metrics = summarize(records)
    metrics['by_category'] = {key: summarize(value) for key, value in sorted(by_category.items())}
    languages = defaultdict(list)
    for record in records:
        languages[record.get('language', 'unspecified')].append(record)
    metrics['by_language'] = {key: summarize(value) for key, value in sorted(languages.items())}
    return records, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--challenge', type=Path, default=DEFAULT_CHALLENGE)
    parser.add_argument('--systems', nargs='+', choices=('a0', 'neural'), default=['a0'])
    parser.add_argument('--data-dir', type=Path, help='A0-only custom dataset; paired --model-path is required.')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if bool(args.data_dir) != bool(args.model_path):
        parser.error('--data-dir and --model-path must be provided together')
    if args.data_dir and 'neural' in args.systems:
        parser.error('Custom A0 paths cannot be mixed with default neural assets')
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite evidence directory: {output}')

    from .demo import Demo
    from .guarded import GuardedQA
    from .metrics import AnswerEntityIndex

    started = time.perf_counter()
    demo = Demo(load_neural='neural' in args.systems, data_dir=args.data_dir, model_path=args.model_path)
    initialization_seconds = time.perf_counter() - started
    guard = GuardedQA(demo)
    index = AnswerEntityIndex(demo.entities)
    cases = build_gold(read_jsonl(args.challenge), demo.entities, demo.movie_ids)

    def decode_ids(result):
        movie = result.get('movie_id')
        # Only facts actually visible to the inference model resolve homonyms.
        preferred = demo.graph.fact_tail_ids(movie) if movie and hasattr(demo, 'graph') else ()
        text = result.get('original_answer', result.get('answer', ''))
        return index.match(text, movie, preferred_ids=preferred)

    data = args.data_dir or ROOT / 'data/processed'
    split_lookup = {qid: split for split, qids in read_json(data / 'splits.json').items() for qid in qids}
    bindings = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha256(path)
                for path in (data / 'entities.json', data / 'splits.json', data / 'manifest.json',
                             args.challenge.resolve()) if path.is_file()}
    model_files = [args.model_path or ROOT / 'models/pipeline/model.pkl']
    if 'neural' in args.systems:
        run = ROOT / 'runs' / demo.neural_run
        model_files += [run / 'config.json', *sorted((run / 'checkpoint').glob('*'))]
        model_files += [ROOT / 'output/selected_model.json', ROOT / 'models/transe/embeddings.npz',
                        ROOT / 'models/transe/mapping.json']
    for path in model_files:
        if path.is_file():
            bindings[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = sha256(path)
    sources = ('trainfree_experiment.py', 'guarded.py', 'freeform.py', 'demo.py', 'metrics.py', 'chinese.py', 'linker.py',
               'common.py', 'neural.py', 'neural_preflight.py', 'locales/zh_entities.json')
    source_hashes = {f'kgqa/{name}': sha256(ROOT / 'kgqa' / name) for name in sources}
    environment = {'python': sys.version, 'python_executable': sys.executable, 'platform': platform.platform(),
                   'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
                   'mkl_num_threads': os.environ.get('MKL_NUM_THREADS'),
                   'torch': None, 'device': None, 'torch_num_threads': None}
    if 'neural' in args.systems:
        # A0-only evaluation must remain runnable without neural dependencies.
        import torch
        environment.update(torch=torch.__version__, device=str(demo.device),
                           torch_num_threads=torch.get_num_threads())
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'resolved_references.jsonl', cases)
    write_json(output / 'manifest.json', {
        'created_at': utc_now(), 'status': 'running', 'evaluation_type': 'development_acceptance',
        'project_root': str(ROOT), 'data_dir': str(data.resolve()),
        'scope': 'Program-authored bilingual cases; four-property fixed-KB task, not human evaluation or held-out generalization.',
        'training_performed': False, 'test_used_for_selection': False,
        'model_weights_updated': False, 'movie_count': len(demo.movie_ids),
        'relations': list(RELATIONS), 'systems': args.systems,
        'neural_run': demo.neural_run, 'initialization_seconds': initialization_seconds,
        'environment': environment,
        'supported_question_movie_split_counts': dict(Counter(split_lookup[row['movie_id']]
                                                     for row in cases if row['expected_status'] == 'ready')),
        'latency_note': 'Sequential wall time per query; model initialization excluded; cold first-query costs retained; no warmup.',
        'bindings': bindings, 'source_sha256': source_hashes,
        'scoring_note': 'All rates use declared case counts; raw systems cannot emit clarification state. Tool fallback is a hybrid system.',
        'refusal_detection_note': 'Final explicit refused flag or exact canonical refusal text; raw paraphrased refusals/clarifications require manual review. Unsafe-answer rate is an automatic non-refusal proxy, not human factual scoring.',
        'relation_metric_note': 'Movie+relation+entity exact requires an explicitly emitted relation; raw neural responses may not report one. Use movie+entity exact for that comparison.',
    })
    all_metrics = {}
    for system in args.systems:
        for mode in ('raw', 'guarded', 'hybrid'):
            predict = (getattr(demo, f'answer_{system}') if mode == 'raw' else
                       lambda question, s=system, m=mode: guard.answer(question, system=s, fallback=m == 'hybrid'))
            records, metrics = evaluate_cases(cases, predict, decode_ids)
            key = f'{system}_{mode}'
            write_jsonl(output / f'{key}_predictions.jsonl', records)
            write_json(output / f'{key}_metrics.json', metrics)
            all_metrics[key] = metrics
            print(f'{key}: n={metrics["count"]}, supported_F1={metrics["supported_entity_f1"]:.2f}, '
                  f'refusal={metrics["unanswerable_refusal_accuracy"]:.2f}, '
                  f'unsafe_answer={metrics["nonanswerable_unsafe_answer_rate"]:.2f}', flush=True)
    write_json(output / 'summary.json', all_metrics)
    manifest = read_json(output / 'manifest.json')
    after = {f'kgqa/{name}': sha256(ROOT / 'kgqa' / name) for name in sources}
    input_after = {name: sha256(Path(name) if Path(name).is_absolute() else ROOT / name)
                   for name in bindings}
    manifest.update(completed_at=utc_now(), source_sha256_after=after,
                    source_unchanged=source_hashes == after,
                    input_sha256_after=input_after, inputs_unchanged=bindings == input_after, status='complete')
    write_json(output / 'manifest.json', manifest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
