"""Compare D0/P0 on program-authored development acceptance questions.

This is not an independent test set or human semantic/factual evaluation.
References are declared plans executed against the frozen catalogue, never P0
outputs. Text factuality and fluency require separate human review.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kgqa.common import read_jsonl, sha256, utc_now, write_json, write_jsonl
from kgqa.planning import validate_plan

DEFAULT_CASES = ROOT / 'benchmarks/semantic_dev_v1.jsonl'
BLOCKED = {'unsupported', 'needs_confirmation', 'needs_clarification', 'not_found'}


def build_gold(cases, explorer):
    """Bind authored references to KB facts before calling either system."""
    seen, gold = set(), []
    for case in cases:
        row = dict(case)
        qid = row.get('question_id')
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError('Development question IDs must be present and unique')
        seen.add(qid)
        if not isinstance(row.get('question'), str) or not row['question'].strip():
            raise ValueError('Development questions must be nonempty text')
        if row.get('language') not in ('zh', 'en') or not isinstance(row.get('category'), str):
            raise ValueError('Development cases need a language and category')
        status = row['expected_status']
        accepted = row.get('acceptable_statuses', [status])
        if not isinstance(accepted, list) or status not in accepted or any(not isinstance(s, str) for s in accepted):
            raise ValueError('acceptable_statuses must include the declared expected status')
        plan_status = row.get('reference_plan_status', 'ready')
        plan = {'schema': 'kgqa-plan/1', 'language': row['language'], 'status': plan_status,
                'movie_ids': row['expected_movie_ids'], 'properties': row['expected_properties'],
                'operation': row['expected_operation'], 'unhandled_constraints': row.get('unhandled_constraints', []),
                'clarification': row.get('clarification'), 'limit': row.get('limit', 30)}
        plan = validate_plan(plan, plan['movie_ids'], catalogue_movie_ids=explorer.movies)
        execution = None
        if plan_status == 'ready':
            execution = explorer.execute(plan, allowed_movie_ids=plan['movie_ids'], question=row['question'])
            if execution['status'] != status:
                raise ValueError(f'Reference status disagrees with the frozen KB: {qid}: '
                                 f'{status} != {execution["status"]}')
        elif status not in BLOCKED:
            raise ValueError('Non-ready references must declare a blocked/clarification status')
        row.update(expected_plan=plan, reference_execution=execution,
                   reference_source='Program-authored intent/status; facts from frozen Explorer.execute before inference',
                   evaluation_type='program_authored_development_acceptance',
                   generated_text_human_reviewed=False)
        gold.append(row)
    return gold


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def execution_signature(result):
    """Compare typed tool facts and scope, not generated prose or language labels."""
    if not isinstance(result, dict):
        return None
    facts = []
    for item in result.get('evidence', []):
        facts.append({key: item.get(key) for key in
                      ('statement_id', 'movie_id', 'property', 'type', 'snaktype', 'value', 'qualifiers')})
    items = [{key: value for key, value in item.items() if key not in {'label', 'text'}}
             for item in result.get('items', [])]
    return {'status': result.get('status'), 'evidence': sorted(_canonical(f) for f in facts),
            'items': sorted(_canonical(item) for item in items),
            'missing': sorted(_canonical(item) for item in result.get('missing', [])),
            'total': result.get('total', 0), 'offset': result.get('offset', 0),
            'limit': result.get('limit', 30), 'next_offset': result.get('next_offset'),
            'comparison': result.get('comparison')}


def declared_plan_match(observed, expected):
    if not isinstance(observed, dict):
        return False
    if any(not isinstance(observed.get(key), list) or
           any(not isinstance(value, str) for value in observed[key])
           for key in ('movie_ids', 'properties')):
        return False
    return (set(observed.get('movie_ids', [])) == set(expected['movie_ids']) and
            set(observed.get('properties', [])) == set(expected['properties']) and
            observed.get('operation') == expected['operation'])


def score_case(case, response, elapsed_ms):
    plan = response.get('plan', {})
    execution = response.get('execution', response)
    evaluable = case['reference_execution'] is not None
    plan_match = declared_plan_match(plan, case['expected_plan'])
    status_match = response.get('status') in case.get('acceptable_statuses', [case['expected_status']])
    tool_match = (execution_signature(execution) == execution_signature(case['reference_execution'])) if evaluable else None
    return {**case, 'observed_status': response.get('status'), 'plan_evaluable': evaluable,
            'declared_plan_match': plan_match if evaluable else None,
            'tool_result_match': tool_match, 'status_match': status_match,
            'structured_task_match': status_match and (plan_match and tool_match if evaluable else
                not response.get('evidence') and not response.get('items') and
                not (isinstance(execution, dict) and (execution.get('evidence') or execution.get('items')))),
            'fallback_used': bool(response.get('fallback_used')), 'elapsed_ms': elapsed_ms,
            'response': response}


def summarize(rows):
    def fraction(group, field):
        return {'numerator': sum(bool(row[field]) for row in group), 'denominator': len(group),
                'percent': 100 * sum(bool(row[field]) for row in group) / len(group) if group else None}
    evaluable = [row for row in rows if row['plan_evaluable']]
    latency = sorted(row['elapsed_ms'] for row in rows)
    return {'count': len(rows), 'plan_evaluable_count': len(evaluable),
            'declared_plan_match': fraction(evaluable, 'declared_plan_match'),
            'tool_result_match': fraction(evaluable, 'tool_result_match'),
            'status_match': fraction(rows, 'status_match'),
            'structured_task_match': fraction(rows, 'structured_task_match'),
            'fallback_rate': fraction(rows, 'fallback_used'),
            'mean_latency_ms': statistics.mean(latency) if latency else None,
            'p95_latency_ms': latency[math.ceil(.95 * len(latency)) - 1] if latency else None,
            'status_counts': dict(Counter(row['observed_status'] for row in rows)),
            'text_semantics_human_reviewed': False}


def evaluate(gold, predict, on_record=None):
    records = []
    for case in gold:
        started = time.perf_counter()
        response = predict(case)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if not isinstance(response, dict):
            raise TypeError('A prediction must return a response object')
        record = score_case(case, response, elapsed_ms)
        records.append(record)
        if on_record is not None:
            on_record(record)
    summary = summarize(records)
    for field in ('category', 'language'):
        grouped = defaultdict(list)
        for row in records:
            grouped[row[field]].append(row)
        summary['by_' + field] = {name: summarize(group) for name, group in sorted(grouped.items())}
    return records, summary


def prepare_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'Refusing to overwrite evidence directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _source_files():
    return [Path(__file__).resolve(), *sorted((ROOT / 'kgqa').rglob('*.py')),
            *sorted((ROOT / 'kgqa/locales').glob('*.json'))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog-dir', type=Path)
    parser.add_argument('--cases', type=Path, default=DEFAULT_CASES)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--systems', nargs='+', choices=('D0', 'P0'), default=['D0', 'P0'])
    args = parser.parse_args(argv)
    output = prepare_output(args.output_dir)
    from kgqa.explorer import Explorer
    from kgqa.v2_data import DEST
    directory = (args.catalog_dir or DEST).resolve()
    started = time.perf_counter()
    explorer = Explorer(directory)
    gold = build_gold(read_jsonl(args.cases), explorer)
    sources = {str(path): sha256(path) for path in _source_files()}
    bindings = {str(path): sha256(path) for path in [args.cases.resolve(), directory/'manifest.json',
                *(directory/name for name in explorer.manifest['files'])] if path.is_file()}
    manifest = {'created_at': utc_now(), 'status': 'running', 'evaluation_type': 'program_authored_development_acceptance',
                'scope': 'Program-authored development cases; not independent testing or human semantic/factual evaluation.',
                'case_count': len(gold),
                'declared_case_movie_count': len({qid for row in gold for qid in row['expected_movie_ids']}),
                'training_performed': False, 'model_weights_updated': False, 'test_used_for_selection': False,
                'movie_count': len(explorer.movies), 'statement_count': len(explorer.rows),
                'catalogue_dir': str(directory), 'systems': args.systems,
                'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform()},
                'source_sha256': sources, 'input_sha256': bindings,
                'scoring_note': 'Declared plan matching and typed tool-result matching do not establish generated text correctness.',
                'latency_note': 'Sequential per-question wall time; initialization excluded; no warmup; original outputs retained.',
                'initialization_seconds': time.perf_counter()-started}
    write_jsonl(output/'resolved_references.jsonl', gold)
    write_json(output/'manifest.json', manifest)
    metrics = {}
    try:
        semantic = None
        if 'P0' in args.systems:
            from kgqa.semantic_qa import SemanticQA
            semantic = SemanticQA(explorer)
        for system in args.systems:
            def predict(case):
                method = explorer.answer if system == 'D0' else semantic.answer
                return method(case['question'], confirmed=case.get('confirmed_movie_ids'),
                              offset=0, limit=case.get('limit', 30))
            with (output/(system+'_predictions.jsonl')).open('w', encoding='utf-8') as stream:
                def save_record(record):
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                    stream.flush()
                records, summary = evaluate(gold, predict, on_record=save_record)
            metrics[system] = summary
            write_json(output/'summary.json', metrics)
            if system == 'P0' and records:
                manifest['client_metadata'] = records[0]['response'].get('model')
        manifest.update(status='complete', completed_at=utc_now(),
                        source_unchanged=(set(sources) == {str(path) for path in _source_files()} and
                            all(sha256(Path(path)) == digest for path, digest in sources.items())),
                        inputs_unchanged=all(sha256(Path(path)) == digest for path, digest in bindings.items()))
    except Exception as error:
        manifest.update(status='failed', completed_at=utc_now(), error_type=type(error).__name__)
        write_json(output/'manifest.json', manifest)
        raise
    write_json(output/'manifest.json', manifest)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
