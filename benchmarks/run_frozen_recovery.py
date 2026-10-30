"""Evaluate the existing name recovery as a separate frozen-prediction method."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--split', required=True, choices=('valid', 'test', 'robustness'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--validation-report', type=Path)
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    if args.run_id not in {'B0', 'B1', 'B2', 'B3', 'C0'}:
        raise ValueError('Expected a frozen suite run ID')
    if output.exists():
        raise FileExistsError('Use a new output directory; previous evidence is never overwritten')
    if output.is_relative_to(project):
        raise ValueError('Additional-method evidence must stay outside the frozen project')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ.update(TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(project))
    from kgqa.expanded_suite import verify_frozen
    from kgqa.linker import EntityLinker, normalize
    from kgqa.common import REFUSAL, read_jsonl, write_json, write_jsonl
    from kgqa.metrics import AnswerEntityIndex, score_rows
    from kgqa.name_recovery import NameRecovery
    from kgqa.neural_preflight import FACT_LIMIT, TOKENIZER_FILES, _facts, tokenizer_identity
    from transformers import AutoTokenizer

    work, data, run = project.parent, project / 'data/processed', project / 'runs' / args.run_id
    verify_frozen(work)
    manifest, config, status = read(data / 'manifest.json'), read(run / 'config.json'), read(run / 'status.json')
    if manifest['movies'] != 5000 or manifest['qa_by_split'] != {'train': 41220, 'valid': 8857, 'test': 8917}:
        raise ValueError('Only the full frozen 5000-movie dataset is supported')
    if config['epochs'] != 5 or status['status'] != 'complete' or status['completed_epochs'] != 5:
        raise ValueError('The chosen run must have completed all five epochs')
    if args.split != 'valid':
        suite = read(work / 'suite_summary.json')
        selected = read(project / 'output/selected_model.json')['selected_run']
        if (suite['status'] != 'complete' or len(suite['runs']) != 5
                or {r['run_id'] for r in suite['runs']} != {'B0', 'B1', 'B2', 'B3', 'C0'}
                or suite['selected_run'] != selected or args.run_id != selected):
            raise ValueError('Test/robustness requires the complete suite and its validation-selected B run')
        for row in suite['runs']:
            state = read(project / 'runs' / row['run_id'] / 'status.json')
            if (state['status'] != 'complete' or state['completed_epochs'] != 5
                    or row['valid']['count'] != 8857 or row['test']['count'] != 8917):
                raise ValueError('Every suite run must complete full validation and test coverage')
    tokenizer_path = run / 'checkpoint'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    mapping_path = project / 'models/transe/mapping.json' if config['use_embeddings'] else None
    protocol = {
        'method': 'Existing unique unknown-token name recovery, transferred without retraining',
        'run_id': args.run_id, 'fact_limit': FACT_LIMIT, 'minimum_alphanumeric_characters': 3,
        'entity_decoder': 'Unchanged AnswerEntityIndex with actual linked retained-tail homonym context',
        'baseline': 'Saved raw answer_ids are preserved; re-extraction discrepancies are reported',
        'input_manifest_sha256': digest(work / 'input_manifest.json'),
        'run_config_sha256': digest(run / 'config.json'),
        'input_sha256': {n: digest(data / n) for n in ('entities.json', 'qa.jsonl', 'splits.json')},
        'mapping_sha256': digest(mapping_path) if mapping_path else None,
        'tokenizer': tokenizer_identity(tokenizer, tokenizer_path),
        'tokenizer_files_sha256': {n: digest(tokenizer_path / n) for n in TOKENIZER_FILES if (tokenizer_path / n).is_file()},
        'source_sha256': {n: digest(project / 'kgqa' / n) for n in
                          ('name_recovery.py', 'metrics.py', 'linker.py', 'neural_preflight.py', 'common.py')},
        'adapter_sha256': digest(Path(__file__)),
    }
    validation = read(args.validation_report) if args.validation_report else None
    if args.split != 'valid' and (validation is None or validation.get('split') != 'valid'
                                  or not validation.get('selected_on_validation') or validation['protocol'] != protocol):
        raise ValueError('A successful matching validation report must freeze the preset before test/robustness')
    predictions_path = run / f'predictions_{args.split}.jsonl'
    raw = read_jsonl(predictions_path)
    if any('answer_ids' not in row for row in raw):
        raise ValueError('Saved raw answer IDs are required; no baseline re-extraction is substituted')
    entities, splits = read(data / 'entities.json'), read(data / 'splits.json')
    mapping = read(mapping_path) if mapping_path else None
    linker, recovery, index = EntityLinker(sum(splits.values(), []), entities), NameRecovery(entities, tokenizer), AnswerEntityIndex(entities)
    recovered, discrepancies = [], []
    started = time.perf_counter()
    for pred in raw:
        tick = time.perf_counter()
        linked, _, link_status = linker.link(pred['question'])
        tails = [tail for _, tail in _facts(linked, entities, mapping)[:FACT_LIMIT]]
        extracted = index.match(pred['answer'], linked, preferred_ids=tails)
        if set(extracted) != set(pred['answer_ids']):
            discrepancies.append({'question_id': pred['question_id'], 'saved_ids': pred['answer_ids'], 'reextracted_ids': extracted})
        recovery_started = time.perf_counter()
        restored = recovery.recover(pred['answer'], tails)
        recovery_ms = 1000 * (time.perf_counter() - recovery_started)
        recovered.append({**pred, **restored, 'linked_movie_id': linked, 'link_status': link_status,
                          'answer_ids': index.match(restored['answer'], linked, preferred_ids=tails),
                          'recovery_ms': recovery_ms, 'processing_ms': 1000 * (time.perf_counter() - tick)})
    processing_seconds = time.perf_counter() - started
    # Gold enters only after all recovery/extraction is complete.
    gold_path = data / ('robustness.jsonl' if args.split == 'robustness' else 'qa.jsonl')
    gold = read_jsonl(gold_path)
    if args.split != 'robustness':
        gold = [row for row in gold if row['split'] == args.split]
        if len(gold) != manifest['qa_by_split'][args.split]:
            raise ValueError('Incomplete gold population')
    gold_index = {row['question_id']: row for row in gold}
    if any(row['question'] != gold_index[row['question_id']]['question'] for row in raw):
        raise ValueError('Saved prediction question text differs from gold question ID')
    baseline, improved = score_rows(gold, raw, entities), score_rows(gold, recovered, entities)
    fixed, regressed, changed = [], [], []
    for before, after in zip(raw, recovered):
        item = gold_index[before['question_id']]
        def correct(pred):
            return (set(pred['answer_ids']) == set(item['answer_ids']) if item['answer_ids']
                    else normalize(pred['answer']) == normalize(REFUSAL))
        a, b = correct(before), correct(after)
        if not a and b: fixed.append(before['question_id'])
        if a and not b: regressed.append(before['question_id'])
        if before['answer'] != after['answer']: changed.append(before['question_id'])
    report = {'scope': 'Additional deterministic postprocessing only; primary raw metrics and model selection are unchanged.',
              'split': args.split, 'protocol': protocol, 'predictions_sha256': digest(predictions_path),
              'gold_sha256': digest(gold_path), 'baseline': baseline, 'with_name_recovery': improved,
              'fixed_question_ids': fixed, 'regressed_question_ids': regressed, 'changed_question_ids': changed,
              'raw_reextraction_discrepancies': discrepancies, 'processing_seconds': processing_seconds,
              'timing_note': 'Frozen latency_ms is unchanged. processing_ms includes linking, recovery and both entity extractions; recovery_ms isolates replacement processing. No model generation was run.',
              'selected_on_validation': args.split == 'valid' and bool(fixed) and not regressed and not discrepancies,
              'validation_report_sha256': digest(args.validation_report) if args.validation_report else None}
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / 'predictions_recovered.jsonl', recovered)
    write_json(output / 'raw_metrics.json', baseline)
    write_json(output / 'recovered_metrics.json', improved)
    write_json(output / 'report.json', report)
    print(json.dumps({'split': args.split, 'count': len(raw), 'raw_f1': baseline['entity_macro_f1'],
                      'recovered_f1': improved['entity_macro_f1'], 'fixed': len(fixed), 'regressed': len(regressed),
                      'raw_reextraction_discrepancies': len(discrepancies), 'processing_seconds': processing_seconds}))


if __name__ == '__main__':
    main()
