"""Paired strict evaluation of name recovery over frozen neural predictions."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from .common import REFUSAL, read_json, read_jsonl, sha256, write_json, write_jsonl
from .linker import EntityLinker, normalize
from .metrics import EntityNameIndex, score_rows
from .name_recovery import NameRecovery
from .neural_preflight import FACT_LIMIT, _facts, tokenizer_identity


def recover_predictions(predictions, entities, movie_ids, tokenizer, mapping=None):
    """Accept questions and raw outputs only, never gold answers or relations."""
    linker = EntityLinker(movie_ids, entities)
    recovery = NameRecovery(entities, tokenizer)
    index = EntityNameIndex(entities)
    output = []
    for pred in predictions:
        started = time.perf_counter()
        movie_id, _, status = linker.link(pred['question'])
        candidates = [tail for _, tail in _facts(movie_id, entities, mapping)[:FACT_LIMIT]]
        result = recovery.recover(pred['answer'], candidates)
        output.append({**pred, **result, 'linked_movie_id': movie_id, 'link_status': status,
                       'answer_ids': index.match(result['answer'], movie_id),
                       'recovery_ms': (time.perf_counter() - started) * 1000})
    return output


def paired_scores(gold, raw, recovered, entities):
    """Gold data are used only for the unchanged strict evaluation."""
    gold_index = {row['question_id']: row for row in gold}
    index = EntityNameIndex(entities)

    def extract(rows):
        return [{**row, 'answer_ids': index.match(row['answer'], gold_index[row['question_id']]['movie_id'])}
                for row in rows]

    before, after = extract(raw), extract(recovered)
    baseline, improved = score_rows(gold, before, entities), score_rows(gold, after, entities)
    before_index = {row['question_id']: row for row in before}
    after_index = {row['question_id']: row for row in after}
    fixed, regressed, remaining, changed = [], [], [], []
    for row in gold:
        qid = row['question_id']; a, b = before_index[qid], after_index[qid]
        def correct(pred):
            return (set(row['answer_ids']) == set(pred['answer_ids']) if row['answer_ids']
                    else normalize(pred['answer']) == normalize(REFUSAL))
        a_ok, b_ok = correct(a), correct(b)
        if not a_ok and b_ok: fixed.append(qid)
        if a_ok and not b_ok: regressed.append(qid)
        if not b_ok:
            remaining.append({'question_id': qid, 'question': row['question'], 'answer': b['answer'],
                              'gold_answer_ids': row['answer_ids'], 'predicted_answer_ids': b['answer_ids']})
        if a['answer'] != b['answer']: changed.append(qid)
    return {'baseline': baseline, 'with_name_recovery': improved, 'fixed_question_ids': fixed,
            'regressed_question_ids': regressed, 'changed_question_ids': changed, 'remaining_errors': remaining}


def run(data_dir, prediction_path, tokenizer, output_dir, split, *, tokenizer_source=None,
        mapping_path=None, validation_report=None, robustness_path=None):
    data_dir, prediction_path, output_dir = map(Path, (data_dir, prediction_path, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Use a new output directory to preserve previous experiment evidence')
    entities = read_json(data_dir / 'entities.json'); splits = read_json(data_dir / 'splits.json')
    mapping = read_json(mapping_path) if mapping_path else None
    sources = ('name_recovery.py', 'recovery_experiment.py', 'neural_preflight.py', 'linker.py', 'metrics.py')
    protocol = {
        'method': 'Unique tokenizer-unknown name restoration from linked retained fact tails',
        'fact_limit': FACT_LIMIT, 'minimum_alphanumeric_characters': 3,
        'input_sha256': {name: sha256(data_dir / name) for name in ('entities.json', 'splits.json', 'qa.jsonl')},
        'mapping_sha256': sha256(mapping_path) if mapping_path else None,
        'tokenizer': tokenizer_identity(tokenizer, tokenizer_source),
        'source_sha256': {name: sha256(Path(__file__).with_name(name)) for name in sources},
    }
    if split != 'valid':
        if validation_report is None:
            raise ValueError('Test/robustness evaluation requires a successful validation report')
        selected = read_json(validation_report)
        if selected.get('split') != 'valid' or not selected.get('selected_on_validation'):
            raise ValueError('The supplied report did not select the method on validation')
        if selected['protocol'] != protocol:
            raise ValueError('Protocol changed after validation; rerun validation before testing')
    if split == 'robustness':
        if robustness_path is None: raise ValueError('Robustness evaluation requires --robustness-file')
        gold = read_jsonl(robustness_path)
    else:
        gold = [row for row in read_jsonl(data_dir / 'qa.jsonl') if row['split'] == split]
    raw = read_jsonl(prediction_path)
    score_rows(gold, raw, entities)
    predicted_questions = {row['question_id']: row['question'] for row in raw}
    if any(predicted_questions[row['question_id']] != row['question'] for row in gold):
        raise ValueError('Prediction question text does not match its dataset question ID')
    recovered = recover_predictions(raw, entities, sum(splits.values(), []), tokenizer, mapping)
    scores = paired_scores(gold, raw, recovered, entities)
    report = {'split': split, 'protocol': protocol,
              'scope': 'Postprocessing frozen predictions; no retraining or regeneration. Strict scores unchanged in definition.',
              'latency_note': 'latency_ms is frozen generation timing; recovery_ms includes linking, restoration and extraction.',
              'predictions_sha256': sha256(prediction_path), **scores,
              'selected_on_validation': split == 'valid' and bool(scores['fixed_question_ids']) and not scores['regressed_question_ids'],
              'recovery_mean_ms': sum(r['recovery_ms'] for r in recovered) / len(recovered)}
    if validation_report: report['validation_report_sha256'] = sha256(validation_report)
    if robustness_path:
        report['robustness_sha256'] = sha256(robustness_path)
        report['by_variant'] = {}
        for variant in sorted({row['variant'] for row in gold}):
            subset = [row for row in gold if row['variant'] == variant]; ids = {row['question_id'] for row in subset}
            report['by_variant'][variant] = paired_scores(subset, [r for r in raw if r['question_id'] in ids],
                                                        [r for r in recovered if r['question_id'] in ids], entities)
    write_jsonl(output_dir / 'predictions.jsonl', recovered); write_json(output_dir / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--tokenizer', required=True, help='Local directory or cached model ID; no download')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('valid', 'test', 'robustness'), required=True)
    parser.add_argument('--mapping', type=Path)
    parser.add_argument('--validation-report', type=Path)
    parser.add_argument('--robustness-file', type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    report = run(args.data_dir, args.predictions, tokenizer, args.output_dir, args.split,
                 tokenizer_source=args.tokenizer, mapping_path=args.mapping,
                 validation_report=args.validation_report, robustness_path=args.robustness_file)
    print(f"{args.split}: exact {report['baseline']['entity_exact']:.4f} -> {report['with_name_recovery']['entity_exact']:.4f}; "
          f"F1 {report['baseline']['entity_macro_f1']:.4f} -> {report['with_name_recovery']['entity_macro_f1']:.4f}; "
          f"fixed={len(report['fixed_question_ids'])}; regressed={len(report['regressed_question_ids'])}")


if __name__ == '__main__':
    main()
