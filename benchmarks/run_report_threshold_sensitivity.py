"""Post-diagnostic A0 threshold sensitivity; no model fitting or selection."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kgqa.common import read_json, read_jsonl, sha256, utc_now, write_json, write_jsonl
from kgqa.personal_experiment import validate_inputs
from kgqa.pipeline import PipelineQA, load_model
from kgqa.metrics import score_rows
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed/movies_5000_20261029')
    parser.add_argument('--model-path', type=Path, default=ROOT / 'models/pipeline_5000_20261029/model.pkl')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing directories are refused.')
    args = parser.parse_args()
    data = args.data_dir.resolve()
    model = args.model_path.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = [data / name for name in ('entities.json', 'splits.json', 'qa.jsonl')]
    files.extend((model, model.with_name('config.json')))
    def asset_key(path):
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return ('model/' if path in (model, model.with_name('config.json')) else 'data/') + path.name

    before = {asset_key(p): sha256(p) for p in files}
    sources = {p.relative_to(ROOT).as_posix(): sha256(p) for p in (ROOT / 'kgqa').rglob('*.py')}
    entities, splits = read_json(data / 'entities.json'), read_json(data / 'splits.json')
    rows = validate_inputs(read_jsonl(data / 'qa.jsonl'), entities, splits)['test']
    assert len(rows) == 8917
    classifier, default = load_model(model)
    assert default == .30
    records = []
    with threadpool_limits(limits=2):
        for threshold in (.30, .995, .998, .999):
            system = PipelineQA(entities, sum(splits.values(), []), classifier, threshold)
            predictions = []
            started = time.perf_counter()
            for row in rows:
                predictions.append({**system.predict(row['question']), 'question_id': row['question_id']})
            elapsed = time.perf_counter() - started
            name = f'threshold_{threshold:.3f}'
            write_jsonl(output / (name + '.jsonl'), predictions)
            metrics = score_rows(rows, predictions, entities)
            confidences = [p['relation_confidence'] for row, p in zip(rows, predictions) if row['answer_ids']]
            record = {'threshold': threshold, 'test_count': len(rows), 'inference_wall_seconds': elapsed,
                      'metrics': metrics,
                      'false_refusals': sum(bool(row['answer_ids']) and not p['answer_ids'] for row, p in zip(rows, predictions)),
                      'answerable_confidence_min': min(confidences), 'answerable_confidence_max': max(confidences),
                      'predictions_sha256': sha256(output / (name + '.jsonl'))}
            records.append(record)
            write_json(output / (name + '_metrics.json'), record)
            print(json.dumps({'threshold': threshold, 'f1': metrics['entity_macro_f1'], 'false_refusals': record['false_refusals'], 'seconds': elapsed}), flush=True)
    after = {asset_key(p): sha256(p) for p in files}
    assert before == after
    assert sources == {p.relative_to(ROOT).as_posix(): sha256(p) for p in (ROOT / 'kgqa').rglob('*.py')}
    result = {'created_at': utc_now(), 'scope': 'Post-diagnostic inference sensitivity after the four original presets showed no score change; no retraining or default-threshold selection.',
              'threshold_policy': 'Fixed additional settings 0.995/0.998/0.999 and repeated 0.30 control in the same process; motivated by saved confidence audit, not a prespecified generalization experiment.',
              'default_threshold_unchanged': default, 'test_count': len(rows), 'seed': 42,
              'input_sha256': before, 'inputs_unchanged': before == after,
              'source_sha256': sources, 'runner_sha256': sha256(Path(__file__)),
              'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                              'OMP_NUM_THREADS': 2, 'MKL_NUM_THREADS': 2,
                              'versions': {n: importlib.metadata.version(n) for n in ('numpy', 'scikit-learn', 'sacrebleu', 'rouge-score', 'threadpoolctl')}},
              'results': records}
    write_json(output / 'summary.json', result)
    print('Complete; frozen inputs, model and source unchanged', flush=True)


if __name__ == '__main__':
    main()
