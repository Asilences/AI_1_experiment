"""CPU evaluation of four preset inference thresholds on one frozen A0 model.

No training, model download, GPU use, or test-based threshold selection occurs.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kgqa.common import read_json, read_jsonl, sha256, utc_now, write_json, write_jsonl
from kgqa.metrics import score_rows
from kgqa.personal_experiment import validate_inputs
from kgqa.pipeline import PipelineQA, load_model
from threadpoolctl import threadpool_limits

THRESHOLDS = (0.30, 0.70, 0.90, 0.99)


def run(data_dir, model_path, output_dir):
    data, model_path, output = map(Path, (data_dir, model_path, output_dir))
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    inputs = {name: sha256(data / name) for name in ('qa.jsonl', 'entities.json', 'splits.json')}
    model_hash = sha256(model_path)
    config_path = model_path.parent / 'config.json'
    config_hash = sha256(config_path)
    config = read_json(config_path)
    if config.get('input_hashes') != inputs or config.get('model_sha256') != model_hash:
        raise ValueError('Frozen configuration input/model SHA256 mismatch')
    entities, splits = read_json(data / 'entities.json'), read_json(data / 'splits.json')
    rows = validate_inputs(read_jsonl(data / 'qa.jsonl'), entities, splits)['test']
    if len(rows) != 8917 or len(sum(splits.values(), [])) != 5000:
        raise ValueError('Expected published 5000-movie snapshot and all 8917 test questions')
    classifier, baseline = load_model(model_path)
    if baseline != config['selected_threshold'] or baseline != THRESHOLDS[0]:
        raise ValueError('Published validation-selected baseline must remain 0.30')
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), *(root / 'kgqa' / name for name in
               ('pipeline.py', 'linker.py', 'metrics.py', 'common.py', 'personal_experiment.py'))]
    source_hashes = {str(path.relative_to(root)): sha256(path) for path in sources}
    output.mkdir(parents=True, exist_ok=False)
    results = []
    with threadpool_limits(limits=2):
        for threshold in THRESHOLDS:
            name = f'threshold_{threshold:.2f}'
            target = output / name
            target.mkdir()
            system = PipelineQA(entities, sum(splits.values(), []), classifier, threshold)
            started = time.perf_counter()
            predictions = []
            for index, row in enumerate(rows, 1):
                predictions.append({**system.predict(row['question']), 'question_id': row['question_id']})
                if index % 1000 == 0:
                    print(f'{name}: {index}/{len(rows)}', flush=True)
            seconds = time.perf_counter() - started
            metrics = score_rows(rows, predictions, entities)
            metrics.update(gold_total=len(rows), prediction_coverage=100.0)
            write_jsonl(target / 'predictions_test.jsonl', predictions)
            write_json(target / 'metrics_test.json', metrics)
            results.append({'setting': name, 'threshold': threshold, 'is_baseline': threshold == baseline,
                            'inference_wall_seconds': seconds, 'metrics': metrics,
                            'predictions_sha256': sha256(target / 'predictions_test.jsonl')})
            print(json.dumps(results[-1]), flush=True)
    if inputs != {name: sha256(data / name) for name in inputs} or sha256(model_path) != model_hash or sha256(config_path) != config_hash:
        raise RuntimeError('Frozen input/model/config changed during evaluation')
    if source_hashes != {str(path.relative_to(root)): sha256(path) for path in sources}:
        raise RuntimeError('Evaluation sources changed during evaluation')
    result = {'created_at': utc_now(), 'scope': 'Four preset inference configurations of one frozen classifier; zero retraining runs.',
              'threshold_policy': 'Presets 0.30/0.70/0.90/0.99 retained from the original 200-movie experiment. No threshold is selected using test results.',
              'selected_baseline_threshold': baseline, 'test_count': len(rows), 'movie_count': 5000,
              'data_dir': str(data.resolve()), 'model_path': str(model_path.resolve()),
              'input_hashes': inputs, 'model_sha256': model_hash, 'config_sha256': config_hash,
              'source_sha256': source_hashes, 'frozen_inputs_unchanged': True, 'results': results,
              'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                              'OMP_NUM_THREADS': os.environ['OMP_NUM_THREADS'], 'MKL_NUM_THREADS': os.environ['MKL_NUM_THREADS'],
                              'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'scikit-learn', 'sacrebleu', 'rouge-score', 'threadpoolctl')}}}
    write_json(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--model-path', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    run(args.data_dir, args.model_path, args.output_dir)


if __name__ == '__main__':
    main()
