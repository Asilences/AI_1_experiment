"""Offline, isolated 5000-film suite; checkpoints remain under WORK/project.

Resume only reuses hash-verified complete stages. A failed/interrupted stage
requires a new work directory; this runner never overwrites its partial output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUNS = (('B0', 1e-4, True), ('B1', 3e-5, True), ('B2', 3e-4, True),
        ('B3', 1e-3, True), ('C0', 1e-4, False))
DATA_FILES = ('entities.json', 'manifest.json', 'qa.jsonl', 'splits.json', 'triples.tsv')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def hashes(root, paths):
    return {Path(p).relative_to(root).as_posix(): digest(p) for p in sorted(paths)}


def frozen_inputs(project):
    paths = [project / 'data/processed' / n for n in DATA_FILES]
    paths += [project / 'data/raw/snapshot_5000.json']
    paths += [p for p in (project / 'models/pipeline').rglob('*') if p.is_file()]
    return hashes(project, paths)


def source_hashes(project):
    return hashes(project, (project / 'kgqa').rglob('*.py'))


def artifact_hashes(project, paths):
    files = []
    for path in paths:
        if path.is_dir():
            children = [p for p in path.rglob('*') if p.is_file()]
            if not children:
                raise ValueError('Empty artifact directory')
            files.extend(children)
        else:
            files.append(path)
    return hashes(project, files)


def prepare(work, source=ROOT):
    """Copy published entrypoints; preserve the paired A0 without retraining."""
    work.mkdir(parents=True, exist_ok=False)
    project = work / 'project'
    manifest = read(source / 'datasets/movies-5000-20261029/manifest.json')
    for entry in manifest['files']:
        path = source / entry['path']
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Published dataset hash mismatch: ' + entry['path'])
    shutil.copytree(source / 'kgqa', project / 'kgqa', ignore=shutil.ignore_patterns('__pycache__'))
    points = manifest['entrypoints']
    shutil.copytree(source / points['data_dir'], project / 'data/processed')
    shutil.copytree((source / points['model_path']).parent, project / 'models/pipeline')
    snapshot = project / 'data/raw/snapshot_5000.json'
    snapshot.parent.mkdir(parents=True)
    shutil.copy2(source / points['snapshot_path'], snapshot)
    write(work / 'input_manifest.json', {
        'schema_version': 1, 'dataset': manifest['dataset_id'],
        'input_sha256': frozen_inputs(project), 'source_sha256': source_hashes(project)})
    return project


def verify_frozen(work):
    project = work / 'project'
    manifest = read(work / 'input_manifest.json')
    if 'source_sha256' not in manifest or 'input_sha256' not in manifest:
        raise ValueError('Legacy manifest requires explicit owner-confirmed migration before resume')
    if manifest['source_sha256'] != source_hashes(project):
        raise ValueError('Frozen project source hash mismatch')
    if manifest['input_sha256'] != frozen_inputs(project):
        raise ValueError('Frozen dataset/A0 hash mismatch')
    return manifest


def transe_complete(project):
    folder = project / 'models/transe'
    required = ('config.json', 'history.json', 'embeddings.npz', 'mapping.json')
    if not all((folder / n).is_file() for n in required):
        return False
    config, history = read(folder / 'config.json'), read(folder / 'history.json')
    return (config.get('epochs') == history.get('epochs') == 100
            and len(history.get('loss', [])) == 100
            and config.get('triples_sha256') == digest(project / 'data/processed/triples.tsv'))


class Runner:
    def __init__(self, work, config, resume=False, execute=None):
        self.work, self.project = work, work / 'project'
        self.resume, self.execute = resume, execute or subprocess.run
        self.identity = {'inputs': verify_frozen(work), 'config': config}

    def stage(self, name, args, outputs, *, occupied=(), adopt=False):
        state_path = self.work / 'stages' / (name + '.json')
        paths = [self.project / p for p in outputs]
        if state_path.exists():
            state = read(state_path)
            if not self.resume or state.get('status') != 'complete':
                raise FileExistsError('Stage unfinished or resume not requested: ' + name)
            if state.get('identity') != self.identity or state.get('outputs') != artifact_hashes(self.project, paths):
                raise ValueError('Completed stage hash/config mismatch: ' + name)
            return
        if any((self.project / p).exists() for p in occupied) and not adopt:
            raise FileExistsError('Refusing to overwrite untracked/partial stage: ' + name)
        state = {'schema_version': 1, 'stage': name, 'identity': self.identity,
                 'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat()}
        write(state_path, state)
        try:
            if not adopt:
                env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                           PYTHONHASHSEED='42', PYTHONPATH=str(self.project),
                           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                log = self.work / 'logs' / (name + '.log')
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open('x', encoding='utf-8') as stream:
                    self.execute([sys.executable, *args], cwd=self.project, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT, check=True)
            if any(not p.exists() for p in paths):
                raise ValueError('Stage missing required artifacts: ' + name)
            state.update(status='complete', outputs=artifact_hashes(self.project, paths),
                         completed_at=datetime.now(timezone.utc).isoformat(), adopted=adopt)
            write(state_path, state)
        except BaseException as error:
            state.update(status='failed', error_type=type(error).__name__)
            write(state_path, state)
            raise


def select_best(rows):
    candidates = [r for r in rows if r['run_id'] in {'B0', 'B1', 'B2', 'B3'}]
    return min(candidates, key=lambda r: (-r['valid']['entity_macro_f1'], r['run_id']))['run_id']


def run_suite(args):
    work = args.work_dir.resolve()
    project = work / 'project' if args.resume else prepare(work)
    model = str(Path(args.model_name).resolve()) if Path(args.model_name).is_dir() else args.model_name
    config = {'seed': 42, 'epochs': 5, 'transe_epochs': 100, 'batch_size': args.batch_size,
              'eval_batch_size': args.eval_batch_size, 'gradient_accumulation': 2,
              'input_max_length': args.input_max_length, 'target_max_length': 96,
              'torch_threads': 4, 'generation_precision': 'float32',
              'runner_source_sha256': digest(Path(__file__)),
              'model': Path(model).name if Path(model).is_dir() else model,
              'pretrained_sha256': hashes(Path(model), [p for p in Path(model).rglob('*') if p.is_file()])
              if Path(model).is_dir() else {}}
    runner = Runner(work, config, args.resume)
    script = ('from transformers import AutoTokenizer,AutoModelForSeq2SeqLM;import sys;'
              'AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True);'
              'AutoModelForSeq2SeqLM.from_pretrained(sys.argv[1],local_files_only=True);'
              'from pathlib import Path;Path("offline_assets.ok").write_text("verified")')
    runner.stage('offline_assets', ['-c', script, model], ['offline_assets.ok'], occupied=['offline_assets.ok'])
    transe = ['models/transe/' + n for n in ('config.json', 'history.json', 'embeddings.npz', 'mapping.json')]
    runner.stage('transe', ['-m', 'kgqa.transe', '--epochs', '100'], transe,
                 occupied=['models/transe'], adopt=args.resume and transe_complete(project))
    results = []
    for run_id, rate, embeddings in RUNS:
        prefix = 'runs/' + run_id + '/'
        command = ['-m', 'kgqa.neural', '--run-id', run_id, '--learning-rate', str(rate),
                   '--epochs', '5', '--batch-size', str(args.batch_size), '--gradient-accumulation', '2',
                   '--eval-batch-size', str(args.eval_batch_size), '--input-max-length', str(args.input_max_length),
                   '--model-name', model]
        if not embeddings:
            command.append('--no-embeddings')
        outputs = [prefix + n for n in ('config.json', 'status.json', 'history.json', 'selection.json',
                                        'predictions_valid.jsonl', 'predictions_test.jsonl', 'checkpoint')]
        runner.stage(run_id + '_train', command, outputs, occupied=['runs/' + run_id])
        status = read(project / prefix / 'status.json')
        if status.get('status') != 'complete' or status.get('completed_epochs') != 5:
            raise ValueError('Training did not complete five epochs: ' + run_id)
        for split in ('valid', 'test'):
            runner.stage(run_id + '_' + split, ['-m', 'kgqa.metrics', run_id, '--split', split],
                         [prefix + 'metrics_' + split + '.json'])
        results.append({'run_id': run_id, 'learning_rate': rate, 'use_embeddings': embeddings,
                        'valid': read(project / prefix / 'metrics_valid.json'),
                        'test': read(project / prefix / 'metrics_test.json'),
                        'selection': read(project / prefix / 'selection.json')})
    selected = select_best(results)
    write(project / 'output/selected_model.json', {'selected_run': selected,
          'validation_f1': next(row['valid']['entity_macro_f1'] for row in results if row['run_id'] == selected),
          'candidates': {row['run_id']: row['valid']['entity_macro_f1'] for row in results if row['run_id'] != 'C0'}})
    write(work / 'suite_summary.json', {'schema_version': 1, 'status': 'complete', 'config': config, 'runs': results,
          'selected_run': selected, 'selection_split': 'valid',
          'selection_metric': 'entity_macro_f1', 'tie_break': 'smallest B run ID',
          'excluded_from_selection': ['C0'], 'input_manifest': 'input_manifest.json'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--model-name', default='google/flan-t5-small')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--input-max-length', type=int, default=512)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.eval_batch_size, args.input_max_length) < 1:
        parser.error('Batch sizes and input length must be positive')
    run_suite(args)


if __name__ == '__main__':
    main()
