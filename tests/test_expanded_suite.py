"""Orchestration checks deliberately require neither torch nor transformers."""
from argparse import Namespace
import subprocess

import pytest

from kgqa import expanded_suite as suite


@pytest.fixture
def work(tmp_path):
    project = tmp_path / 'project'
    for name in suite.DATA_FILES:
        suite.write(project / 'data/processed' / name, {})
    suite.write(project / 'data/raw/snapshot_5000.json', {})
    suite.write(project / 'models/pipeline/config.json', {})
    (project / 'kgqa').mkdir()
    (project / 'kgqa/__init__.py').write_text('')
    suite.write(tmp_path / 'input_manifest.json', {
        'source_sha256': suite.source_hashes(project),
        'input_sha256': suite.frozen_inputs(project)})
    return tmp_path


def test_resume_reuses_complete_and_rejects_mutation(work):
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
        assert kwargs['env']['OMP_NUM_THREADS'] == '4'
        (kwargs['cwd'] / 'artifact').write_text('done')

    suite.Runner(work, {'seed': 42}, execute=execute).stage('sample', ['-m', 'fake'], ['artifact'])
    suite.Runner(work, {'seed': 42}, True, execute).stage('sample', [], ['artifact'])
    assert len(calls) == 1
    (work / 'project/artifact').write_text('changed')
    with pytest.raises(ValueError, match='mismatch'):
        suite.Runner(work, {'seed': 42}, True, execute).stage('sample', [], ['artifact'])


def test_failure_is_persisted_and_never_restarted(work):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    with pytest.raises(subprocess.CalledProcessError):
        suite.Runner(work, {}, execute=fail).stage('bad', [], ['missing'])
    assert suite.read(work / 'stages/bad.json')['status'] == 'failed'
    with pytest.raises(FileExistsError, match='unfinished'):
        suite.Runner(work, {}, True, fail).stage('bad', [], ['missing'])


def test_partial_output_and_config_change_rejected(work):
    (work / 'project/partial').mkdir()
    with pytest.raises(FileExistsError, match='partial'):
        suite.Runner(work, {}).stage('partial', [], [], occupied=['partial'])
    suite.Runner(work, {}).stage('adopted', [], [], adopt=True)
    with pytest.raises(ValueError, match='mismatch'):
        suite.Runner(work, {'changed': True}, True).stage('adopted', [], [])


def test_frozen_source_and_data_rejected(work):
    (work / 'project/kgqa/__init__.py').write_text('changed')
    with pytest.raises(ValueError, match='source'):
        suite.verify_frozen(work)


def test_legacy_manifest_requires_owner_migration(work):
    suite.write(work / 'input_manifest.json', {'inputs': {}})
    with pytest.raises(ValueError, match='owner-confirmed'):
        suite.verify_frozen(work)


def test_transe_adoption_requires_100_epochs_and_paired_hash(work):
    project = work / 'project'
    folder = project / 'models/transe'
    suite.write(folder / 'config.json', {'epochs': 100, 'triples_sha256': suite.digest(project / 'data/processed/triples.tsv')})
    suite.write(folder / 'history.json', {'epochs': 100, 'loss': [1.] * 100})
    suite.write(folder / 'mapping.json', {})
    (folder / 'embeddings.npz').write_bytes(b'embeddings')
    assert suite.transe_complete(project)
    suite.write(folder / 'history.json', {'epochs': 100, 'loss': [1.] * 99})
    assert not suite.transe_complete(project)


def test_selection_uses_validation_and_excludes_ablation():
    rows = [{'run_id': name, 'valid': {'entity_macro_f1': score},
             'test': {'entity_macro_f1': 100 - score}}
            for name, score in [('B2', 9), ('B1', 9), ('B0', 8), ('B3', 7), ('C0', 99)]]
    assert suite.select_best(rows) == 'B1'


def test_checkpoint_contents_are_hash_checked(work):
    project = work / 'project'
    suite.write(project / 'checkpoint/config.json', {})
    (project / 'checkpoint/model.safetensors').write_bytes(b'weights')
    suite.Runner(work, {}).stage('checkpoint', [], ['checkpoint'], adopt=True)
    (project / 'checkpoint/model.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='mismatch'):
        suite.Runner(work, {}, True).stage('checkpoint', [], ['checkpoint'])


def test_complete_suite_commands_and_collection(work, monkeypatch):
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        project = kwargs['cwd']
        if command[1] == '-c':
            (project / 'offline_assets.ok').write_text('verified')
        elif command[2] == 'kgqa.transe':
            for name in ('config.json', 'history.json', 'mapping.json', 'embeddings.npz'):
                suite.write(project / 'models/transe' / name, {})
        elif command[2] == 'kgqa.neural':
            name = command[command.index('--run-id') + 1]
            assert command[command.index('--epochs') + 1] == '5'
            assert command[command.index('--gradient-accumulation') + 1] == '2'
            assert ('--no-embeddings' in command) == (name == 'C0')
            folder = project / 'runs' / name
            for file in ('config.json', 'history.json', 'selection.json', 'predictions_valid.jsonl', 'predictions_test.jsonl', 'checkpoint/config.json'):
                suite.write(folder / file, {})
            suite.write(folder / 'status.json', {'status': 'complete', 'completed_epochs': 5})
        else:
            name, split = command[3], command[-1]
            suite.write(project / 'runs' / name / ('metrics_' + split + '.json'),
                        {'entity_macro_f1': 80 if name == 'C0' else 10})

    monkeypatch.setattr(suite.subprocess, 'run', execute)
    args = Namespace(work_dir=work, resume=True, model_name='cached/model',
                     batch_size=8, eval_batch_size=128, input_max_length=512)
    suite.run_suite(args)
    assert len(calls) == 17
    summary = suite.read(work / 'suite_summary.json')
    assert summary['selected_run'] == 'B0'
    assert len(summary['runs']) == 5
    assert str(work) not in (work / 'suite_summary.json').read_text()
    suite.run_suite(args)
    assert len(calls) == 17
