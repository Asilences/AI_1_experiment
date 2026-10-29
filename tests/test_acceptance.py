from pathlib import Path
import shutil
import socket

import pytest

from kgqa import acceptance
from kgqa.common import REFUSAL, RELATIONS, read_json, read_jsonl, write_jsonl


@pytest.fixture(scope='module')
def accepted_output(tmp_path_factory):
    output = tmp_path_factory.mktemp('acceptance_parent') / 'offline'
    def no_network(*args, **kwargs):
        raise AssertionError('Offline acceptance must not open a network connection')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, 'connect', no_network)
        acceptance.run_acceptance(output)
    return output


def test_real_offline_acceptance_covers_the_required_behaviors(accepted_output):
    summary = read_json(accepted_output / 'summary.json')
    assert summary['status'] == 'PASS'
    assert summary['source'] == 'synthetic'
    assert summary['scientific_result'] is False
    assert 'not a measured Wikidata result' in summary['scope']
    assert all(summary['checks'].values())
    assert summary['fixture_movies'] == 10
    assert summary['movie_split_sizes'] == {'train': 6, 'valid': 2, 'test': 2}
    assert summary['training_runs'] == 1
    assert summary['test_forward_passes'] == 15
    assert (accepted_output / summary['model_path']).is_file()
    data = accepted_output / summary['data_dir']
    qa, splits = read_jsonl(data / 'qa.jsonl'), read_json(data / 'splits.json')
    assert len(qa) == 150
    assert {row['relation_id'] for row in qa} == {*RELATIONS, 'UNKNOWN'}
    assert not set(splits['train']) & (set(splits['valid']) | set(splits['test']))
    assert not set(splits['valid']) & set(splits['test'])
    examples = read_json(accepted_output / 'examples.json')['examples']
    assert len(examples['multi_answer']['answer_ids']) > 1
    assert examples['alias']['answer_ids']
    assert examples['unknown_movie_refusal']['answer'] == REFUSAL
    assert examples['unknown_movie_refusal']['movie_id'] is None


def test_generated_metadata_uses_portable_paths(accepted_output):
    summary = read_json(accepted_output / 'summary.json')
    for key in ('data_dir', 'model_path', 'run_dir', 'examples_path'):
        assert not Path(summary[key]).is_absolute()
    run = accepted_output / summary['run_dir']
    config, manifest = read_json(run / 'config.json'), read_json(run / 'data_manifest.json')
    assert config['data_scope'] == acceptance.SCOPE
    assert not Path(config['data_dir']).is_absolute()
    assert not Path(config['model_path']).is_absolute()
    assert all(not Path(item['path']).is_absolute() for item in manifest['input_files'].values())
    environment = read_json(run / 'environment.json')
    assert environment['executable'] == Path(environment['executable']).name
    assert 'acceptance.py' in read_json(run / 'source_hashes.json')


def test_repeated_explicit_destination_is_refused_without_changing_outputs(accepted_output):
    old = (accepted_output / 'summary.json').read_bytes()
    with pytest.raises(FileExistsError):
        acceptance.run_acceptance(accepted_output)
    assert (accepted_output / 'summary.json').read_bytes() == old


def test_default_destination_is_unique_and_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(acceptance, 'ROOT', tmp_path)
    first, second = acceptance._default_output_dir(), acceptance._default_output_dir()
    assert first != second
    assert first.parent == second.parent == tmp_path / 'output/acceptance'
    assert not first.exists() and not second.exists()


def test_missing_prediction_is_detected(accepted_output, tmp_path):
    copied = tmp_path / 'broken'
    shutil.copytree(accepted_output, copied)
    predictions_file = copied / acceptance.RUN_PATH / 'predictions_test.jsonl'
    write_jsonl(predictions_file, read_jsonl(predictions_file)[1:])
    checks, _, _ = acceptance.inspect_outputs(copied)
    assert checks['full_test_coverage'] is False
    assert checks['reloaded_predictions_stable'] is False


def test_wrong_answer_is_detected_even_with_complete_coverage(accepted_output, tmp_path):
    copied = tmp_path / 'wrong_answer'
    shutil.copytree(accepted_output, copied)
    predictions_file = copied / acceptance.RUN_PATH / 'predictions_test.jsonl'
    predictions = read_jsonl(predictions_file)
    predictions[0]['answer_ids'] = ['SYN_WRONG_ANSWER']
    write_jsonl(predictions_file, predictions)
    checks, _, _ = acceptance.inspect_outputs(copied)
    assert checks['full_test_coverage'] is True
    assert checks['four_relations_answered'] is True
    assert checks['all_test_answers_correct'] is False


def test_cli_reports_failure_and_nonzero_exit(tmp_path, capsys):
    assert acceptance.main(['--output-dir', str(tmp_path)]) == 1
    assert 'FAIL' in capsys.readouterr().err
    assert not (tmp_path / 'summary.json').exists()


def test_failed_check_is_saved_and_returns_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(acceptance.personal_experiment, 'run', lambda *args: None)
    monkeypatch.setattr(acceptance, '_portable_metadata', lambda output: None)
    monkeypatch.setattr(acceptance, 'inspect_outputs', lambda output: ({'run_complete': False}, {}, {}))
    output = tmp_path / 'failed'
    assert acceptance.main(['--output-dir', str(output)]) == 1
    assert 'FAIL' in capsys.readouterr().err
    summary = read_json(output / 'summary.json')
    assert summary['status'] == 'FAIL'
    assert summary['checks'] == {'run_complete': False}


def test_fixture_creation_is_self_contained_and_needs_no_network(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError('Offline acceptance must not open a network connection')
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    data = tmp_path / 'data'
    acceptance.build_fixture(data)
    assert read_json(data / 'manifest.json')['source'] == 'synthetic'
    qa, entities, splits = read_jsonl(data / 'qa.jsonl'), read_json(data / 'entities.json'), read_json(data / 'splits.json')
    grouped = acceptance.personal_experiment.validate_inputs(qa, entities, splits, read_jsonl(data / 'robustness.jsonl'))
    assert {key: len(rows) for key, rows in grouped.items()} == {'train': 90, 'valid': 30, 'test': 30}
