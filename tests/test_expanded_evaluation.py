import pytest

from kgqa import expanded_evaluation, robustness
from kgqa.acceptance import build_fixture
from kgqa.common import read_json, read_jsonl, sha256, write_json, write_jsonl
from kgqa.pipeline import train


def test_validation_perturbations_never_sample_test_movies(tmp_path):
    data = tmp_path / 'data'
    build_fixture(data)
    rows = robustness.build(5, 42, data, tmp_path / 'valid-questions', split='valid')
    splits = read_json(data / 'splits.json')
    assert rows and all(row['split'] == 'valid' for row in rows)
    assert {row['movie_id'] for row in rows} <= set(splits['valid'])
    assert not ({row['movie_id'] for row in rows} & set(splits['test']))
    assert read_json(tmp_path / 'valid-questions/robustness_manifest.json')['source_split'] == 'valid'
    model_path = train(data, tmp_path / 'model')
    result = expanded_evaluation.run(data, model_path, tmp_path / 'valid-result', limit=5, split='valid')
    assert result['evaluation_split'] == 'valid' and result['valid']['count'] == 30
    assert result['archived_test_regression'] is None and 'test' not in result


def test_versioned_run_keeps_frozen_inputs_and_uses_matched_denominators(tmp_path):
    data, models, output = tmp_path / 'data', tmp_path / 'model', tmp_path / 'evaluation'
    build_fixture(data)
    entities = read_json(data / 'entities.json')
    entities['SYN_FILM_09']['aliases'] = []
    write_json(data / 'entities.json', entities)
    model_path = train(data, models)
    before = {str(p): sha256(p) for p in [*data.iterdir(), model_path] if p.is_file()}
    result = expanded_evaluation.run(data, model_path, output, limit=5)
    assert result['movie_count'] == 10 and result['test']['count'] == 30
    assert result['test']['entity_exact'] == 100
    assert result['robustness']['clean']['count'] == 5
    assert result['robustness']['alias']['count'] < result['robustness']['clean']['count']
    for kind in ('alias', 'paraphrase', 'typo'):
        paired = result['paired_comparisons'][kind]
        assert paired['matched_count'] == result['robustness'][kind]['count']
        assert paired['clean']['count'] == paired['variant']['count'] == paired['matched_count']
    for kind, metrics in result['robustness'].items():
        assert result['robustness_errors'][kind]['incorrect_rows'] == round((100 - metrics['entity_exact']) * metrics['count'] / 100)
    assert before == {str(p): sha256(p) for p in [*data.iterdir(), model_path] if p.is_file()}
    assert (output / 'questions' / 'robustness.jsonl').is_file()
    assert read_json(output / 'summary.json')['frozen_inputs_unchanged']
    assert read_jsonl(output / 'predictions_robustness.jsonl')
    with pytest.raises(FileExistsError):
        expanded_evaluation.run(data, model_path, output, limit=5)


def test_mismatched_model_is_rejected_before_loading(tmp_path, monkeypatch):
    data, models = tmp_path / 'data', tmp_path / 'model'
    build_fixture(data)
    models.mkdir()
    model = models / 'model.pkl'
    model.write_bytes(b'not executable')
    write_json(models / 'config.json', {'input_hashes': {}, 'model_sha256': sha256(model)})
    monkeypatch.setattr(expanded_evaluation, 'load_model', lambda *args: pytest.fail('Must not load mismatched model'))
    with pytest.raises(ValueError, match='hashes do not match'):
        expanded_evaluation.run(data, model, tmp_path / 'evaluation')


def test_robustness_preserves_disambiguating_title_and_custom_data(tmp_path):
    data, output = tmp_path / 'data', tmp_path / 'output'
    write_json(data / 'entities.json', {
        'M1': {'label': 'Shared Movie', 'question_title': 'Shared Movie (2000)', 'aliases': ['First Movie']},
    })
    write_jsonl(data / 'qa.jsonl', [{
        'question_id': 'q1', 'movie_id': 'M1', 'relation_id': 'P57', 'split': 'test',
        'question': 'Who directed Shared Movie (2000)?', 'is_answerable': True,
        'answer_ids': ['D1'], 'reference_answers': ['The answer.'],
    }])
    before = (data / 'qa.jsonl').read_bytes()
    rows = robustness.build(data_dir=data, output_dir=output)
    assert next(row for row in rows if row['variant'] == 'paraphrase')['question'] == 'Can you tell me who helmed Shared Movie (2000)?'
    assert next(row for row in rows if row['variant'] == 'alias')['question'] == 'Who directed First Movie?'
    assert (data / 'qa.jsonl').read_bytes() == before
    with pytest.raises(FileExistsError):
        robustness.build(data_dir=data, output_dir=output)
    with pytest.raises(ValueError, match='isolated'):
        robustness.build(data_dir=data)
