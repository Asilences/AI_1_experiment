from copy import deepcopy
import csv
import pickle

import pytest

from kgqa import personal_experiment as experiment
from kgqa.common import REFUSAL, answer_text, read_json, read_jsonl, write_json, write_jsonl


@pytest.fixture
def tiny_data(tmp_path):
    entities = {'D': {'label': 'Jane Director'}, 'G': {'label': 'Adventure'}}
    splits = {'train': ['F1', 'F2'], 'valid': ['F3'], 'test': ['F4']}
    qa = []
    for split, movies in splits.items():
        for movie in movies:
            title = f'Sample Movie {movie}'
            entities[movie] = {'label': title, 'aliases': [f'Alternate {movie}'],
                               'facts': {'P57': ['D'], 'P136': ['G']}}
            for relation, template in [('P57', 'Who directed {title}?'), ('P136', 'What genre is {title}?')]:
                answer_ids = entities[movie]['facts'][relation]
                qa.append({'question_id': f'{movie}_{relation}', 'movie_id': movie, 'split': split,
                           'relation_id': relation, 'question': template.format(title=title),
                           'answer_ids': answer_ids, 'is_answerable': True,
                           'reference_answers': [answer_text(title, relation, [entities[answer_ids[0]]['label']])]})
            qa.append({'question_id': f'{movie}_UNKNOWN', 'movie_id': movie, 'split': split,
                       'relation_id': 'UNKNOWN', 'question': f'What was the budget of {title}?',
                       'answer_ids': [], 'is_answerable': False, 'reference_answers': [REFUSAL]})
    path = tmp_path / 'processed'
    write_json(path / 'entities.json', entities)
    write_json(path / 'splits.json', splits)
    write_jsonl(path / 'qa.jsonl', qa)
    return path, entities, splits, qa


def test_runner_saves_reproducible_records_and_preserves_existing_runs(tiny_data, tmp_path):
    data, entities, splits, qa = tiny_data
    # Existing collaborative models and runs must remain byte-for-byte untouched.
    old_model = tmp_path / 'models/pipeline/model.pkl'
    old_model.parent.mkdir(parents=True)
    old_model.write_bytes(b'old shared model')
    old_run = tmp_path / 'runs/A0/metrics_test.json'
    write_json(old_run, {'historical': True})
    robust = [{**row, 'question_id': f"robust_clean_{row['question_id']}", 'variant': 'clean',
               'source_question_id': row['question_id']} for row in qa if row['split'] == 'test']
    robust.append({**robust[0], 'question_id': 'robust_alias_one', 'variant': 'alias'})
    write_jsonl(data / 'robustness.jsonl', robust)
    write_json(data / 'robustness_manifest.json', {'comparisons': {'alias': {
        'matched_count': 1, 'pairs': [{'clean_question_id': robust[0]['question_id'],
                                     'variant_question_id': robust[-1]['question_id']}]}}})
    before = {path: path.read_bytes() for path in (old_model, old_run, data / 'qa.jsonl')}
    directory = experiment.run(data, tmp_path / 'runs', 'new_personal')
    assert all(path.read_bytes() == content for path, content in before.items())
    config = read_json(directory / 'config.json')
    assert config['selection_split'] == 'valid'
    assert config['training_runs'] == 1
    assert config['test_selection'] is False
    assert config['fixed_test_thresholds'] == [.30, .70, .90, .99]
    assert 'false-refusal' in config['fixed_threshold_rationale']
    assert config['feature_fit_split'] == 'train'
    assert read_json(directory / 'status.json')['test_forward_passes'] == 15
    assert read_json(directory / 'metrics_test.json')['prediction_coverage'] == 100
    assert len(read_jsonl(directory / 'predictions_test.jsonl')) == 3
    with (directory / 'model.pkl').open('rb') as stream:
        saved = pickle.load(stream)
    assert saved['threshold'] == config['threshold']
    # Movie title tokens from validation/test must never enter the vocabulary.
    assert not any('f3' in term or 'f4' in term for term in saved['features'].transformer_list[0][1].vocabulary_)
    selection = read_json(directory / 'validation_thresholds.json')
    assert len(selection['records']) == 12
    assert selection['tie_break'] == 'lowest threshold'
    assert selection['count'] == 3
    sensitivity = read_json(directory / 'sensitivity_summary.json')
    assert len(sensitivity['settings']) == 5
    for setting in sensitivity['settings']:
        assert len(setting['repeats']) == 3
        assert setting['deterministic_predictions'] is True
        assert setting['metric_population_std']['entity_exact'] == 0
        for repeat in setting['repeats']:
            assert (directory / repeat['prediction_path']).is_file()
            assert (directory / repeat['metrics_path']).is_file()
    assert read_json(directory / 'robustness_summary.json')['variant_counts'] == {'alias': 1, 'clean': 3}
    assert read_json(directory / 'metrics_robustness.json')['alias']['count'] == 1
    matched = read_json(directory / 'metrics_robustness_matched.json')['alias']
    assert matched['matched_count'] == 1
    assert matched['all_clean_count'] == 3
    assert matched['matched_clean_metrics']['count'] == matched['matched_variant_metrics']['count'] == 1
    with (directory / 'sensitivity_comparison.csv').open(encoding='utf-8-sig', newline='') as stream:
        comparison_rows = list(csv.DictReader(stream))
    assert len(comparison_rows) == 5
    assert {row['total_test_forward_passes'] for row in comparison_rows} == {'15'}
    assert {row['threshold_selection_split'] for row in comparison_rows} == {'valid'}
    assert all(float(row['entity_exact_population_std']) == 0 for row in comparison_rows)
    assert set(read_json(directory / 'source_hashes.json')) >= {'pipeline.py', 'personal_experiment.py'}
    assert 'scikit-learn' in read_json(directory / 'environment.json')['packages']
    with pytest.raises(FileExistsError, match='overwrite'):
        experiment.run(data, tmp_path / 'runs', 'new_personal')


def test_threshold_selection_is_valid_only_and_ties_choose_lowest():
    gold = [{'question_id': 'v1', 'split': 'valid', 'answer_ids': ['D']},
            {'question_id': 'v2', 'split': 'valid', 'answer_ids': []}]
    predictions = [{'question_id': 'v1', 'answer_ids': ['D'], 'relation_confidence': .80},
                   {'question_id': 'v2', 'answer_ids': ['D'], 'relation_confidence': .45}]
    threshold, records = experiment.select_threshold(gold, predictions)
    assert threshold == .50
    assert next(row for row in records if row['threshold'] == .30)['entity_exact'] == 50
    assert next(row for row in records if row['threshold'] == .50)['entity_exact'] == 100
    test_gold = [{**row, 'split': 'test'} for row in gold]
    with pytest.raises(ValueError, match='validation rows only'):
        experiment.select_threshold(test_gold, predictions)
    all_wrong = [{**prediction, 'answer_ids': ['G']} for prediction in predictions[:1]]
    assert experiment.select_threshold(gold[:1], all_wrong)[0] == .30


@pytest.mark.parametrize('problem', ['duplicate_qa_id', 'movie_overlap', 'wrong_split', 'missing_label',
                                   'missing_movie_label', 'missing_facts', 'wrong_gold', 'empty_split', 'single_class'])
def test_bad_inputs_fail_before_a_run_directory_is_created(tiny_data, tmp_path, problem):
    data, entities, splits, qa = tiny_data
    if problem == 'duplicate_qa_id': qa.append(deepcopy(qa[0]))
    elif problem == 'movie_overlap': splits['test'].append('F1')
    elif problem == 'wrong_split': qa[0]['split'] = 'test'
    elif problem == 'missing_label': del entities['D']['label']
    elif problem == 'missing_movie_label': del entities['F1']['label']
    elif problem == 'missing_facts': del entities['F1']['facts']
    elif problem == 'wrong_gold': qa[0]['answer_ids'] = ['G']
    elif problem == 'empty_split': qa[:] = [row for row in qa if row['split'] != 'valid']
    elif problem == 'single_class': qa[:] = [row for row in qa if row['split'] != 'train' or row['relation_id'] == 'P57']
    write_json(data / 'entities.json', entities)
    write_json(data / 'splits.json', splits)
    write_jsonl(data / 'qa.jsonl', qa)
    with pytest.raises(ValueError):
        experiment.run(data, tmp_path / 'runs', 'invalid')
    assert not (tmp_path / 'runs/invalid').exists()


@pytest.mark.parametrize('run_id', ['../replace', '/absolute', '..', 'bad/name', 'bad\\name', ''])
def test_run_id_cannot_escape_output_directory(tiny_data, tmp_path, run_id):
    with pytest.raises(ValueError, match='run_id'):
        experiment.run(tiny_data[0], tmp_path / 'runs', run_id)


def test_duplicate_robustness_ids_and_non_test_sources_are_rejected(tiny_data, tmp_path):
    data, entities, splits, qa = tiny_data
    row = {**qa[-1], 'question_id': 'robust_one', 'variant': 'clean', 'source_question_id': qa[-1]['question_id']}
    write_jsonl(data / 'robustness.jsonl', [row, row])
    with pytest.raises(ValueError, match='Duplicate question_id'):
        experiment.run(data, tmp_path / 'runs', 'duplicate_robust')
    row['source_question_id'] = qa[0]['question_id']
    write_jsonl(data / 'robustness.jsonl', [row])
    with pytest.raises(ValueError, match='source must be a test question'):
        experiment.run(data, tmp_path / 'runs', 'leaking_robust')


def test_missing_processed_inputs_never_generate_synthetic_substitutes(tmp_path):
    with pytest.raises(FileNotFoundError, match='Required processed input'):
        experiment.run(tmp_path / 'missing', tmp_path / 'runs', 'no_data')
    assert not (tmp_path / 'runs').exists()


@pytest.mark.parametrize('field', ['movie', 'relation', 'reference'])
def test_robustness_must_preserve_the_exact_source_gold(tiny_data, tmp_path, field):
    data, entities, splits, qa = tiny_data
    # The row remains a valid test QA example, but is not the declared source.
    test_rows = [row for row in qa if row['split'] == 'test']
    source = test_rows[0]
    row = {**source, 'question_id': 'robust_bad', 'variant': 'alias',
           'source_question_id': source['question_id']}
    if field == 'relation':
        row.update({key: test_rows[1][key] for key in ('relation_id', 'answer_ids', 'reference_answers')})
    elif field == 'reference':
        row['reference_answers'] = ['Unrelated reference answer.']
    else:
        entities['F5'] = deepcopy(entities['F4'])
        entities['F5']['label'] = 'Sample Movie F5'
        splits['test'].append('F5')
        row['movie_id'] = 'F5'
        write_json(data / 'entities.json', entities)
        write_json(data / 'splits.json', splits)
    write_jsonl(data / 'robustness.jsonl', [row])
    with pytest.raises(ValueError, match='source movie, relation, gold IDs and references'):
        experiment.run(data, tmp_path / 'runs', 'changed_source')
    assert not (tmp_path / 'runs/changed_source').exists()


def test_manifest_pairs_require_gold_consistency_and_complete_variant_coverage(tiny_data):
    data, entities, splits, qa = tiny_data
    source = qa[-1]
    clean = {**source, 'question_id': 'clean', 'variant': 'clean'}
    alias = {**source, 'question_id': 'alias', 'variant': 'alias'}
    pair = {'clean_question_id': 'clean', 'variant_question_id': 'alias'}
    manifest = {'comparisons': {'alias': {'pairs': [pair]}}}
    assert len(experiment._validate_pairs([clean, alias], manifest)['alias']) == 1
    with pytest.raises(ValueError, match='entire variant pool'):
        experiment._validate_pairs([clean, alias, {**alias, 'question_id': 'another_alias'}], manifest)
    with pytest.raises(ValueError, match='gold must agree'):
        experiment._validate_pairs([clean, {**alias, 'reference_answers': ['Changed gold']}], manifest)
    with pytest.raises(ValueError, match='duplicate robustness pair'):
        experiment._validate_pairs([clean, alias], {'comparisons': {'alias': {'pairs': [pair, pair]}}})
