"""Acceptance harness invariants, separate from model/adapter behavior tests."""
import copy
import importlib.util
from pathlib import Path

import pytest

from kgqa.common import read_json, write_json

SPEC = importlib.util.spec_from_file_location('chinese_acceptance',
    Path(__file__).resolve().parents[1] / 'benchmarks/run_chinese_acceptance.py')
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


@pytest.fixture
def inputs():
    splits = {'train': ['Q1'], 'valid': ['Q2', 'Q3'], 'test': ['Q4', 'Q5']}
    entities = {f'Q{i}': {'label': f'Movie {i}', 'question_title': f'Movie {i}',
                          'label_zh': f'电影{i}', 'facts': {r: [f'A{j}'] for j, r in enumerate(benchmark.RELATIONS)}}
                for i in range(1, 6)}
    entities.update({f'A{i}': {'label': f'Answer {i}'} for i in range(4)})
    return entities, splits


def perfect_predictions(questions, gold):
    truth = {r['question_id']: r for r in gold}
    return [{'question_id': q['question_id'], 'movie_id': truth[q['question_id']]['movie_id'],
             'input_relation_id': truth[q['question_id']]['relation_id'],
             'relation_id': truth[q['question_id']]['relation_id'],
             'answer_ids': truth[q['question_id']]['answer_ids'],
             'refused': truth[q['question_id']]['expected_refusal'],
             'answer': '暂不支持。' if truth[q['question_id']]['expected_refusal'] else '中文回答。'}
            for q in questions]


def test_fixture_is_deterministic_balanced_and_movie_held_out(inputs):
    entities, splits = inputs
    one = benchmark.build_fixture(entities, splits, 2)
    two = benchmark.build_fixture(dict(reversed(list(entities.items()))), splits, 2)
    assert one == two
    questions, specifications, selected = one
    assert selected == {'valid': ['Q2', 'Q3'], 'test': ['Q4', 'Q5']}
    assert len(questions) == 2 * 2 * 4 * 3 + 20
    assert all('answer_ids' not in q and 'movie_id' not in q and 'relation_id' not in q for q in questions)
    assert len({q['question_id'] for q in questions}) == len(questions)
    assert not set(sum(selected.values(), [])) & set(splits['train'])
    assert {s['relation_id'] for s in specifications.values() if not s['expected_refusal']} == set(benchmark.RELATIONS)
    assert {q['variant'] for q in questions} == {'canonical', 'paraphrase', 'mixed_name', 'negative'}


def test_selection_never_silently_reduces_required_movie_count(inputs):
    entities, splits = inputs
    entities['Q2']['facts']['P57'] = []
    with pytest.raises(ValueError, match='only 1'):
        benchmark.build_fixture(entities, splits, 2)
    splits['test'].append('Q1')
    with pytest.raises(ValueError, match='overlap'):
        benchmark.build_fixture(entities, splits, 1)


def test_predictor_receives_original_questions_only(inputs):
    questions, _, _ = benchmark.build_fixture(*inputs, 1)
    seen = []
    def predict(question):
        assert isinstance(question, str)
        seen.append(question)
        return {'answer': '拒答', 'answer_ids': [], 'refused': True}
    predictions = benchmark.predict_questions(questions, predict)
    assert seen == [q['question'] for q in questions]
    assert [p['question_id'] for p in predictions] == [q['question_id'] for q in questions]


def test_scoring_detects_wrong_ids_missing_grounding_and_false_refusal(inputs):
    entities, splits = inputs
    questions, spec, _ = benchmark.build_fixture(entities, splits, 1)
    gold = benchmark.build_gold(spec, entities)
    predictions = perfect_predictions(questions, gold)
    summary, _ = benchmark.score_predictions(questions, gold, predictions)
    assert summary['status'] == 'PASS'
    broken = copy.deepcopy(predictions)
    broken[0]['answer_ids'] = ['WRONG']
    broken[1]['input_relation_id'] = 'WRONG'
    broken[2]['movie_id'] = 'WRONG'
    negative = next(p for p in broken if p['refused'])
    negative['refused'] = False  # Empty output alone must not count as a correct refusal.
    summary, rows = benchmark.score_predictions(questions, gold, broken)
    assert summary['status'] == 'FAIL'
    assert summary['positive_entity_exact'] < 1
    assert summary['input_entity_link_accuracy'] < 1
    assert summary['input_intent_accuracy'] < 1
    assert summary['negative_refusal_accuracy'] < 1
    assert not rows[0]['entity_exact']


def test_scoring_requires_exact_unique_question_coverage(inputs):
    entities, splits = inputs
    questions, spec, _ = benchmark.build_fixture(entities, splits, 1)
    gold = benchmark.build_gold(spec, entities)
    predictions = perfect_predictions(questions, gold)
    for broken in (predictions[:-1], predictions + [predictions[0]]):
        with pytest.raises(ValueError, match='unique and match'):
            benchmark.score_predictions(questions, gold, broken)


def test_existing_output_is_refused_before_loading_any_assets(tmp_path):
    output = tmp_path / 'already_exists'
    output.mkdir()
    marker = output / 'summary.json'
    marker.write_bytes(b'original evidence')
    with pytest.raises(FileExistsError):
        benchmark.run(tmp_path / 'absent-data', tmp_path / 'absent-model', output)
    assert marker.read_bytes() == b'original evidence'


def test_cli_refuses_existing_destination_and_exits_nonzero(tmp_path, capsys):
    assert benchmark.main(['--data-dir', str(tmp_path), '--model-path', str(tmp_path / 'none'),
                           '--output-dir', str(tmp_path)]) == 1
    assert 'FAIL' in capsys.readouterr().err


def test_http_sample_uses_actual_best_endpoint_and_balances_four_relations(inputs, monkeypatch):
    entities, splits = inputs
    questions, spec, _ = benchmark.build_fixture(entities, splits, 2)
    gold = benchmark.build_gold(spec, entities)
    by_question = {q['question']: spec[q['question_id']] for q in questions}
    calls = []
    def endpoint(url, body=None):
        calls.append((url, body))
        if body is None:
            return {'movie_count': 5, 'question_languages': ['zh', 'en'], 'neural': {'run_id': 'B0', 'available': True}}
        assert body['system'] == 'best'
        case = by_question[body['question']]
        return {'results': [{'system': 'best', 'ok': True, 'answer': '中文回答', 'language': 'zh',
                             'movie': {'id': case['movie_id']},
                             'answer_ids': entities[case['movie_id']]['facts'][case['relation_id']]}]}
    monkeypatch.setattr(benchmark, '_json_http', endpoint)
    summary = benchmark.http_neural_sample('http://127.0.0.1:7860', questions, gold, count=8)
    assert summary['count'] == 8 and summary['positive_entity_exact'] == 1
    assert len(calls) == 9
    sampled = [next(q for q in questions if q['question_id'] == p['question_id']) for p in summary['predictions']]
    assert len({(q['split'], q['question_id'].split('-')[-2]) for q in sampled}) == 8
    assert all(set(body) == {'question', 'system'} for _, body in calls[1:])


def test_v2_adds_year_uniformly_without_changing_movie_selection(inputs):
    entities, splits = inputs
    for movie in sum(splits.values(), []):
        entities[movie]['release_year'] = '1990'
    questions, _, selected = benchmark.build_fixture(entities, splits, 2)
    assert selected == {'valid': ['Q2', 'Q3'], 'test': ['Q4', 'Q5']}
    assert all('》（1990年版）' in q['question'] for q in questions if q['variant'] != 'negative')
    assert all(q['question_id'].startswith('zh-controlled-v2-') for q in questions)


def test_http_unavailable_neural_model_cannot_be_recorded_as_completed(inputs, monkeypatch):
    questions, spec, _ = benchmark.build_fixture(*inputs, 1)
    gold = benchmark.build_gold(spec, inputs[0])
    monkeypatch.setattr(benchmark, '_json_http', lambda url, body=None: {'question_languages': ['zh']} if body is None else {'results': [{'system': 'best', 'ok': False, 'error': 'Unavailable'}]})
    with pytest.raises(ValueError, match='inference failed'):
        benchmark.http_neural_sample('http://127.0.0.1:7860', questions, gold)


def test_http_sampling_balances_positive_intents_and_includes_negatives_before_requests(inputs):
    questions, _, _ = benchmark.build_fixture(*inputs, 2)
    sample = benchmark.select_http_questions(questions)
    assert len(sample) == 16
    assert len({(q['split'], q['question_id'].split('-')[-2]) for q in sample[:8]}) == 8
    assert all(q['variant'] == 'negative' for q in sample[8:])
    assert {q['split'] for q in sample[8:]} == {'valid', 'test'}


def test_old_http_server_is_explicitly_unscorable_not_zero_accuracy(inputs, monkeypatch):
    questions, spec, _ = benchmark.build_fixture(*inputs, 1)
    gold = benchmark.build_gold(spec, inputs[0])
    monkeypatch.setattr(benchmark, '_json_http', lambda *args: {'movie_count': 5000})
    with pytest.raises(ValueError, match='metrics are unavailable'):
        benchmark.http_neural_sample('http://127.0.0.1:7860', questions, gold)
