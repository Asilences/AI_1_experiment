import pytest

from kgqa.common import REFUSAL
from kgqa.trainfree_experiment import build_gold, entity_f1, evaluate_cases, refusal_flag, summarize


def test_gold_reads_frozen_facts_and_rejects_missing_bindings():
    entities = {'M': {'facts': {'P57': ['D2', 'D1']}}, 'D1': {}, 'D2': {}}
    case = {'question_id': 'one', 'question': 'Who directed Film?', 'expected_status': 'ready',
            'movie_id': 'M', 'relation_id': 'P57'}
    result = build_gold([case], entities, ['M'])[0]
    assert result['answer_ids'] == ['D1', 'D2']
    assert result['reference_facts'] == [['M', 'P57', 'D1'], ['M', 'P57', 'D2']]
    with pytest.raises(ValueError, match='Missing reference'):
        build_gold([{**case, 'relation_id': 'P364'}], entities, ['M'])
    with pytest.raises(ValueError, match='unique'):
        build_gold([case, case], entities, ['M'])


def test_unknown_names_are_not_automatically_treated_as_refusal():
    assert not refusal_flag({'answer': 'Someone invented this answer.', 'answer_ids': []})
    assert refusal_flag({'answer': '中文拒答', 'original_answer': REFUSAL})
    assert refusal_flag({'answer': 'Please clarify', 'refused': True})
    assert not refusal_flag({'answer': 'Correct tool answer', 'raw_answer': REFUSAL, 'refused': False})


def test_scoring_uses_declared_counts_and_preserves_outputs():
    cases = [
        {'question_id': 'a', 'question': 'first', 'expected_status': 'ready', 'movie_id': 'M',
         'relation_id': 'P57', 'answer_ids': ['D1', 'D2'], 'category': 'supported'},
        {'question_id': 'b', 'question': 'second', 'expected_status': 'unanswerable',
         'answer_ids': [], 'category': 'unsupported'},
        {'question_id': 'c', 'question': 'third', 'expected_status': 'needs_clarification',
         'answer_ids': [], 'category': 'ambiguous'},
    ]
    predictions = iter([
        {'answer': 'partial', 'answer_ids': ['D1'], 'movie_id': 'M', 'relation': 'P57'},
        {'answer': 'hallucination', 'answer_ids': []},
        {'interpretation': {'status': 'needs_clarification'},
         'result': {'answer': REFUSAL, 'answer_ids': [], 'refused': True}},
    ])
    rows, scores = evaluate_cases(cases, lambda _: next(predictions), lambda _: [])
    assert scores['supported_entity_f1'] == pytest.approx(100 * 2 / 3)
    assert scores['supported_entity_exact'] == 0
    assert scores['unanswerable_refusal_accuracy'] == 0
    assert scores['nonanswerable_unsafe_answer_rate'] == 50
    assert scores['clarification_status_accuracy'] == 100
    assert rows[1]['response']['answer'] == 'hallucination'
    assert scores['by_category']['unsupported']['count'] == 1


def test_f1_empty_cases_and_empty_summary():
    assert entity_f1([], []) == 1
    assert entity_f1(['a'], []) == 0
    assert summarize([])['p95_latency_ms'] is None
