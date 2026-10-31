from copy import deepcopy
from pathlib import Path

import pytest

from kgqa.common import read_jsonl
from scripts.evaluate_semantic_dev import build_gold, evaluate, prepare_output, summarize
from test_explorer import explorer as explorer


def declared(**changes):
    row = {'question_id': 'one', 'language': 'en', 'category': 'facts',
           'question': 'Who appears in Inception?', 'expected_movie_ids': ['Q1'],
           'expected_properties': ['P161'], 'expected_operation': 'facts', 'expected_status': 'answered'}
    row.update(changes)
    return row


def test_reference_is_generated_from_declared_plan_and_kb(explorer):
    gold = build_gold([declared()], explorer)[0]
    assert {item['value'] for item in gold['reference_execution']['evidence']} == {'Q5', 'Q6'}
    assert gold['expected_plan']['movie_ids'] == ['Q1']
    assert gold['generated_text_human_reviewed'] is False
    with pytest.raises(ValueError, match='disagrees'):
        build_gold([declared(expected_properties=['P725'])], explorer)
    with pytest.raises(ValueError, match='unique'):
        build_gold([declared(), declared()], explorer)


def test_blocked_gold_never_uses_model_or_executes_graph(explorer, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No facts should be queried for a blocked reference')
    monkeypatch.setattr(explorer, 'execute', forbidden)
    gold = build_gold([declared(reference_plan_status='unsupported', expected_status='unsupported',
                                unhandled_constraints=['before 2010'], clarification='Not supported.')], explorer)
    assert gold[0]['reference_execution'] is None
    assert gold[0]['expected_plan']['unhandled_constraints'] == ['before 2010']


def test_tool_match_is_separate_from_plan_match_and_text_factuality(explorer):
    gold = build_gold([declared()], explorer)
    response = deepcopy(gold[0]['reference_execution'])
    response['plan'] = {**gold[0]['expected_plan'], 'properties': ['P57']}
    response['answer'] = 'An arbitrary unsupported sentence cannot be validated by these tool metrics.'
    rows, metrics = evaluate(gold, lambda case: response)
    assert metrics['tool_result_match'] == {'numerator': 1, 'denominator': 1, 'percent': 100}
    assert metrics['declared_plan_match']['percent'] == 0
    assert metrics['structured_task_match']['percent'] == 0
    assert metrics['text_semantics_human_reviewed'] is False
    assert rows[0]['response']['answer'] == response['answer']


def test_denominators_status_and_fallback_preserve_raw_response(explorer):
    gold = build_gold([
        declared(),
        declared(question_id='blocked', category='unsupported', reference_plan_status='unsupported',
                 expected_status='unsupported', unhandled_constraints=['before 2010'], clarification='Not supported.'),
    ], explorer)
    answer = deepcopy(gold[0]['reference_execution'])
    answer.update(plan=gold[0]['expected_plan'], fallback_used=True, raw_answer='wrong raw answer')
    outputs = iter([answer, {'status': 'unsupported', 'answer': 'Not supported.', 'evidence': [], 'items': []}])
    rows, summary = evaluate(gold, lambda case: next(outputs))
    assert summary['declared_plan_match']['denominator'] == 1
    assert summary['status_match']['denominator'] == 2
    assert summary['structured_task_match']['percent'] == 100
    assert summary['fallback_rate']['percent'] == 50
    assert summary['by_category']['unsupported']['count'] == 1
    assert rows[0]['response']['raw_answer'] == 'wrong raw answer'


def test_blocked_status_with_returned_facts_is_not_a_structured_success(explorer):
    gold = build_gold([declared(reference_plan_status='unsupported', expected_status='unsupported',
                                unhandled_constraints=['date'], clarification='Not supported.')], explorer)
    response = {'status': 'unsupported', 'evidence': [], 'items': [],
                'execution': {'evidence': [{'statement_id': 'leaked'}], 'items': []}}
    _, summary = evaluate(gold, lambda case: response)
    assert summary['status_match']['percent'] == 100
    assert summary['structured_task_match']['percent'] == 0


def test_per_question_evidence_can_be_retained_before_later_failure(explorer):
    gold = build_gold([declared(), declared(question_id='second')], explorer)
    saved = []
    def predict(case):
        if case['question_id'] == 'second':
            raise RuntimeError('simulated later failure')
        result = deepcopy(case['reference_execution'])
        result['plan'] = case['expected_plan']
        return result
    with pytest.raises(RuntimeError):
        evaluate(gold, predict, on_record=saved.append)
    assert len(saved) == 1 and saved[0]['question_id'] == 'one'


def test_nonempty_output_is_never_overwritten(tmp_path):
    directory = prepare_output(tmp_path/'run')
    (directory/'keep.txt').write_text('original', encoding='utf-8')
    with pytest.raises(FileExistsError):
        prepare_output(directory)
    assert (directory/'keep.txt').read_text(encoding='utf-8') == 'original'
    assert summarize([])['declared_plan_match']['percent'] is None


def test_default_cases_are_explicit_development_cases_and_bilingual():
    path = Path(__file__).resolve().parents[1]/'benchmarks/semantic_dev_v1.jsonl'
    rows = read_jsonl(path)
    assert len(rows) == 20 and len({row['question_id'] for row in rows}) == 20
    assert sum(row['language'] == 'zh' for row in rows) == 10
    assert sum(row['language'] == 'en' for row in rows) == 10
    assert {'intersection', 'reverse', 'compare'} <= {row['expected_operation'] for row in rows}
    assert {'missing_fact', 'fuzzy_confirmation', 'unsupported_condition'} <= {row['category'] for row in rows}
