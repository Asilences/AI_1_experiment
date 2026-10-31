from types import SimpleNamespace

import pytest

from kgqa.chinese import ChineseAdapter
from kgqa.common import REFUSAL
from kgqa.guarded import GuardedQA, RelationGraph
from kgqa.linker import EntityLinker


@pytest.fixture
def guarded():
    entities = {
        'Q1': {'label': 'Juniper Trail', 'aliases': ['杜松之路'],
               'facts': {'P57': ['Q2', 'Q3'], 'P136': ['Q4']}},
        'Q2': {'label': 'Jane Director'}, 'Q3': {'label': 'Alex Director'},
        'Q4': {'label': 'drama film'},
    }
    demo = SimpleNamespace(entities=entities, movie_ids=['Q1'], linker=EntityLinker(['Q1'], entities),
                           chinese=ChineseAdapter(entities, ['Q1']),
                           answer_a0=lambda _: {'answer': 'Juniper Trail was directed by Jane Director and Alex Director.',
                                                 'answer_ids': ['Q2', 'Q3'], 'evidence': [], 'latency_ms': 1})
    return GuardedQA(demo)


def test_verified_answer_uses_question_property_and_keeps_original(guarded):
    response = guarded.answer('Who directed Juniper Trail?')['result']
    assert response['guard']['action'] == 'accepted'
    assert response['answer_ids'] == ['Q2', 'Q3']
    assert len(response['evidence']) == 2 and all('director' in x for x in response['evidence'])
    assert response['answer'] == response['raw_answer'] and not response['fallback_used']


@pytest.mark.parametrize('text,reason', [
    ('Juniper Trail was directed by Jane Director.', 'missing_or_unrecognized_entities'),
    ('Juniper Trail was directed by drama film.', 'outside_relation_evidence'),
    ('Juniper Trail was not directed by Jane Director and Alex Director.', 'unverified_surface'),
    ('Juniper Trail was directed by Jane Director and Alex Director. It won every award.', 'unverified_surface'),
    (REFUSAL, 'model_refusal'),
])
def test_bad_generation_is_rejected_or_labelled_tool_fallback(guarded, monkeypatch, text, reason):
    monkeypatch.setattr(guarded, '_neural', lambda *args: {'answer': text, 'visible_answer_ids': ['Q2', 'Q3'],
                                                         'model_evidence': ['observed'], 'latency_ms': 2})
    rejected = guarded.answer('Who directed Juniper Trail?', 'neural', fallback=False)['result']
    assert rejected['refused'] and rejected['answer'] == REFUSAL
    assert rejected['raw_answer'] == text and rejected['guard']['reason'] == reason
    recovered = guarded.answer('Who directed Juniper Trail?', 'neural')['result']
    assert recovered['guard']['action'] == 'tool_fallback' and recovered['fallback_used']
    assert set(recovered['answer_ids']) == {'Q2', 'Q3'}
    assert recovered['raw_answer'] == text and not recovered['refused']


def test_unexecuted_conditions_never_reach_generator(guarded, monkeypatch):
    def unexpected(*args):
        raise AssertionError('Unsupported scope reached generator')
    monkeypatch.setattr(guarded, '_neural', unexpected)
    for question in ('Which directors of Juniper Trail were born before 1980?',
                     'What was the budget of Juniper Trail?'):
        response = guarded.answer(question, 'neural')
        assert response['interpretation']['status'] != 'ready'
        assert response['result']['refused'] and response['result']['raw_answer'] is None


def test_chinese_display_does_not_erase_model_output(guarded, monkeypatch):
    monkeypatch.setattr(guarded, '_neural', lambda *args: {'answer': REFUSAL, 'latency_ms': 2})
    result = guarded.answer('《杜松之路》的导演是谁？', 'neural')['result']
    assert result['language'] == 'zh' and '导演' in result['answer']
    assert result['raw_answer'] == REFUSAL and result['fallback_used']


def test_relation_filter_keeps_original_visibility_cap():
    base = SimpleNamespace(use_embeddings=False,
        entities={'M': {'label': 'Film', 'facts': {'P57': ['D1', 'D2'], 'P136': ['G1']}},
                  'D1': {'label': 'Director'}, 'D2': {'label': 'Beyond cap'}, 'G1': {'label': 'Genre'}},
        _retained_facts=lambda qid: [('Film | director | Director', 'D1'), ('Film | genre | Genre', 'G1')],
        facts=lambda qid: [('Film | director | Director', None), ('Film | genre | Genre', None)])
    graph = RelationGraph(base, 'M', 'P57')
    assert graph.facts('M') == [('Film | director | Director', None)]
    assert graph.fact_tail_ids('M') == ['D1']
    assert graph.facts('other') == []


def test_relation_filter_matches_real_graph_contract(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from kgqa.common import write_json
    from kgqa.neural import GraphData
    write_json(tmp_path / 'entities.json', {
        'M': {'label': 'Film', 'facts': {'P57': ['D'], 'P136': ['G']}},
        'D': {'label': 'Director'}, 'G': {'label': 'Genre'}})
    graph = RelationGraph(GraphData(False, tmp_path), 'M', 'P57')
    assert graph.fact_tail_ids('M') == ['D']
    assert graph.facts('M') == [('Film | director | Director', None)]


def test_nonvisible_entities_cannot_pass_as_model_supported(guarded, monkeypatch):
    monkeypatch.setattr(guarded, '_neural', lambda *args: {
        'answer': 'Juniper Trail was directed by Jane Director and Alex Director.',
        'visible_answer_ids': ['Q2'], 'latency_ms': 2})
    result = guarded.answer('Who directed Juniper Trail?', 'neural', fallback=False)['result']
    assert result['refused'] and result['guard']['reason'] == 'evidence_truncated'
