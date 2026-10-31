"""Synthetic orchestration tests, not evidence of a real model's NLU ability."""
from collections import deque

import pytest

from kgqa.common import sha256, write_json, write_jsonl
from kgqa.candidates import CandidateIndex
from kgqa.explorer import Explorer
from kgqa.frozen_llm import LLMError
from kgqa.semantic_qa import SemanticQA
from kgqa.v2_data import PROPERTIES


def row(movie, prop, value, qualifiers=None):
    return {'id': f'{movie}${prop}-{value}', 'subject': movie, 'property': prop, 'value': value,
            'type': PROPERTIES[prop][2], 'snaktype': 'value', 'rank': 'normal',
            'qualifiers': qualifiers or {}, 'references': [], 'revision': 1,
            'source_url': f'https://www.wikidata.org/wiki/{movie}', 'retrieved_at': '2026-10-31'}


@pytest.fixture
def catalogue(tmp_path):
    movies = [{'id': 'Q1', 'label': 'Inception', 'label_zh': '盗梦空间', 'aliases': [], 'release_year': 2010},
              {'id': 'Q2', 'label': 'Interstellar', 'label_zh': '星际穿越', 'aliases': [], 'release_year': 2014},
              {'id': 'Q3', 'label': 'Dunkirk', 'label_zh': '敦刻尔克', 'aliases': [], 'release_year': 2017}]
    entities = {m['id']: m for m in movies} | {
        'Q4': {'label': 'Christopher Nolan', 'label_zh': '克里斯托弗·诺兰'},
        'Q5': {'label': 'Actor One'}, 'Q6': {'label': 'Actor Two'},
        'Q30': {'label': 'United States'}, 'Q4917': {'label': 'US dollar'}}
    rows = [row('Q1', 'P57', 'Q4'), row('Q2', 'P57', 'Q4'), row('Q3', 'P57', 'Q4'),
            row('Q1', 'P161', 'Q5'), row('Q1', 'P161', 'Q6'), row('Q2', 'P161', 'Q5'),
            row('Q1', 'P577', {'time': '+2010-07-16T00:00:00Z', 'precision': 11,
                              'calendarmodel': 'http://www.wikidata.org/entity/Q1985727'}),
            row('Q1', 'P2142', {'amount': '+800.25', 'unit': 'http://www.wikidata.org/entity/Q4917'})]
    write_json(tmp_path / 'movies.json', movies)
    write_json(tmp_path / 'entities.json', entities)
    write_jsonl(tmp_path / 'statements.jsonl', rows)
    for name in ('coverage.json', 'provenance.json'):
        write_json(tmp_path / name, {})
    names = ['movies.json', 'entities.json', 'statements.jsonl', 'coverage.json', 'provenance.json']
    write_json(tmp_path / 'manifest.json', {'schema': 'movie-catalog-v2/1', 'movie_count': 3,
               'files': {name: sha256(tmp_path / name) for name in names}})
    return Explorer(tmp_path)


class FakeLLM:
    def __init__(self, *responses):
        self.responses = deque(responses)
        self.requests = []
        self.last_call = None

    def metadata(self):
        return {'model': 'synthetic-protocol-only', 'frozen': True, 'revision_verified': False}

    def complete_json(self, messages, max_tokens):
        self.requests.append({'messages': messages, 'max_tokens': max_tokens})
        self.last_call = {'elapsed_ms': 0, 'synthetic': True}
        if not self.responses:
            raise AssertionError('Unexpected model call in synthetic test')
        value = self.responses.popleft()
        if isinstance(value, Exception):
            raise value
        return value


def plan(ids=None, properties=None, operation='facts', language='en', **changes):
    value = {'schema': 'kgqa-plan/1', 'language': language, 'status': 'ready',
             'movie_ids': ids or ['Q1'], 'properties': properties or ['P57'],
             'operation': operation, 'unhandled_constraints': [], 'clarification': None, 'limit': 30}
    return value | changes


def no_execute(*args, **kwargs):
    raise AssertionError('A blocked/ambiguous question must not execute a graph query')


@pytest.mark.parametrize('mention,question', [('Zyxwv Utopian', 'Who wrote Zyxwv Utopian?'),
                                            ('Inceptoin', 'Who put Inceptoin on screen?')])
def test_unknown_or_approximate_title_is_not_silently_queried(catalogue, monkeypatch, mention, question):
    monkeypatch.setattr(catalogue, 'execute', no_execute)
    client = FakeLLM({'mentions': [mention], 'language': 'en'})
    result = SemanticQA(catalogue, client).answer(question)
    assert result['status'] in {'needs_clarification', 'needs_confirmation'}
    assert not result['evidence'] and result['execution'] is None
    assert len(client.requests) == 1 and result['raw_plan'] is None


def test_no_named_movie_and_invented_mention_are_not_queried(catalogue, monkeypatch):
    monkeypatch.setattr(catalogue, 'execute', no_execute)
    result = SemanticQA(catalogue, FakeLLM({'mentions': [], 'language': 'en'})).answer('Who directed it?')
    assert result['status'] == 'needs_clarification'
    with pytest.raises(LLMError) as error:
        SemanticQA(catalogue, FakeLLM({'mentions': ['Inception']})).answer('Who directed it?')
    assert error.value.code == 'invented_mention'


def test_confirmed_title_must_be_one_of_current_question_candidates(catalogue, monkeypatch):
    monkeypatch.setattr(catalogue, 'execute', no_execute)
    _, cards, _ = catalogue.index.mentions('Inceptoin')
    unrelated = next(q for q in catalogue.movies if q not in {c['id'] for c in cards})
    client = FakeLLM({'mentions': ['Inceptoin'], 'language': 'en'})
    result = SemanticQA(catalogue, client).answer('Who directed Inceptoin?', confirmed=[unrelated])
    assert result['execution'] is None and not result['evidence']
    assert result['status'] in {'needs_confirmation', 'needs_clarification'}
    with pytest.raises(ValueError):
        SemanticQA(catalogue, FakeLLM({'mentions': ['Inception']})).answer(
            'Who directed Inception?', confirmed=['Q3'])


def test_confirmed_typo_and_structured_plan_bypass_legacy_regex(catalogue, monkeypatch):
    monkeypatch.setattr(catalogue, 'plan', no_execute)
    expected = 'Inception was directed by Christopher Nolan.'
    client = FakeLLM({'mentions': ['Inceptoin']}, plan(),
                     {'answer': expected, 'citations': ['Q1$P57-Q4']})
    result = SemanticQA(catalogue, client).answer('Who put Inceptoin on screen?', confirmed=['Q1'])
    assert result['status'] == 'answered' and result['answer'] == 'Inception was directed by Christopher Nolan (Q4).'
    assert result['raw_answer'] == expected and result['fallback_used'] is True
    assert result['guard']['action'] == 'tool_template'
    assert result['execution']['validated_plan']['movie_ids'] == ['Q1']
    assert len(result['calls']) == 3


def test_plan_cannot_omit_a_resolved_second_movie(catalogue, monkeypatch):
    monkeypatch.setattr(catalogue, 'execute', no_execute)
    client = FakeLLM({'mentions': ['Inception', 'Interstellar']}, plan())
    result = SemanticQA(catalogue, client).answer('Who directed Inception and Interstellar?')
    assert result['status'] == 'needs_clarification' and not result['evidence']
    assert result['execution'] is None and len(client.requests) == 2


def test_extraction_cannot_drop_a_clear_second_movie(catalogue, monkeypatch):
    monkeypatch.setattr(catalogue, 'execute', no_execute)
    client = FakeLLM({'mentions': ['Inception']}, plan())
    result = SemanticQA(catalogue, client).answer('Who directed Inception and Interstellar?')
    assert result['status'] == 'needs_clarification' and not result['evidence']
    assert result['execution'] is None


def test_extraction_cannot_drop_an_unknown_quoted_second_movie(catalogue, monkeypatch):
    monkeypatch.setattr(catalogue, 'execute', no_execute)
    client = FakeLLM({'mentions': ['Inception']}, plan())
    result = SemanticQA(catalogue, client).answer('Who directed Inception and "Zyxwv Utopian"?')
    assert result['status'] == 'needs_clarification' and result['execution'] is None


def test_pronoun_extraction_cannot_authorize_the_short_title_it(catalogue, monkeypatch):
    movie = {'id': 'Q7', 'label': 'It', 'label_zh': '', 'aliases': [], 'release_year': 2017}
    catalogue.movies['Q7'] = movie
    catalogue.entities['Q7'] = movie
    catalogue.index = CandidateIndex(list(catalogue.movies.values()))
    catalogue.by_pair['Q7', 'P57'].append(row('Q7', 'P57', 'Q4'))
    monkeypatch.setattr(catalogue, 'execute', no_execute)
    client = FakeLLM({'mentions': ['it']}, plan(ids=['Q7']))
    result = SemanticQA(catalogue, client).answer('Who directed it?')
    assert result['status'] == 'needs_clarification' and result['execution'] is None


def test_quoted_short_title_can_still_be_queried(catalogue):
    movie = {'id': 'Q7', 'label': 'It', 'label_zh': '', 'aliases': [], 'release_year': 2017}
    catalogue.movies['Q7'] = movie
    catalogue.entities['Q7'] = movie
    catalogue.index = CandidateIndex(list(catalogue.movies.values()))
    catalogue.by_pair['Q7', 'P57'].append(row('Q7', 'P57', 'Q4'))
    client = FakeLLM({'mentions': ['It']}, plan(ids=['Q7']),
                     {'answer': 'It was directed by Christopher Nolan.', 'citations': ['Q7$P57-Q4']})
    result = SemanticQA(catalogue, client).answer('Who directed "It"?')
    assert result['status'] == 'answered' and result['plan']['movie_ids'] == ['Q7']
    assert result['fallback_used'] is True


def test_ready_plan_cannot_silently_drop_an_explicit_unsupported_date_filter(catalogue, monkeypatch):
    monkeypatch.setattr(catalogue, 'execute', no_execute)
    client = FakeLLM({'mentions': ['Inception']}, plan(properties=['P161']))
    result = SemanticQA(catalogue, client).answer('Which cast members of Inception appeared after 2010?')
    assert result['status'] in {'unsupported', 'needs_clarification'} and result['execution'] is None


def test_unsupported_plan_retains_condition_and_does_not_read_facts(catalogue):
    class NoGraphReads:
        def __getitem__(self, key):
            raise AssertionError('Unsupported plan read graph facts')
    catalogue.by_pair = NoGraphReads()
    value = plan(status='unsupported', unhandled_constraints=['after 2010'],
                 clarification='Date filtering is not implemented.')
    client = FakeLLM({'mentions': ['Inception']}, value)
    result = SemanticQA(catalogue, client).answer('Who directed Inception after 2010?')
    assert result['status'] == 'unsupported' and not result['evidence']
    assert result['plan']['unhandled_constraints'] == ['after 2010']
    assert result['raw_answer'] is None and len(client.requests) == 2


@pytest.mark.parametrize('generated', [
    {'answer': 'Inception was directed by Christopher Nolan.', 'citations': ['invented-statement']},
    {'answer': 'Inception was directed by someone.', 'citations': ['Q1$P57-Q4']},
    {'answer': 'Inception was not directed by Christopher Nolan.', 'citations': ['Q1$P57-Q4']},
    {'answer': 'Inception was directed by Christopher Nolan.', 'citations': []},
])
def test_missing_slots_wrong_citations_or_negation_preserve_raw_and_use_template(catalogue, generated):
    client = FakeLLM({'mentions': ['Inception']}, plan(), generated)
    result = SemanticQA(catalogue, client).answer('Who directed Inception?')
    assert result['status'] == 'answered'
    assert result['fallback_used'] is True and result['guard']['action'] == 'tool_template'
    assert result['raw_answer'] == generated['answer']
    assert 'Christopher Nolan' in result['answer']
    assert 'not directed' not in result['answer']


@pytest.mark.parametrize('text', [
    'Christopher Nolan starred in Inception.',
    'Christopher Nolan was directed by Inception.',
    'Inception was directed by Christopher Nolan. Inception won an Academy Award.',
    'Inception was directed by Christopher Nolan (Q4). It won an Academy Award.',
    '《盗梦空间》由克里斯托弗·诺兰主演。',
    '克里斯托弗·诺兰由《盗梦空间》执导。',
    '《盗梦空间》记录的导演为克里斯托弗·诺兰 (Q4)。该片获得奥斯卡奖。',
])
def test_correct_citation_does_not_authorize_wrong_predicate_direction_or_extra_claim(catalogue, text):
    language = 'zh' if '《' in text else 'en'
    title = '盗梦空间' if language == 'zh' else 'Inception'
    client = FakeLLM({'mentions': [title]}, plan(language=language),
                     {'answer': text, 'citations': ['Q1$P57-Q4']})
    result = SemanticQA(catalogue, client).answer('《盗梦空间》的导演是谁？' if language == 'zh' else 'Who directed Inception?')
    assert result['fallback_used'] is True
    assert result['raw_answer'] == text
    assert result['guard']['method'] == 'controlled_evidence_text_v2'
    assert result['answer'] == ('《盗梦空间》记录的导演为克里斯托弗·诺兰 (Q4)。' if language == 'zh' else
                                'Inception was directed by Christopher Nolan (Q4).')


def test_only_controlled_evidence_sentence_is_accepted_and_names_are_not_rewritten(catalogue):
    expected = 'Inception was directed by Christopher Nolan (Q4).'
    client = FakeLLM({'mentions': ['Inception']}, plan(),
                     {'answer': expected, 'citations': ['Q1$P57-Q4']})
    result = SemanticQA(catalogue, client).answer('Who directed Inception?')
    assert result['answer'] == expected and result['raw_answer'] == expected
    assert result['fallback_used'] is False
    assert result['guard']['reason'] == 'controlled_text_match'


@pytest.mark.parametrize('prop,expected', [('P577', '2010-07-16'), ('P2142', '800.25')])
def test_dates_and_amounts_use_lossless_template_without_generation(catalogue, prop, expected):
    client = FakeLLM({'mentions': ['Inception']}, plan(properties=[prop]))
    result = SemanticQA(catalogue, client).answer('Show the records for Inception.')
    assert result['fallback_used'] is True and result['raw_answer'] is None
    assert result['guard']['reason'] == 'long_or_typed_result'
    assert expected in result['answer'] and len(client.requests) == 2
    if prop == 'P577':
        assert 'precision=11' in result['answer']
    else:
        assert 'US dollar' in result['answer']


def test_typed_template_keeps_regional_qualifier(catalogue):
    catalogue.by_pair['Q1', 'P577'][0]['qualifiers'] = {
        'P291': [{'snaktype': 'value', 'datavalue': {'value': {'id': 'Q30'}}}]}
    client = FakeLLM({'mentions': ['Inception']}, plan(properties=['P577']))
    result = SemanticQA(catalogue, client).answer('Show the recorded release dates for Inception.')
    assert result['fallback_used'] is True and 'United States' in result['answer']
    assert result['evidence'][0]['qualifiers']['P291'][0]['datavalue']['value']['id'] == 'Q30'


def test_long_list_is_paginated_after_query_and_not_compressed_by_model(catalogue):
    client = FakeLLM({'mentions': ['Inception']}, plan(properties=['P161'], limit=1))
    result = SemanticQA(catalogue, client).answer('List one cast member of Inception.')
    assert result['total'] == 2 and result['limit'] == 1 and result['next_offset'] == 1
    assert result['fallback_used'] is True and result['raw_answer'] is None
    assert '2 recorded items' in result['answer'] and len(client.requests) == 2


def test_service_disconnect_is_explicit_before_query_and_labeled_after_query(catalogue, monkeypatch):
    failing = FakeLLM(LLMError('service_unavailable', 'Model service unavailable.'))
    with pytest.raises(LLMError) as error:
        SemanticQA(catalogue, failing).answer('Who directed Inception?')
    assert error.value.code == 'service_unavailable'
    client = FakeLLM({'mentions': ['Inception']}, plan(),
                     LLMError('service_unavailable', 'Model service unavailable.'))
    result = SemanticQA(catalogue, client).answer('Who directed Inception?')
    assert result['fallback_used'] is True and result['raw_answer'] is None
    assert result['guard']['reason'] == 'service_unavailable'
    assert 'Christopher Nolan' in result['answer']


def test_schema_repair_is_bounded_and_not_a_hidden_rule_fallback(catalogue, monkeypatch):
    monkeypatch.setattr(catalogue, 'plan', no_execute)
    client = FakeLLM({'mentions': ['Inception']}, {'wrong': 'schema'}, plan(),
                     {'answer': 'Inception was directed by Christopher Nolan.', 'citations': ['Q1$P57-Q4']})
    result = SemanticQA(catalogue, client).answer('Who put Inception on screen?')
    assert result['status'] == 'answered' and len(client.requests) == 4
    assert result['raw_plan'] == plan()
