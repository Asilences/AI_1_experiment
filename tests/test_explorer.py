import pytest
import http.client
import json
import threading

from kgqa.common import sha256, write_json, write_jsonl
from kgqa.explorer import Explorer
from kgqa.v2_data import PROPERTIES
from kgqa.web import QAService, make_server


@pytest.fixture
def explorer(tmp_path):
    movies = [{'id': 'Q1', 'label': 'Inception', 'label_zh': '盗梦空间', 'aliases': [], 'release_year': 2010},
              {'id': 'Q2', 'label': 'Interstellar', 'label_zh': '星际穿越', 'aliases': [], 'release_year': 2014},
              {'id': 'Q3', 'label': 'Dunkirk', 'label_zh': '敦刻尔克', 'aliases': [], 'release_year': 2017}]
    entities = {m['id']: m for m in movies} | {'Q4': {'label': 'Christopher Nolan', 'label_zh': '克里斯托弗·诺兰'},
                  'Q5': {'label': 'Actor One'}, 'Q6': {'label': 'Actor Two'}, 'Q4917': {'label': 'US dollar'}}
    def row(movie, prop, value, qualifiers=None, kind=None):
        return {'id': f'{movie}${prop}-{value}', 'subject': movie, 'property': prop, 'value': value,
                'type': kind or PROPERTIES[prop][2], 'snaktype': 'value', 'rank': 'normal',
                'qualifiers': qualifiers or {}, 'references': [], 'revision': 7,
                'source_url': f'https://www.wikidata.org/wiki/{movie}', 'retrieved_at': '2026-10-29'}
    rows = [row('Q1', 'P57', 'Q4'), row('Q2', 'P57', 'Q4'), row('Q3', 'P57', 'Q4'),
            row('Q1', 'P161', 'Q5'), row('Q1', 'P161', 'Q6'), row('Q2', 'P161', 'Q5'),
            row('Q1', 'P577', {'time': '+2010-07-16T00:00:00Z', 'precision': 11,
                             'calendarmodel': 'http://www.wikidata.org/entity/Q1985727'}),
            row('Q1', 'P2142', {'amount': '+800.25', 'unit': 'http://www.wikidata.org/entity/Q4917'}),
            row('Q2', 'P2142', {'amount': '+700.15', 'unit': 'http://www.wikidata.org/entity/Q4917'})]
    write_json(tmp_path/'movies.json', movies)
    write_json(tmp_path/'entities.json', entities)
    write_jsonl(tmp_path/'statements.jsonl', rows)
    for name in ('coverage.json', 'provenance.json'):
        write_json(tmp_path/name, {})
    names = ['movies.json', 'entities.json', 'statements.jsonl', 'coverage.json', 'provenance.json']
    write_json(tmp_path/'manifest.json', {'schema': 'movie-catalog-v2/1', 'movie_count': 3,
               'files': {name: sha256(tmp_path/name) for name in names}})
    return Explorer(tmp_path)


def test_bilingual_multi_property_partial_and_source(explorer):
    answer = explorer.answer('盗梦空间的导演、演员和编剧是谁？')
    assert answer['status'] == 'partial'
    assert answer['missing'] == [{'movie_id': 'Q1', 'property': 'P58'}]
    assert len(answer['evidence']) == 3
    assert '克里斯托弗' in answer['answer']
    assert all(e['source_url'].startswith('https://www.wikidata.org/wiki/') for e in answer['evidence'])
    assert explorer.answer('When was Inception released?')['evidence'][0]['value']['precision'] == 11


def test_approximate_requires_explicit_confirmation_and_pagination(explorer):
    answer = explorer.answer('Who is in the cast of Inceptoin?')
    assert answer['status'] == 'needs_confirmation' and not answer['evidence']
    confirmed = explorer.answer('Who is in the cast of Inceptoin?', ['Q1'], limit=1)
    assert confirmed['status'] == 'answered' and confirmed['total'] == 2 and confirmed['next_offset'] == 1
    page2 = explorer.answer('Who is in the cast of Inceptoin?', ['Q1'], offset=1, limit=1)
    assert page2['evidence'][0]['value'] != confirmed['evidence'][0]['value']


def test_intersection_and_two_hop_paths(explorer):
    shared = explorer.answer('盗梦空间和星际穿越有哪些共同演员？')
    assert shared['status'] == 'answered' and shared['items'][0]['entity_id'] == 'Q5'
    assert {e['movie_id'] for e in shared['evidence']} == {'Q1', 'Q2'}
    other = explorer.answer('盗梦空间的导演还拍过哪些电影？')
    assert {i['movie_id'] for i in other['items']} == {'Q2', 'Q3'}
    assert all(i['via'] == 'Q4' for i in other['items'])
    assert len(other['evidence']) == 3
    compared = explorer.answer('Compare the directors of Inception and Interstellar')
    assert compared['status'] == 'answered' and len(compared['evidence']) == 2


def test_amounts_do_not_assume_global_scope_or_currency(explorer):
    question = 'Compare the box office of Inception and Interstellar'
    answer = explorer.answer(question)
    assert answer['status'] == 'not_comparable' and 'comparison' not in answer
    for qid in ('Q1', 'Q2'):
        explorer.by_pair[qid, 'P2142'][0]['qualifiers'] = {
            'P3005': [{'snaktype': 'value', 'datavalue': {'value': {'id': 'Q2'}}}]}
    answer = explorer.answer(question)
    assert answer['comparison']['winner'] == 'Q1'
    assert answer['comparison']['difference'] == '100.10'


@pytest.mark.parametrize('question', [
    '盗梦空间的预算是多少？', '盗梦空间的主演是谁？', '盗梦空间的中文配音语言是什么？',
    '盗梦空间在中国什么时候上映？', 'Which country banned Inception?',
    'Was Inception directed by Christopher Nolan?', '盗梦空间的导演妻子是谁？',
    'What is the plot of Inception?', 'Inception cast before 2010',
    'When was Inception released in France?', 'Which country was Inception filmed in?',
])
def test_unsupported_constraints_never_turn_into_other_facts(explorer, question):
    answer = explorer.answer(question)
    assert answer['status'] == 'unsupported' and not answer['evidence']


def test_invalid_confirmations_and_offsets_fail(explorer):
    for ids in (['Q999'], 'Q1', [1]):
        with pytest.raises(ValueError):
            explorer.answer('cast', ids)
    for value in (-1, True, '1'):
        with pytest.raises(ValueError):
            explorer.answer('Q1 cast', offset=value)


def test_real_http_explore_confirmation_and_validation(explorer, tmp_path):
    service = QAService(data_dir=tmp_path/'absent', model_path=tmp_path/'absent.pkl')
    service.explorer = explorer
    server = make_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def post(payload):
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=10)
        try:
            connection.request('POST', '/api/explore', json.dumps(payload), {'Content-Type': 'application/json'})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()
    try:
        status, data = post({'question': 'Who wrote Inceptoin?'})
        assert status == 200 and data['status'] == 'needs_confirmation'
        status, data = post({'question': 'Who directed Inceptoin?', 'movie_ids': ['Q1']})
        assert status == 200 and data['evidence'][0]['value'] == 'Q4'
        status, data = post({'question': 'Q1 cast', 'movie_ids': ['Q999']})
        assert status == 400 and data['error']
        status, _ = post({'question': 'Q1 cast', 'offset': True})
        assert status == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
