import random

import pytest

from kgqa import fetch
from kgqa.common import read_json, sha256, write_json


QUERY = ('SELECT DISTINCT ?film WHERE { ?film wdt:P31 wd:Q11424; wdt:P577 ?date. '
         'FILTER(?date >= "1980-01-01T00:00:00Z"^^xsd:dateTime && '
         '?date < "1990-01-01T00:00:00Z"^^xsd:dateTime) } LIMIT 800')
TIMESTAMP = '2026-10-18T16:15:08Z'


def cached_response(ids):
    return {
        'retrieved_at': TIMESTAMP, 'url': 'https://query.wikidata.org/sparql',
        'params': {'query': QUERY, 'format': 'json'},
        'response': {'results': {'bindings': [
            {'film': {'type': 'uri', 'value': f'http://www.wikidata.org/entity/{qid}'}} for qid in ids]}},
    }


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, 'ROOT', tmp_path)
    monkeypatch.setattr(fetch, 'RAW', tmp_path / 'data/raw')
    monkeypatch.setattr(fetch, 'setup_logging', lambda name: None)

    def save(ids=None, document=None):
        path = tmp_path / 'source/candidates_decade_1980.json'
        write_json(path, document if document is not None else cached_response(ids or ['Q1']))
        return path

    return tmp_path, save


@pytest.fixture
def entity_api(monkeypatch):
    calls = []

    def fake(ids, claims=True):
        calls.append((list(ids), claims))
        result = {}
        for qid in ids:
            entity = {'id': qid, 'labels': {'en': {'value': f'Entity {qid}'}}, 'lastrevid': 123}
            if claims:
                entity['claims'] = {
                    relation: [{'rank': 'normal', 'mainsnak': {
                        'snaktype': 'value', 'datatype': 'wikibase-item',
                        'datavalue': {'value': {'id': f'Q{9001 + index}'}}}}]
                    for index, relation in enumerate(fetch.RELATIONS)
                }
            result[qid] = entity
        return result

    monkeypatch.setattr(fetch, 'get_entities', fake)
    monkeypatch.setattr(fetch, 'request_json', lambda *args, **kwargs: pytest.fail('Unexpected network request'))
    monkeypatch.setattr(fetch, 'candidates', lambda: pytest.fail('Explicit cache must not trigger candidate queries'))
    return calls


def test_cached_1980s_pool_produces_exactly_requested_200_with_factual_provenance(files, entity_api):
    root, save = files
    ids = [f'Q{i}' for i in range(800, 0, -1)]
    source = save(ids)
    id_cache = root / 'data/raw/candidate_ids.json'
    write_json(id_cache, ['Q99999'])
    original_hash = sha256(id_cache)
    global_random_state = random.getstate()
    final = fetch.run(200, candidate_cache=source)
    assert random.getstate() == global_random_state
    expected_order = [f'Q{i}' for i in range(1, 801)]
    random.Random(42).shuffle(expected_order)
    movie_requests = [qid for batch, claims in entity_api if claims for qid in batch]
    assert movie_requests == expected_order[:200]
    assert [len(batch) for batch, claims in entity_api if claims] == [80, 80, 40]
    assert [movie['id'] for movie in final['movies']] == expected_order[:200]
    assert final['target'] == len(final['movies']) == 200
    assert read_json(root / 'data/raw/snapshot_200.json') == final
    assert not (root / 'data/raw/snapshot_1000.json').exists()
    assert sha256(id_cache) == original_hash
    pool = final['candidate_pool']
    assert pool['source_path'] == str(source.resolve())
    assert pool['source_sha256'] == sha256(source)
    assert pool['source_url'] == 'https://query.wikidata.org/sparql'
    assert pool['query_params'] == {'query': QUERY, 'format': 'json'}
    assert pool['query'] == QUERY and pool['retrieved_at'] == TIMESTAMP
    assert pool['unique_count'] == pool['binding_count'] == 800
    assert pool['shuffle_seed'] == 42
    assert 'cached SPARQL candidate response; limited scope defined by source query; not all Wikidata' in final['sampling']
    assert '1980-2024' not in final['sampling']


def test_cached_ids_are_numeric_sorted_deduplicated_and_seeded(files, entity_api):
    _, save = files
    source = save(['Q10', 'Q2', 'Q1', 'Q2'])
    first = fetch.run(3, source)
    source = save(['Q2', 'Q10', 'Q2', 'Q1'])
    second = fetch.run(3, source)
    expected = ['Q1', 'Q2', 'Q10']
    random.Random(42).shuffle(expected)
    assert [row['id'] for row in first['movies']] == [row['id'] for row in second['movies']] == expected
    assert first['candidate_pool']['binding_count'] == 4
    assert first['candidate_pool']['unique_count'] == 3


def test_default_id_cache_order_retained_without_claiming_original_query_scope(files, entity_api):
    root, _ = files
    path = root / 'data/raw/candidate_ids.json'
    write_json(path, ['Q3', 'Q1', 'Q2'])
    result = fetch.run(2)
    assert [movie['id'] for movie in result['movies']] == ['Q3', 'Q1']
    assert result['candidate_pool']['source_path'] == str(path.resolve())
    assert result['candidate_pool']['source_sha256'] == sha256(path)
    assert 'source-query scope not verified' in result['sampling']
    assert 'not all Wikidata' in result['sampling']
    assert '1980-2024' not in result['sampling']


@pytest.mark.parametrize('target', [0, -1, True, 1.5, '200'])
def test_invalid_target_is_rejected_before_any_requests(files, entity_api, target):
    _, save = files
    with pytest.raises(ValueError, match='target must be a positive integer'):
        fetch.run(target, save())
    assert entity_api == []


def malformed_document(kind):
    cached = cached_response(['Q1'])
    if kind == 'missing_response':
        del cached['response']
    elif kind == 'bad_bindings':
        cached['response']['results']['bindings'] = {}
    elif kind == 'bad_film':
        cached['response']['results']['bindings'] = [{'film': {'value': 'http://example.org/entity/Q1'}}]
    elif kind == 'bad_id':
        cached['response']['results']['bindings'] = [{'film': {'value': 'http://www.wikidata.org/entity/NOT_Q'}}]
    elif kind == 'missing_query':
        del cached['params']['query']
    elif kind == 'empty_query':
        cached['params']['query'] = ' '
    elif kind == 'missing_timestamp':
        del cached['retrieved_at']
    elif kind == 'empty_timestamp':
        cached['retrieved_at'] = ''
    elif kind == 'invalid_binding':
        cached['response']['results']['bindings'] = [None]
    elif kind == 'not_object':
        cached = []
    return cached


@pytest.mark.parametrize('kind', ['missing_response', 'bad_bindings', 'bad_film', 'bad_id', 'missing_query',
                                  'empty_query', 'missing_timestamp', 'empty_timestamp', 'invalid_binding', 'not_object'])
def test_malformed_candidate_response_is_rejected_before_requests(files, entity_api, kind):
    root, save = files
    source = save(document=malformed_document(kind))
    with pytest.raises(ValueError, match='Malformed cached SPARQL candidate response'):
        fetch.run(1, source)
    assert entity_api == []
    assert not (root / 'data/raw/snapshot_1.json').exists()


def test_empty_candidate_response_is_rejected_before_requests(files, entity_api):
    _, save = files
    source = save(document=cached_response([]))
    with pytest.raises(ValueError, match='non-empty list'):
        fetch.run(1, source)
    assert entity_api == []


@pytest.mark.parametrize('ids', [[], ['invalid'], 'Q1', [None], ['Q0']])
def test_invalid_default_candidate_id_list_is_rejected_before_requests(files, entity_api, ids):
    root, _ = files
    write_json(root / 'data/raw/candidate_ids.json', ids)
    with pytest.raises(ValueError, match='Candidate cache'):
        fetch.run(1)
    assert entity_api == []


def test_too_small_cached_pool_preserves_cache_and_does_not_write_incomplete_snapshot(files, entity_api):
    root, save = files
    source = save(['Q1', 'Q2'])
    original = source.read_bytes()
    with pytest.raises(RuntimeError, match='Only 2 eligible movies; requested 3'):
        fetch.run(3, source)
    assert source.read_bytes() == original
    assert not (root / 'data/raw/snapshot_3.json').exists()


def test_get_entities_keeps_standard_40_id_api_batches(monkeypatch):
    batches = []

    def request(url, params, cache_name):
        ids = params['ids'].split('|')
        batches.append(ids)
        assert url == 'https://www.wikidata.org/w/api.php'
        return {'entities': {qid: {'id': qid} for qid in ids}}

    monkeypatch.setattr(fetch, 'request_json', request)
    ids = [f'Q{i}' for i in range(1, 84)]
    assert set(fetch.get_entities(ids)) == set(ids)
    assert [len(batch) for batch in batches] == [40, 40, 3]
