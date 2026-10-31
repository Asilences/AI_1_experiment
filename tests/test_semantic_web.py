"""HTTP integration checks; fake inference here is not a model benchmark."""
import http.client
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from kgqa.web import QAService, make_server


@pytest.fixture
def semantic_service(tmp_path):
    service = QAService(tmp_path / 'missing-data', tmp_path / 'missing.pkl', catalog_dir=tmp_path / 'missing-catalogue')
    service.explorer = SimpleNamespace(movies={'Q1': {'label': 'Example Film'}}, rows=[], manifest={})
    service.catalog_error = ''
    return service


@pytest.fixture
def semantic_server(semantic_service):
    server = make_server(semantic_service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def request(server, path='/api/semantic', payload=None, headers=None):
    connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
    try:
        body = None if payload is None else json.dumps(payload)
        connection.request('GET' if payload is None else 'POST', path, body,
                           {'Content-Type': 'application/json', **(headers or {})})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_status_distinguishes_catalogue_from_unchecked_model(semantic_service, semantic_server, monkeypatch):
    monkeypatch.setenv('KGQA_LLM_API_KEY', 'do-not-expose-this-value')
    status, data = request(semantic_server, '/api/status')
    assert status == 200
    assert data['semantic']['available']
    assert data['semantic']['model_status'] == 'not_checked'
    assert semantic_service.semantic_engine is None
    assert 'do-not-expose-this-value' not in json.dumps(data)


def test_http_semantic_loads_lazily_and_preserves_auditable_result(semantic_service, semantic_server, monkeypatch):
    received = []
    expected = {'question': 'Example Film director and writer?', 'status': 'answered',
                'answer': 'Example Film was directed by Jane and written by Joe.',
                'plan': {'movie_ids': ['Q1'], 'properties': ['P57', 'P58'], 'operation': 'facts'},
                'evidence': [{'statement_id': 'Q1$fact'}], 'raw_answer': 'original model output',
                'fallback_used': True, 'total': 2, 'next_offset': 1}

    class Engine:
        def __init__(self, explorer):
            assert explorer is semantic_service.explorer
        def answer(self, question, **kwargs):
            received.append((question, kwargs))
            return expected

    monkeypatch.setitem(sys.modules, 'kgqa.semantic_qa', SimpleNamespace(SemanticQA=Engine))
    status, data = request(semantic_server, payload={'question': expected['question'], 'movie_ids': ['Q1'],
                                                   'offset': 0, 'limit': 1})
    assert status == 200 and data == expected
    assert received == [(expected['question'], {'confirmed': ['Q1'], 'offset': 0, 'limit': 1})]
    assert semantic_service.semantic_service_status == 'reachable'
    assert not semantic_service.lock.locked()


def test_missing_model_never_silently_returns_rule_answer(semantic_service, semantic_server):
    class BrokenEngine:
        def answer(self, *args, **kwargs):
            raise RuntimeError('Authorization: Bearer never-print-me')
    semantic_service.semantic_engine = BrokenEngine()
    status, data = request(semantic_server, payload={'question': 'Who directed Example Film?'})
    assert status == 503 and 'P0' in data['error']
    assert 'never-print-me' not in json.dumps(data)
    assert 'answer' not in data
    assert semantic_service.semantic_service_status == 'unavailable'
    assert not semantic_service.lock.locked()


@pytest.mark.parametrize('payload', [
    {'question': ''}, {'question': None}, {'question': 'x' * 1001},
    {'question': 'q', 'offset': True}, {'question': 'q', 'offset': -1},
    {'question': 'q', 'limit': 0}, {'question': 'q', 'limit': False},
    {'question': 'q', 'movie_ids': 'Q1'}, {'question': 'q', 'movie_ids': ['Q404']},
    {'question': 'q', 'movie_ids': [None]}, {'question': 'q', 'movie_ids': ['Q1'] * 6},
])
def test_bad_request_never_loads_model(semantic_service, semantic_server, payload):
    status, data = request(semantic_server, payload=payload)
    assert status == 400 and data['error']
    assert semantic_service.semantic_engine is None


def test_semantic_missing_catalogue_and_busy_state_are_explicit(semantic_service, semantic_server):
    with semantic_service.lock:
        status, data = request(semantic_server, payload={'question': 'Example Film director?'})
    assert status == 409 and data['error']
    semantic_service.explorer = None
    status, data = request(semantic_server, payload={'question': 'Example Film director?'})
    assert status == 503 and '目录' in data['error']


def test_semantic_api_keeps_loopback_origin_boundary(semantic_server):
    status, _ = request(semantic_server, payload={'question': 'q'}, headers={'Origin': 'https://example.com'})
    assert status == 403
