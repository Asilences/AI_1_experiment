import http.client
import json
import threading
from types import SimpleNamespace

import pytest

from kgqa.acceptance import build_fixture
from kgqa.common import REFUSAL, write_json
from kgqa.personal_experiment import run
from kgqa.web import QAService, make_server


@pytest.fixture(scope='module')
def web_service(tmp_path_factory):
    root = tmp_path_factory.mktemp('web')
    data = root / 'data'
    build_fixture(data)
    run(data, root / 'runs', 'web_test')
    return QAService(data_dir=data, model_path=root / 'runs' / 'web_test' / 'model.pkl')


@pytest.fixture
def web_server(web_service):
    server = make_server(web_service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def request(server, path, data=None, headers=None, raw=None):
    connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=10)
    try:
        body = raw if raw is not None else json.dumps(data) if data is not None else None
        request_headers = {'Content-Type': 'application/json', **(headers or {})}
        connection.request('POST' if body is not None else 'GET', path, body, request_headers)
        response = connection.getresponse()
        content = response.read()
        result = json.loads(content) if response.getheader('Content-Type').startswith('application/json') else content
        return response.status, result, dict(response.getheaders())
    finally:
        connection.close()


def test_real_a0_browser_api_answers_and_refuses_unknown_movie(web_server):
    status, result, _ = request(web_server, '/api/answer', {'question': 'Who directed Juniper Trail?', 'system': 'both'})
    assert status == 200
    a0, neural = result['results']
    assert a0['ok'] and 'was directed by' in a0['answer']
    assert a0['movie']['label'] == 'Juniper Trail'
    assert a0['relation'] == 'P57' and a0['evidence']
    assert not neural['ok'] and '独立 A0' in neural['error']
    status, result, _ = request(web_server, '/api/answer', {'question': 'Who directed Zyxwuv Qpkjrt?', 'system': 'a0'})
    assert status == 200
    assert result['results'][0]['refused']
    assert result['results'][0]['answer'] == REFUSAL
    assert result['results'][0]['movie'] is None


def test_status_and_movie_search_reflect_selected_data(web_server):
    status, result, _ = request(web_server, '/api/status')
    assert status == 200 and result['movie_count'] == 10
    assert result['a0']['available'] and not result['neural']['available']
    assert len(result['examples']) == 4
    status, result, _ = request(web_server, '/api/movies?q=Juniper&limit=1')
    assert status == 200 and len(result['movies']) == 1
    assert result['movies'][0]['label'] == 'Juniper Trail'
    assert result['movies'][0]['aliases']
    status, result, _ = request(web_server, '/api/movies?q=NoSuchMovie')
    assert status == 200 and result['total'] == 0


@pytest.mark.parametrize('payload', [
    {'question': '', 'system': 'a0'},
    {'question': 'x' * 1001, 'system': 'a0'},
    {'question': None},
    {'question': 'Who directed Juniper Trail?', 'system': []},
    [],
])
def test_invalid_questions_are_rejected(web_server, payload):
    status, result, _ = request(web_server, '/api/answer', payload)
    assert status == 400 and result['error']


def test_http_boundary_rejects_foreign_origins_invalid_json_and_large_body(web_server):
    status, _, _ = request(web_server, '/api/answer', {'question': 'test'}, {'Origin': 'https://example.com'})
    assert status == 403
    status, _, _ = request(web_server, '/api/status', headers={'Host': 'attacker.example'})
    assert status == 403
    status, _, _ = request(web_server, '/api/answer', raw='{bad json')
    assert status == 400
    status, _, _ = request(web_server, '/api/answer', raw='x' * 17000)
    assert status == 413


def test_api_reports_busy_without_queueing_model_calls(web_service, web_server):
    with web_service.lock:
        status, result, _ = request(web_server, '/api/answer', {'question': 'Who directed Juniper Trail?', 'system': 'a0'})
    assert status == 409 and '处理中' in result['error']


def test_inference_error_releases_lock_and_has_actionable_message(web_service, web_server, monkeypatch):
    def broken_model():
        raise RuntimeError('internal model failure')
    monkeypatch.setattr(web_service, '_load_demo', broken_model)
    status, result, _ = request(web_server, '/api/answer', {'question': 'test', 'system': 'a0'})
    assert status == 200 and not result['results'][0]['ok']
    assert '启动终端' in result['results'][0]['error']
    assert not web_service.lock.locked()


def test_only_ui_assets_are_served(web_server):
    for path in ['/', '/app.js', '/styles.css']:
        status, content, headers = request(web_server, path)
        assert status == 200 and content
        assert headers['X-Content-Type-Options'] == 'nosniff'
    for path in ['/../requirements.txt', '/data/processed/entities.json', '/%2e%2e/requirements.txt']:
        status, _, _ = request(web_server, path)
        assert status == 404


def test_missing_data_is_reported_without_loading_model(tmp_path):
    service = QAService(tmp_path / 'absent_data', tmp_path / 'absent_model.pkl')
    state = service.status()
    assert not state['a0']['available'] and not state['neural']['available']
    assert service.demo is None
    result = service.answer('Who directed Example Film?', 'a0')
    assert not result['results'][0]['ok'] and '知识库' in result['results'][0]['error']


def test_catalog_counts_only_unique_facts_with_named_entities(tmp_path):
    write_json(tmp_path / 'entities.json', {
        'M': {'label': 'Example Film', 'facts': {'P57': ['D', 'D', 'missing'], 'P136': ['unnamed']}},
        'D': {'label': 'Jane Director'}, 'unnamed': {'label': ''},
    })
    write_json(tmp_path / 'splits.json', {'train': ['M'], 'valid': [], 'test': []})
    service = QAService(tmp_path, tmp_path / 'model.pkl')
    assert service.status()['fact_count'] == 1
    assert service.search()['movies'][0]['relations'] == ['P57']


def test_recovery_requires_neural_paths_and_is_forwarded_lazily(tmp_path, monkeypatch):
    from kgqa import demo, web
    with pytest.raises(ValueError, match='default neural dataset'):
        QAService(tmp_path, tmp_path / 'model.pkl', recover_names=True)
    monkeypatch.setattr(web, 'ROOT', tmp_path)
    received = []
    monkeypatch.setattr(demo, 'Demo', lambda **kwargs: received.append(kwargs) or object())
    service = QAService(recover_names=True)
    assert service.demo is None
    service._load_demo()
    assert received == [{'load_neural': False, 'recover_names': True}]


def test_web_names_improved_system_and_preserves_recovery_evidence(tmp_path, monkeypatch):
    from kgqa import web
    monkeypatch.setattr(web, 'ROOT', tmp_path)
    data = tmp_path / 'data/processed'
    write_json(data / 'entities.json', {'M': {'label': 'Film', 'facts': {}}})
    write_json(data / 'splits.json', {'train': ['M'], 'valid': [], 'test': []})
    write_json(tmp_path / 'output/selected_model.json', {'selected_run': 'B0'})
    write_json(tmp_path / 'runs/B0/config.json', {'use_embeddings': False})
    for name in ('config.json', 'tokenizer_config.json', 'model.safetensors'):
        write_json(tmp_path / 'runs/B0/checkpoint' / name, {})
    model = tmp_path / 'models/pipeline/model.pkl'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'fixture')
    monkeypatch.setattr(web.importlib.util, 'find_spec', lambda name: object())
    service = QAService(recover_names=True)
    assert service.status()['neural']['name'] == 'B0 神经生成 + 名称恢复'
    fake_demo = SimpleNamespace(
        neural=object(), entities={'M': {'label': 'Film'}},
        linker=SimpleNamespace(link=lambda question: ('M', 1, 'exact')),
        answer_neural=lambda question: {'answer': 'restored', 'raw_answer': 'original',
            'replacements': [{'entity_id': 'D'}], 'recovery_ms': .2, 'latency_ms': 10, 'evidence': ['fact']},
    )
    monkeypatch.setattr(service, '_load_demo', lambda: fake_demo)
    answer = service.answer('Who directed Film?', 'best')['results'][0]
    assert answer['ok'] and answer['name'] == 'B0 神经生成 + 名称恢复'
    assert answer['answer'] == 'restored' and answer['raw_answer'] == 'original'
    assert answer['replacements'] == [{'entity_id': 'D'}]
    assert answer['recovery_ms'] == .2 and answer['generation_ms'] == 10
