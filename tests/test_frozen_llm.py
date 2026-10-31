"""Synthetic HTTP protocol tests; these do not evaluate language understanding."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from kgqa.frozen_llm import FrozenLLM, LLMError, parse_json_object


@pytest.fixture
def endpoint():
    state = {'status': 200, 'content': {'choices': [{'message': {'content': '{"status":"ready"}'},
                                                   'finish_reason': 'stop'}]},
             'models': {'data': [{'id': 'test-frozen'}]}, 'requests': [], 'headers': {}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(state['status'])
            for name, value in state['headers'].items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(json.dumps(state['models']).encode())

        def do_POST(self):
            state['requests'].append({'path': self.path, 'authorization': self.headers.get('Authorization'),
                                      'body': json.loads(self.rfile.read(int(self.headers['Content-Length'])))})
            self.send_response(state['status'])
            for name, value in state['headers'].items():
                self.send_header(name, value)
            self.end_headers()
            content = state['content']
            self.wfile.write(content if isinstance(content, bytes) else json.dumps(content).encode())

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state['url'] = f'http://127.0.0.1:{server.server_port}/v1'
    yield state
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_http_contract_metadata_and_usage(endpoint):
    client = FrozenLLM(endpoint['url']+'/', 'test-frozen', api_key='secret-value')
    endpoint['content']['usage'] = {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15,
                                   'untrusted': 'hidden server value'}
    value = client.complete_json([{'role': 'user', 'content': '导演是谁？'}], max_tokens=100)
    assert value == {'status': 'ready'}
    request = endpoint['requests'][0]
    assert request['path'] == '/v1/chat/completions'
    assert request['authorization'] == 'Bearer secret-value'
    assert request['body']['model'] == 'test-frozen'
    assert request['body']['messages'] == [{'role': 'user', 'content': '导演是谁？'}]
    assert request['body']['response_format'] == {'type': 'json_object'}
    assert request['body']['temperature'] == 0 and request['body']['stream'] is False
    assert client.last_call['usage'] == {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
    assert client.metadata()['frozen'] is True and client.metadata()['revision_verified'] is False
    assert 'secret-value' not in repr(client) + json.dumps(client.metadata()) + json.dumps(client.last_call)
    assert client.health() == {'available': True, 'model': 'test-frozen', 'error_code': None}


def test_env_configuration_and_json_mode(endpoint):
    client = FrozenLLM.from_env({'KGQA_LLM_BASE_URL': endpoint['url'], 'KGQA_LLM_MODEL': 'test-frozen',
                                'KGQA_LLM_API_KEY': 'token', 'KGQA_LLM_TIMEOUT': '2.5',
                                'KGQA_LLM_JSON_MODE': '0'})
    client.complete_json([{'role': 'system', 'content': 'JSON'}])
    assert client.timeout == 2.5 and 'response_format' not in endpoint['requests'][0]['body']
    assert FrozenLLM.from_env({}).model == 'Qwen2.5-3B-Instruct'
    for env in ({'KGQA_LLM_TIMEOUT': 'not-a-number'}, {'KGQA_LLM_TIMEOUT': 'nan'},
                {'KGQA_LLM_JSON_MODE': 'true'}):
        with pytest.raises(LLMError, match='配置|必须'):
            FrozenLLM.from_env(env)


@pytest.mark.parametrize('content', ['{"x":1}', '  {"x": 1}  ', '```json\n{"x":1}\n```',
                                    '```\n{"x":1}\n```'])
def test_accepts_one_json_object(content):
    assert parse_json_object(content) == {'x': 1}


@pytest.mark.parametrize('content', ['', 'Some explanation {"x":1}', '{} {}', '[]', 'null',
                                    '{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}',
                                    '```python\n{}\n```', '```json\n{}\n```\n```json\n{}\n```',
                                    '<think>analysis</think>{}', '{"x":"<analysis>hidden</analysis>"}'])
def test_rejects_untrusted_json_forms(content):
    with pytest.raises(LLMError):
        parse_json_object(content)


def test_bounds_are_enforced_before_execution(endpoint):
    client = FrozenLLM(endpoint['url'], 'test-frozen', max_response_bytes=80)
    endpoint['content'] = b'x'*81
    with pytest.raises(LLMError) as error:
        client.complete_json([{'role': 'user', 'content': 'test'}])
    assert error.value.code == 'response_too_large'
    with pytest.raises(LLMError) as error:
        parse_json_object('{"x":"long"}', max_chars=2)
    assert error.value.code == 'content_too_large'


@pytest.mark.parametrize('response,code', [
    (b'not JSON secret-server-value', 'invalid_response'),
    ({'choices': []}, 'invalid_response'),
    ({'choices': [{'message': {'content': '{}'}, 'finish_reason': 'length'}]}, 'truncated_output'),
    ({'choices': [{'message': {'content': None}}]}, 'invalid_content'),
    ({'choices': [{'message': {'content': '<think>hidden</think>{}'}}]}, 'thinking_output'),
])
def test_response_failures_are_closed_and_sanitized(endpoint, response, code):
    endpoint['content'] = response
    client = FrozenLLM(endpoint['url'], 'test-frozen', api_key='token-to-hide')
    with pytest.raises(LLMError) as error:
        client.complete_json([{'role': 'user', 'content': 'test'}])
    assert error.value.code == code
    assert 'secret-server-value' not in str(error.value)
    assert 'token-to-hide' not in str(error.value)
    assert client.last_call is None


def test_service_error_does_not_echo_auth_or_body(endpoint):
    endpoint['status'] = 401
    endpoint['content'] = b'token-to-hide secret-server-value'
    client = FrozenLLM(endpoint['url'], 'test-frozen', api_key='token-to-hide')
    with pytest.raises(LLMError) as error:
        client.complete_json([{'role': 'user', 'content': 'test'}])
    assert error.value.code == 'authentication_failed'
    assert 'token-to-hide' not in str(error.value) and 'secret-server-value' not in str(error.value)
    assert client.health()['error_code'] == 'authentication_failed'


def test_health_requires_configured_model(endpoint):
    client = FrozenLLM(endpoint['url'], 'missing-model')
    assert client.health()['error_code'] == 'model_not_listed'
    endpoint['models'] = {'data': ['malformed']}
    assert client.health()['available'] is False


def test_redirect_is_not_followed_with_credentials(endpoint):
    endpoint['status'] = 307
    endpoint['headers']['Location'] = endpoint['url'] + '/redirected'
    client = FrozenLLM(endpoint['url'], 'test-frozen', api_key='private-token')
    with pytest.raises(LLMError) as error:
        client.complete_json([{'role': 'user', 'content': 'test'}])
    assert error.value.code == 'http_error' and len(endpoint['requests']) == 1


@pytest.mark.parametrize('kwargs', [
    {'base_url': 'file:///tmp/server'}, {'base_url': 'http://user:password@localhost/v1'},
    {'base_url': 'http://localhost/v1?token=secret'}, {'base_url': 'http://localhost/v1#fragment'},
    {'base_url': 'http://localhost:wrong/v1'}, {'base_url': 'http://localhost:0/v1'},
    {'base_url': 'http://local\nhost/v1'}, {'base_url': None},
    {'model': ''}, {'model': 'bad\nmodel'}, {'api_key': 'bad\nkey'},
    {'timeout': 0}, {'timeout': True}, {'timeout': float('inf')},
    {'max_response_bytes': False}, {'max_content_chars': 3000000}, {'json_mode': 'yes'},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(LLMError):
        FrozenLLM(**kwargs)


@pytest.mark.parametrize('messages,max_tokens', [([], 1), ([{}], 1),
    ([{'role': 'tool', 'content': 'x'}], 1), ([{'role': 'user', 'content': None}], 1),
    ([{'role': [], 'content': 'x'}], 1),
    ([{'role': 'user', 'content': 'x'*150001}], 1), ([{'role': 'user', 'content': 'x'}], True),
    ([{'role': 'user', 'content': 'x'}], 0), ([{'role': 'user', 'content': 'x'}], 4097),
])
def test_invalid_requests_fail_before_network(messages, max_tokens):
    with pytest.raises(LLMError) as error:
        FrozenLLM().complete_json(messages, max_tokens)
    assert error.value.code == 'invalid_request'
