"""Bounded JSON-only client for a separately deployed, frozen language model.

Uses the standard library so protocol tests and CLI imports stay lightweight.
The client never reads endpoint/model/authentication settings from user questions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from http.client import HTTPException
import json
import math
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class LLMError(RuntimeError):
    """An error safe to expose: no server body, URL credentials or auth token."""

    def __init__(self, code, public_message):
        self.code = code
        self.public_message = public_message
        super().__init__(public_message)


class _NoRedirect(HTTPRedirectHandler):
    # Keep configured authentication on its configured endpoint only.
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate key')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('non-finite JSON number')


def parse_json_object(content, max_chars=65536):
    """Accept one JSON object, optionally in one complete Markdown JSON fence."""
    if not isinstance(content, str) or not content.strip():
        raise LLMError('invalid_content', '模型未返回有效的 JSON 文本。')
    if len(content) > max_chars:
        raise LLMError('content_too_large', '模型输出超过允许长度。')
    text = content.strip()
    if re.search(r'</?(?:think|analysis)\b', text, re.IGNORECASE):
        raise LLMError('thinking_output', '模型返回了思考文本，未作为查询计划执行。')
    if text.startswith('```'):
        match = re.fullmatch(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n```', text,
                             re.IGNORECASE | re.DOTALL)
        if match is None or '```' in match[1]:
            raise LLMError('invalid_json', '模型输出不是单个完整 JSON 对象。')
        text = match[1]
    try:
        value = json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    except (ValueError, TypeError, RecursionError):
        raise LLMError('invalid_json', '模型输出不是单个有效 JSON 对象。') from None
    if not isinstance(value, dict):
        raise LLMError('invalid_json_type', '模型输出的顶层必须是 JSON 对象。')
    return value


@dataclass
class FrozenLLM:
    """OpenAI-compatible chat endpoint; inference only, no training operations."""

    base_url: str = 'http://127.0.0.1:7870/v1'
    model: str = 'Qwen2.5-3B-Instruct'
    api_key: str | None = field(default=None, repr=False)
    timeout: float = 120.0
    max_response_bytes: int = 1048576
    max_content_chars: int = 65536
    json_mode: bool = True
    last_call: dict | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.base_url, str) or any(ord(c) < 32 for c in self.base_url):
            raise LLMError('invalid_configuration', '模型服务地址配置无效。')
        try:
            parsed = urlsplit(self.base_url)
            port = parsed.port
        except (TypeError, ValueError):
            raise LLMError('invalid_configuration', '模型服务地址配置无效。') from None
        if (parsed.scheme not in {'http', 'https'} or not parsed.hostname or
                parsed.username is not None or parsed.password is not None or
                parsed.query or parsed.fragment or port == 0):
            raise LLMError('invalid_configuration', '模型服务地址必须是无凭据、无查询参数的 HTTP(S) 地址。')
        self.base_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip('/'), '', ''))
        if not isinstance(self.model, str) or not 1 <= len(self.model) <= 200 or any(
                ord(c) < 32 for c in self.model):
            raise LLMError('invalid_configuration', '模型标识配置无效。')
        if (isinstance(self.timeout, bool) or not isinstance(self.timeout, (float, int)) or
                not math.isfinite(self.timeout) or not 0 < self.timeout <= 300):
            raise LLMError('invalid_configuration', '模型超时配置必须在 0—300 秒之间。')
        for value in (self.max_response_bytes, self.max_content_chars):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2097152:
                raise LLMError('invalid_configuration', '模型响应长度限制配置无效。')
        if not isinstance(self.json_mode, bool):
            raise LLMError('invalid_configuration', 'JSON 模式配置无效。')
        if self.api_key is not None and (not isinstance(self.api_key, str) or any(
                ord(c) < 32 for c in self.api_key)):
            raise LLMError('invalid_configuration', '模型认证配置无效。')

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ
        try:
            timeout = float(env.get('KGQA_LLM_TIMEOUT', '120'))
        except (TypeError, ValueError):
            raise LLMError('invalid_configuration', '模型超时配置无效。') from None
        mode = env.get('KGQA_LLM_JSON_MODE', '1')
        if mode not in {'0', '1'}:
            raise LLMError('invalid_configuration', 'KGQA_LLM_JSON_MODE 必须为 0 或 1。')
        return cls(base_url=env.get('KGQA_LLM_BASE_URL', 'http://127.0.0.1:7870/v1'),
                   model=env.get('KGQA_LLM_MODEL', 'Qwen2.5-3B-Instruct'),
                   api_key=env.get('KGQA_LLM_API_KEY') or None, timeout=timeout,
                   json_mode=mode == '1')

    def metadata(self):
        """Configuration only: this does not claim the service is reachable."""
        return {'backend': 'openai_compatible_http', 'base_url': self.base_url,
                'model': self.model, 'frozen': True, 'revision': None,
                'revision_verified': False, 'timeout_seconds': self.timeout,
                'temperature': 0, 'json_mode': self.json_mode,
                'max_response_bytes': self.max_response_bytes,
                'max_content_chars': self.max_content_chars}

    def _request(self, suffix, payload=None, timeout=None):
        headers = {'Accept': 'application/json'}
        body = None
        if payload is not None:
            headers['Content-Type'] = 'application/json'
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        request = Request(self.base_url + suffix, data=body, headers=headers,
                          method='GET' if payload is None else 'POST')
        try:
            with build_opener(_NoRedirect()).open(request, timeout=timeout or self.timeout) as response:
                raw = response.read(self.max_response_bytes + 1)
        except HTTPError as error:
            code = 'authentication_failed' if error.code in {401, 403} else 'http_error'
            error.close()
            raise LLMError(code, '模型服务请求失败，请检查服务配置和可用状态。') from None
        except (URLError, TimeoutError, OSError, ValueError, HTTPException):
            raise LLMError('service_unavailable', '模型服务不可用或请求超时。') from None
        if len(raw) > self.max_response_bytes:
            raise LLMError('response_too_large', '模型服务响应超过允许长度。')
        try:
            value = json.loads(raw.decode('utf-8'), object_pairs_hook=_object, parse_constant=_constant)
        except (UnicodeError, ValueError, TypeError, RecursionError):
            raise LLMError('invalid_response', '模型服务未返回有效的 JSON 响应。') from None
        if not isinstance(value, dict):
            raise LLMError('invalid_response', '模型服务响应结构无效。')
        return value

    def health(self):
        """Probe /models; returned diagnostic fields do not contain secrets."""
        value = {'available': False, 'model': self.model, 'error_code': None}
        try:
            response = self._request('/models', timeout=min(self.timeout, 5))
            models = response.get('data')
            if not isinstance(models, list) or not any(
                    isinstance(item, dict) and item.get('id') == self.model for item in models):
                value['error_code'] = 'model_not_listed'
            else:
                value['available'] = True
        except LLMError as error:
            value['error_code'] = error.code
        return value

    def complete_json(self, messages, max_tokens=768):
        if (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or
                not 1 <= max_tokens <= 4096):
            raise LLMError('invalid_request', '模型输出 token 上限必须为 1—4096。')
        if not isinstance(messages, list) or not 1 <= len(messages) <= 50:
            raise LLMError('invalid_request', '模型消息格式无效。')
        copied = []
        for message in messages:
            if (not isinstance(message, dict) or not isinstance(message.get('role'), str) or
                    message['role'] not in {'system', 'user', 'assistant'} or
                    not isinstance(message.get('content'), str)):
                raise LLMError('invalid_request', '模型消息格式无效。')
            copied.append({'role': message['role'], 'content': message['content']})
        input_chars = sum(len(message['content']) for message in copied)
        if input_chars > 150000:
            raise LLMError('invalid_request', '模型输入超过允许长度。')
        payload = {'model': self.model, 'messages': copied, 'max_tokens': max_tokens,
                   'temperature': 0, 'stream': False}
        if self.json_mode:
            payload['response_format'] = {'type': 'json_object'}
        started = time.perf_counter()
        self.last_call = None
        response = self._request('/chat/completions', payload)
        try:
            choice = response['choices'][0]
            content = choice['message']['content']
        except (KeyError, IndexError, TypeError):
            raise LLMError('invalid_response', '模型服务响应中缺少完整消息。') from None
        if choice.get('finish_reason') == 'length':
            raise LLMError('truncated_output', '模型输出达到长度上限，未执行不完整结果。')
        value = parse_json_object(content, self.max_content_chars)
        usage = response.get('usage')
        safe_usage = {k: usage[k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')
                      if isinstance(usage, dict) and isinstance(usage.get(k), int) and
                      not isinstance(usage[k], bool) and usage[k] >= 0}
        self.last_call = {'elapsed_ms': round((time.perf_counter() - started) * 1000, 2),
                          'input_chars': input_chars, 'max_tokens': max_tokens, 'usage': safe_usage}
        return value
