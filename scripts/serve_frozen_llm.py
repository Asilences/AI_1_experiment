"""Loopback-only frozen Transformers inference, exposing a minimal chat API."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=7870)
    parser.add_argument('--gpu-memory-fraction', type=float, default=.25)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 0 < args.gpu_memory_fraction <= 1:
        parser.error('Invalid port or GPU memory fraction')
    manifest = json.loads((args.model_dir / 'frozen_model_manifest.json').read_text(encoding='utf-8'))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
                                                torch_dtype=dtype, attn_implementation='sdpa').to(device)
    model.eval()
    lock = threading.Lock()
    model_id = manifest['model'].rsplit('/', 1)[-1]
    metadata = {'id': model_id, 'model': manifest['model'], 'revision': manifest['revision'],
                'device': device, 'dtype': str(dtype), 'frozen': True, 'memory_fraction': args.gpu_memory_fraction}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, value, code=200):
            body = json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ('/health', '/v1/models'):
                self.send_json({'object': 'list', 'data': [metadata]})
            else:
                self.send_json({'error': 'not_found'}, 404)

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                return self.send_json({'error': 'not_found'}, 404)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 65536:
                    raise ValueError('body')
                payload = json.loads(self.rfile.read(length))
                messages = payload['messages']
                maximum = payload.get('max_tokens', 768)
                if not isinstance(messages, list) or not 1 <= len(messages) <= 12:
                    raise ValueError('messages')
                if not isinstance(maximum, int) or not 1 <= maximum <= 1024:
                    raise ValueError('max_tokens')
                if payload.get('model') != model_id:
                    raise ValueError('model')
                if any(m.get('role') not in ('system', 'user', 'assistant') or
                       not isinstance(m.get('content'), str) for m in messages):
                    raise ValueError('messages')
            except (ValueError, KeyError, TypeError, AttributeError):
                return self.send_json({'error': 'invalid_request'}, 400)
            started = time.perf_counter()
            try:
                with lock, torch.inference_mode():
                    batch = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                          return_tensors='pt', return_dict=True)
                    if batch['input_ids'].shape[1] > 4096:
                        return self.send_json({'error': 'input_too_long'}, 400)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = model.generate(**batch, do_sample=False, max_new_tokens=maximum,
                                         pad_token_id=tokenizer.eos_token_id)
                    content = tokenizer.decode(out[0, batch['input_ids'].shape[1]:], skip_special_tokens=True)
                    usage = {'prompt_tokens': batch['input_ids'].shape[1],
                             'completion_tokens': out.shape[1] - batch['input_ids'].shape[1]}
                self.send_json({'id': 'local-frozen', 'object': 'chat.completion', 'model': model_id,
                                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content},
                                             'finish_reason': 'stop' if usage['completion_tokens'] < maximum else 'length'}],
                                'usage': usage, 'metadata': metadata,
                                'elapsed_ms': (time.perf_counter() - started) * 1000})
            except (RuntimeError, ValueError):
                self.send_json({'error': 'inference_failed'}, 503)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(json.dumps({'status': 'ready', **metadata}, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
