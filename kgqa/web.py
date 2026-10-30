"""Local browser UI for the existing QA models; no extra web dependencies."""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
import webbrowser

from .common import REFUSAL, RELATIONS, ROOT, read_json, sha256
from .chinese import ChineseAdapter

ASSETS = Path(__file__).with_name('ui')
SYSTEMS = {'a0', 'best', 'both'}
MAX_BODY = 16384


def startup_paths(data_dir=None, model_path=None, legacy=False):
    """An explicitly activated dataset is paired with its model by hashes."""
    if legacy and (data_dir is not None or model_path is not None):
        raise ValueError('--legacy cannot be combined with custom data/model paths')
    profile_path = ROOT / '.work' / 'ui-dataset.json'
    if legacy or data_dir is not None or model_path is not None or not profile_path.exists():
        return data_dir, model_path
    profile = read_json(profile_path)
    resolved = []
    for key in ('data_dir', 'model_path'):
        relative = Path(profile[key])
        destination = (ROOT / relative).resolve()
        if relative.is_absolute() or not destination.is_relative_to(ROOT.resolve()):
            raise ValueError('Active dataset paths must stay inside the project')
        resolved.append(destination)
    data, model = resolved
    for name, expected in profile['input_hashes'].items():
        if name not in {'entities.json', 'splits.json', 'qa.jsonl'} or sha256(data / name) != expected:
            raise ValueError('Active dataset has changed; rebuild or reactivate its matching model')
    if set(profile['input_hashes']) != {'entities.json', 'splits.json', 'qa.jsonl'} or sha256(model) != profile['model_sha256']:
        raise ValueError('Active model or dataset hashes do not match')
    return data, model


class APIError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class QAService:
    def __init__(self, data_dir=None, model_path=None, catalog_dir=None, recover_names=False):
        self.custom_paths = data_dir is not None or model_path is not None
        if recover_names and self.custom_paths:
            raise ValueError('Name recovery requires the default neural dataset, not custom A0 paths')
        self.recover_names = recover_names
        self.data_dir = Path(data_dir) if data_dir else ROOT / 'data' / 'processed'
        self.model_path = Path(model_path) if model_path else ROOT / 'models' / 'pipeline' / 'model.pkl'
        self.demo = None
        self.freeform_parser = None
        self.entities = {}
        self.lock = threading.Lock()
        self.movies, self.data_error = [], ''
        self.dataset_name = '原始实验电影库'
        self.explorer = None
        self.catalog_error = ''
        try:
            entities = read_json(self.data_dir / 'entities.json')
            self.entities = entities
            splits = read_json(self.data_dir / 'splits.json')
            display = ChineseAdapter(entities, sum(splits.values(), []))
            manifest_path = self.data_dir / 'manifest.json'
            if manifest_path.exists():
                self.dataset_name = read_json(manifest_path).get('name', self.dataset_name)
            for qid in sorted(set(sum(splits.values(), []))):
                entity = entities[qid]
                facts = {r: {tail for tail in entity.get('facts', {}).get(r, [])
                             if entities.get(tail, {}).get('label')} for r in RELATIONS}
                self.movies.append({'id': qid, 'label': entity['label'],
                                    'display_label': entity.get('display_label', entity['label']),
                                    'question_title': entity.get('question_title', entity['label']),
                                    'label_zh': display.format_entity(qid) if any('\u3400' <= c <= '\u9fff'
                                                                                for c in display.format_entity(qid)) else '',
                                    'aliases': entity.get('aliases', []),
                                    'relations': [r for r in RELATIONS if facts[r]],
                                    'fact_count': sum(len(v) for v in facts.values())})
            self.movies.sort(key=lambda item: (item['label'].casefold(), item['id']))
            if not self.movies:
                self.data_error = '知识库中没有电影，请检查所选数据。'
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            self.data_error = '未能读取本地知识库，请先准备数据，或指定有效的数据目录。'
        from .v2_data import DEST
        directory = Path(catalog_dir) if catalog_dir is not None else DEST
        if (directory / 'manifest.json').is_file():
            try:
                summary = read_json(directory / 'manifest.json')
                if summary.get('movie_count', 0) > len(self.movies):
                    raise ValueError('扩展目录与当前实验目录规模不匹配。')
                from .explorer import Explorer
                explorer = Explorer(directory)
                if not set(explorer.movies).issubset({m['id'] for m in self.movies}):
                    raise ValueError('扩展目录电影与当前实验目录不匹配。')
                self.explorer = explorer
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.catalog_error = str(error)
        elif catalog_dir is not None:
            self.catalog_error = '未找到已冻结的扩展目录。'

    def _neural_state(self):
        if self.custom_paths:
            return {'available': False, 'name': '神经生成', 'detail': '当前使用独立 A0 数据与模型，仅开放管道问答。'}
        try:
            run_id = read_json(ROOT / 'output' / 'selected_model.json')['selected_run']
            if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', run_id):
                raise ValueError('Invalid run ID')
            run = ROOT / 'runs' / run_id
            config = read_json(run / 'config.json')
            checkpoint = run / 'checkpoint'
            files = ['config.json', 'tokenizer_config.json']
            if config['use_embeddings']:
                files.append('graph_prefix.pt')
                if not all((ROOT / 'models' / 'transe' / name).is_file()
                           for name in ('embeddings.npz', 'mapping.json')):
                    raise FileNotFoundError('Missing graph embeddings')
            if (not all((checkpoint / name).is_file() for name in files)
                    or not any((checkpoint / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin'))):
                raise FileNotFoundError('Missing checkpoint')
            dependencies = all(importlib.util.find_spec(name) is not None for name in ('torch', 'transformers'))
            return {'available': dependencies, 'name': f'{run_id} 神经生成' + (' + 名称恢复' if self.recover_names else ''), 'run_id': run_id,
                    'detail': '首次使用时加载本地模型。' if dependencies else '当前 Python 环境未安装神经模型依赖。'}
        except (OSError, ValueError, KeyError, TypeError):
            return {'available': False, 'name': '神经生成', 'detail': '未找到完整的本地神经模型，请先完成训练或恢复检查点。'}

    def status(self):
        available = not self.data_error and self.model_path.is_file()
        neural = self._neural_state()
        if not available:
            neural.update(available=False, detail=self.data_error or '请先准备配套的 A0 模型和知识库。')
        examples = []
        examples_zh = []
        priorities = {'Shutter Island': 0, 'Boss Level': 1, 'Dogma': 2, '10 Things I Hate About You': 3}
        preferred = sorted(self.movies, key=lambda m: (priorities.get(m['label'], 4), m['label']))
        templates = {'P57': 'Who directed {title}?', 'P136': 'What genre is {title}?',
                     'P495': 'Which country did {title} originate in?',
                     'P364': 'What is the original language of {title}?'}
        used_movies = set()
        for relation, template in templates.items():
            candidates = [m for m in preferred if relation in m['relations']]
            movie = next((m for m in candidates if m['id'] not in used_movies), next(iter(candidates), None))
            if movie:
                used_movies.add(movie['id'])
                examples.append({'relation': relation, 'question': template.format(title=movie['question_title'])})
        chinese_templates = {'P57': '《{title}》的导演是谁？', 'P136': '《{title}》是什么类型的电影？',
                             'P495': '《{title}》是哪个国家制作的？', 'P364': '《{title}》的原始语言是什么？'}
        zh_priorities = {'Inception': 0, 'Titanic': 1}
        zh_movies = sorted((m for m in self.movies if m['label_zh']),
                           key=lambda m: (zh_priorities.get(m['label'], 2), m['label']))
        for relation, template in chinese_templates.items():
            movie = next((m for m in zh_movies if relation in m['relations']), None)
            if movie:
                examples_zh.append({'relation': relation, 'question': template.format(title=movie['label_zh'])})
        return {'movie_count': len(self.movies), 'fact_count': sum(m['fact_count'] for m in self.movies),
                'dataset_name': self.dataset_name,
                'relation_count': len(RELATIONS), 'custom_data': self.custom_paths, 'examples': examples,
                'examples_zh': examples_zh, 'question_languages': ['zh', 'en'],
                'chinese_coverage': {'movies': len(self.movies), 'with_chinese_title': len(zh_movies)},
                'a0': {'available': available, 'name': 'A0 管道问答',
                       'detail': self.data_error or ('本地模型就绪。' if available else '未找到 A0 模型，请先完成训练或指定模型路径。')},
                'neural': neural,
                'explorer': {'available': self.explorer is not None, 'method': 'deterministic_tools_v2',
                             'movie_count': len(self.explorer.movies) if self.explorer else 0,
                             'statement_count': len(self.explorer.rows) if self.explorer else 0,
                             'unresolved_entity_label_count': self.explorer.manifest.get('unresolved_entity_label_count', 0) if self.explorer else 0,
                             'detail': self.catalog_error or '独立扩展事实与只读工具，含来源、限定条件和候选确认。'},
                'freeform': {'available': True, 'method': 'bounded_rules_v1',
                             'detail': '四类单跳自然问法；有限中文转英文标准问题，答案保持英文。'}}

    def search(self, query='', offset=0, limit=30):
        query = query.casefold().strip()
        matches = [m for m in self.movies if not query or any(
            query in str(name).casefold() for name in [m['label'], m['label_zh'], m['id'], *m['aliases']])]
        return {'movies': matches[offset:offset + limit], 'total': len(matches), 'offset': offset}

    def explore(self, question, movie_ids=None, offset=0, limit=30):
        if self.explorer is None:
            raise APIError(self.catalog_error or '扩展目录尚未准备，请先完成V2数据构建。', 503)
        if not self.lock.acquire(blocking=False):
            raise APIError('上一条问题仍在处理中，请稍后再试。', 409)
        try:
            try:
                return self.explorer.answer(question, movie_ids, offset, limit)
            except ValueError as error:
                raise APIError(str(error)) from error
        finally:
            self.lock.release()

    def _load_demo(self):
        if self.demo is None:
            from .demo import Demo
            self.demo = (Demo(load_neural=False, data_dir=self.data_dir, model_path=self.model_path)
                         if self.custom_paths else Demo(load_neural=False, recover_names=self.recover_names))
        return self.demo

    def answer(self, question, system, mode='standard'):
        if not isinstance(question, str) or not question.strip() or len(question) > 1000:
            raise APIError('请输入 1—1000 个字符的中文或英文电影问题。')
        if not isinstance(system, str) or system not in SYSTEMS:
            raise APIError('请选择有效的问答系统。')
        if not isinstance(mode, str) or mode not in {'standard', 'freeform'}:
            raise APIError('请选择有效的问题处理模式。')
        if not self.lock.acquire(blocking=False):
            raise APIError('上一条问题仍在处理中，请稍后再试。', 409)
        started = time.perf_counter()
        try:
            state = self.status()
            interpretation = None
            if mode == 'freeform' and not self.data_error:
                if self.freeform_parser is None:
                    from .freeform import FreeformParser
                    self.freeform_parser = FreeformParser([movie['id'] for movie in self.movies], self.entities)
                interpretation = self.freeform_parser.parse(question)
            results = []
            for target in (('a0', 'best') if system == 'both' else (system,)):
                entry = state['a0' if target == 'a0' else 'neural']
                result = {'system': target, 'name': entry['name'], 'ok': False}
                tick = time.perf_counter()
                if not entry['available']:
                    result['error'] = entry['detail']
                else:
                    try:
                        first_load = self.demo is None
                        demo = self._load_demo()
                        cold = first_load or target == 'best' and demo.neural is None
                        model_question = interpretation['canonical_question'] if interpretation else question
                        if interpretation and interpretation['status'] == 'ready':
                            if demo.linker.link(model_question)[0] != interpretation['movie_id']:
                                interpretation.update(status='needs_clarification', message='标准片名仍有歧义，请选择明确版本。')
                        if interpretation and interpretation['status'] != 'ready':
                            raw = {'answer': REFUSAL, 'relation': interpretation['relation_id'], 'evidence': []}
                        else:
                            raw = demo.answer_a0(model_question) if target == 'a0' else demo.answer_neural(model_question)
                        if interpretation:
                            movie_id, score, link_status = (interpretation['movie_id'], interpretation['link_score'], interpretation['link_status'])
                        elif raw.get('language') == 'zh':
                            movie_id = raw.get('movie_id')
                            score, link_status = (1.0 if movie_id else 0.0), raw.get('link_status', '')
                        else:
                            movie_id, score, link_status = demo.linker.link(question)
                        movie = ({'id': movie_id, 'label': (demo.chinese.format_entity(movie_id) if raw.get('language') == 'zh'
                                                          else demo.entities[movie_id].get('display_label', demo.entities[movie_id]['label']))}
                                 if movie_id else None)
                        result.update(ok=True, answer=raw['answer'], refused=raw.get('refused', raw['answer'] == REFUSAL),
                                      movie=movie, link_status=link_status, link_score=score,
                                      relation=raw.get('relation'), confidence=raw.get('confidence'),
                                      evidence=raw.get('evidence', []), cold_start=cold,
                                      generation_ms=raw.get('latency_ms'))
                        for key in ('language', 'model_question', 'original_answer', 'language_status', 'language_note', 'answer_ids'):
                            if key in raw:
                                result[key] = raw[key]
                        if 'raw_answer' in raw:
                            result.update(raw_answer=raw['raw_answer'], replacements=raw['replacements'],
                                          recovery_ms=raw['recovery_ms'])
                    except Exception:
                        logging.exception('UI inference failed for %s', target)
                        result['error'] = '本地模型运行失败，请检查启动终端的错误信息及数据、模型是否配套。'
                result['elapsed_ms'] = round((time.perf_counter() - tick) * 1000, 2)
                results.append(result)
            return {'question': question.strip(), 'mode': mode, 'interpretation': interpretation, 'results': results,
                    'elapsed_ms': round((time.perf_counter() - started) * 1000, 2)}
        finally:
            self.lock.release()


def make_server(service, port=7860):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, message, *args):
            logging.info(message, *args)

        def send_bytes(self, body, mime, status=200):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_json(self, value, status=200):
            self.send_bytes(json.dumps(value, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8', status)

        def valid_request(self):
            host = self.headers.get('Host', '')
            try:
                parsed = urlsplit('http://' + host)
                valid = parsed.hostname in {'127.0.0.1', 'localhost'} and parsed.port == self.server.server_port
            except ValueError:
                valid = False
            origin = self.headers.get('Origin')
            if not valid or origin and origin != 'http://' + host:
                self.send_json({'error': '此服务仅接受本机页面的请求。'}, 403)
                return False
            return True

        def do_GET(self):
            if not self.valid_request():
                return
            url = urlsplit(self.path)
            if url.path == '/api/status':
                self.send_json(service.status())
            elif url.path == '/api/movies':
                query = parse_qs(url.query)
                try:
                    offset = max(0, int(query.get('offset', ['0'])[0]))
                    limit = max(1, min(100, int(query.get('limit', ['30'])[0])))
                except ValueError:
                    self.send_json({'error': '分页参数无效。'}, 400)
                    return
                self.send_json(service.search(query.get('q', [''])[0][:200], offset, limit))
            elif url.path in {'/', '/index.html', '/app.js', '/styles.css'}:
                name = 'index.html' if url.path == '/' else url.path[1:]
                mime = {'index.html': 'text/html', 'app.js': 'text/javascript', 'styles.css': 'text/css'}[name]
                self.send_bytes((ASSETS / name).read_bytes(), mime + '; charset=utf-8')
            else:
                self.send_json({'error': '页面不存在。'}, 404)

        def do_POST(self):
            if not self.valid_request():
                return
            path = urlsplit(self.path).path
            if path not in {'/api/answer', '/api/explore'}:
                self.send_json({'error': '接口不存在。'}, 404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_BODY:
                    raise APIError('请求内容为空或过大。', 413)
                if self.headers.get_content_type() != 'application/json':
                    raise APIError('请求必须使用 JSON 格式。', 415)
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise APIError('请求格式无效。')
                if path == '/api/explore':
                    self.send_json(service.explore(data.get('question'), data.get('movie_ids'), data.get('offset', 0), data.get('limit', 30)))
                else:
                    self.send_json(service.answer(data.get('question'), data.get('system', 'both'), data.get('mode', 'standard')))
            except (ValueError, UnicodeError):
                self.send_json({'error': '请求格式无效。'}, 400)
            except APIError as error:
                self.send_json({'error': str(error)}, error.status)

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--data-dir', type=Path, help='Custom processed data; enables A0 only.')
    parser.add_argument('--model-path', type=Path, help='Custom A0 model.pkl; enables A0 only.')
    parser.add_argument('--catalog-dir', type=Path, help='Separate frozen V2 catalogue for extended tool queries.')
    parser.add_argument('--open', action='store_true', help='Open the local UI in your browser.')
    parser.add_argument('--legacy', action='store_true', help='Use default experiment data and neural models instead of the activated A0 profile.')
    parser.add_argument('--recover-names', action='store_true', help='Enable auditable name recovery for neural answers.')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        data_dir, model_path = startup_paths(args.data_dir, args.model_path, args.legacy)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'数据版本配置无效：{error}。可用 --legacy 启动原始实验库，或重新激活配套数据与模型。\n')
    if args.recover_names and (data_dir is not None or model_path is not None):
        parser.error('--recover-names requires the default neural dataset; use --legacy when an A0-only profile is active')
    try:
        server = make_server(QAService(data_dir, model_path, catalog_dir=args.catalog_dir,
                                       recover_names=args.recover_names), args.port)
    except OSError as error:
        parser.exit(1, f'无法启动本地页面：{error}。可使用 --port 7861 更换端口。\n')
    url = f'http://127.0.0.1:{args.port}'
    print(f'电影知识图谱问答 UI：{url}\n保留此终端；按 Ctrl+C 停止服务。', flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
