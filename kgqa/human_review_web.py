"""在本机逐题填写真实人工评价，原始空白评分表与系统密钥保持不变。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import threading
import unicodedata
from difflib import SequenceMatcher
from urllib.parse import parse_qs, urlsplit

from .common import ROOT
from .human_eval import SCORE_RANGES, VISIBLE_FIELDS, _answer_id, _read_score_csv


def _normal_words(value):
    value = unicodedata.normalize('NFKC', str(value)).casefold()
    value = re.sub(r'[^\w\s]', ' ', value, flags=re.UNICODE)
    return set(value.split())


def suggestion_for(row):
    """Create a transparent machine suggestion for human confirmation.

    This is a pre-annotation aid, not a human score. Exact normalized matches
    receive the strongest suggestion; near matches remain visible for review.
    The final CSV is written only after the evaluator clicks the confirmation
    button or edits the four rubric fields.
    """
    answer = str(row.get('answer', ''))
    references = [part.strip() for part in str(row.get('reference', '')).split('||') if part.strip()]
    if not references:
        references = ['']
    answer_words = _normal_words(answer)
    candidates = []
    for reference in references:
        reference_words = _normal_words(reference)
        ratio = SequenceMatcher(None, ' '.join(sorted(answer_words)),
                                ' '.join(sorted(reference_words))).ratio()
        missing = reference_words - answer_words
        extra = answer_words - reference_words
        candidates.append((ratio, len(missing), len(extra), reference))
    ratio, missing_count, extra_count, best_reference = max(candidates)
    if ratio >= 0.985 and missing_count == 0 and extra_count == 0:
        scores = {'accuracy_0_2': 2, 'completeness_0_2': 2,
                  'fluency_1_5': 5, 'unsupported_claim_0_1': 0}
        label = '√ 建议确认'
        reason = '答案与参考答案规范化后完全一致。'
    elif ratio >= 0.78:
        scores = {'accuracy_0_2': 2 if missing_count <= 1 else 1,
                  'completeness_0_2': 2 if missing_count == 0 else 1,
                  'fluency_1_5': 4,
                  'unsupported_claim_0_1': 1 if extra_count > 2 else 0}
        label = '× 建议核查'
        reason = '答案与参考答案相近，但存在措辞、遗漏或附加内容，需要人工确认。'
    else:
        scores = {'accuracy_0_2': 0, 'completeness_0_2': 0,
                  'fluency_1_5': 2, 'unsupported_claim_0_1': 1}
        label = '× 建议修改'
        reason = '答案与参考答案差异较大，建议逐项修改评分。'
    return {'scores': scores, 'label': label, 'reason': reason,
            'method': 'normalized_reference_similarity_v1',
            'similarity': round(ratio, 4), 'reference_used': best_reference}


class ReviewStore:
    def __init__(self, source, output):
        self.rows = _read_score_csv(source, VISIBLE_FIELDS)
        self.index = {_answer_id(r, 'review'): r for r in self.rows}
        if len(self.index) != len(self.rows):
            raise ValueError('Duplicate answer IDs')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def path(self, reviewer):
        if not isinstance(reviewer, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', reviewer):
            raise ValueError('评价者代号限1—40个英文字母、数字、下划线或短横线')
        return self.output / f'reviewer-{reviewer}.json'

    def load(self, reviewer):
        path = self.path(reviewer)
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

    def suggestions(self):
        return {f'{row["question_id"]}|{row["answer_position"]}': suggestion_for(row)
                for row in self.rows}

    def save(self, reviewer, values):
        path = self.path(reviewer)
        qid = values.get('question_id')
        position = values.get('answer_position')
        if not isinstance(qid, str) or type(position) is not int or (qid, position) not in self.index:
            raise ValueError('评分对象不在本次题集中')
        scores = {}
        for field, (low, high) in SCORE_RANGES.items():
            score = values.get(field)
            if type(score) is not int or not low <= score <= high:
                raise ValueError(f'{field}需要实际填写，范围{low}—{high}')
            scores[field] = score
        review_mode = values.get('review_mode', 'manual')
        if review_mode not in {'manual', 'confirm_suggestion'}:
            raise ValueError('review_mode无效')
        if review_mode == 'confirm_suggestion':
            suggested = suggestion_for(self.index[(qid, position)])['scores']
            if scores != suggested:
                raise ValueError('确认建议的分数与服务器建议不一致，请改用手动修改')
        notes = values.get('notes', '')
        if not isinstance(notes, str) or len(notes) > 4000:
            raise ValueError('备注过长或格式错误')
        if review_mode == 'confirm_suggestion' and not notes:
            notes = 'AI建议已由评价者逐题确认；不是未经复核的自动评分。'
        item = {**scores, 'notes': notes, 'review_mode': review_mode,
                'updated_at': datetime.now(timezone.utc).isoformat()}
        with self.lock:
            state = self.load(reviewer)
            state[f'{qid}|{position}'] = item
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(path)
            with (self.output / f'reviewer-{reviewer}.events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'question_id': qid, 'answer_position': position, **item}, ensure_ascii=False) + '\n')
            # 导出文件仍保留未填空格；只有完整文件才可通过正式汇总校验。
            csv_path = self.output / f'reviewer-{reviewer}.csv'
            csv_path.write_text(self.csv_text(state), encoding='utf-8-sig')
        return state

    def csv_text(self, state):
        output = io.StringIO(newline='')
        writer = csv.DictWriter(output, fieldnames=VISIBLE_FIELDS)
        writer.writeheader()
        for row in self.rows:
            qid, position = _answer_id(row, 'review')
            saved = state.get(f'{qid}|{position}', {})
            writer.writerow({**row, **{k: saved.get(k, '') for k in [*SCORE_RANGES, 'notes']}})
        return output.getvalue()


def make_server(store, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, value, status=200, content_type='application/json; charset=utf-8'):
            data = value.encode('utf-8') if isinstance(value, str) else json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def valid_origin(self):
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            return (self.headers.get('Host') in allowed and
                    (self.headers.get('Origin') is None or
                     self.headers.get('Origin') in {'http://' + h for h in allowed}))

        def do_GET(self):
            if not self.valid_origin():
                return self.send({'error': '仅接受本机同源请求'}, 403)
            parsed = urlsplit(self.path)
            if parsed.path == '/':
                return self.send((ROOT / 'kgqa/ui/human_review.html').read_text(encoding='utf-8'), content_type='text/html; charset=utf-8')
            try:
                reviewer = parse_qs(parsed.query).get('reviewer', [''])[0]
                saved = store.load(reviewer)
                if parsed.path == '/api/review':
                    public = [{k: row[k] for k in ['question_id', 'question', 'reference', 'answer_position', 'answer']} for row in store.rows]
                    return self.send({'rows': public, 'scores': saved, 'suggestions': store.suggestions(),
                                      'output_dir': str(store.output.resolve())})
                if parsed.path == '/api/export':
                    return self.send('\ufeff' + store.csv_text(saved), content_type='text/csv; charset=utf-8')
                self.send({'error': '接口不存在'}, 404)
            except ValueError as error:
                self.send({'error': str(error)}, 400)

        def do_POST(self):
            if not self.valid_origin():
                return self.send({'error': '仅接受本机同源请求'}, 403)
            if self.path != '/api/score':
                return self.send({'error': '接口不存在'}, 404)
            try:
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('需要JSON请求')
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 32768:
                    raise ValueError('请求长度无效')
                values = json.loads(self.rfile.read(size))
                if not isinstance(values, dict):
                    raise ValueError('请求格式无效')
                state = store.save(values.get('reviewer'), values)
                self.send({'saved': True, 'completed_answers': len(state), 'total_answers': len(store.rows)})
            except (ValueError, TypeError, UnicodeError) as error:
                self.send({'error': str(error)}, 400)

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'reports/人工评价/5000部电影_20261030/human_evaluation.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output/human_review')
    parser.add_argument('--port', type=int, default=7865)
    args = parser.parse_args(argv)
    server = make_server(ReviewStore(args.source, args.output_dir), args.port)
    print(f'人工评价入口：http://127.0.0.1:{server.server_port}/；原始评分保存到{args.output_dir}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
