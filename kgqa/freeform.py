"""An explicit natural-question adapter, separate from the experiment baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import unicodedata

from .common import REFUSAL, ROOT, write_json
from .demo import Demo
from .linker import EntityLinker, normalize

DEFAULT_DATA = ROOT / 'data/processed/movies_5000_20261029'
DEFAULT_MODEL = ROOT / 'models/pipeline_5000_20261029/model.pkl'
CANONICAL = {
    'P57': 'Who directed {title}?', 'P136': 'What genre is {title}?',
    'P495': 'Which country did {title} originate in?',
    'P364': 'What is the original language of {title}?',
}
PATTERNS = {
    'P57': r'\b(?:director|directors|directed|directing|helmed|filmmaker|filmmakers)\b|behind the camera|导演|導演|执导|執導|谁拍|誰拍',
    'P136': r'\b(?:genre|genres)\b|\b(?:kind|type|category)\b.*\b(?:movie|film)\b|类型|類型|什么片|什麼片|什么类别|什麼類別',
    'P495': r'\b(?:country|countries|nation|nations|origin|originate|originated)\b|\bwhere\b.*\bfrom\b|哪国|哪國|哪个国家|哪個國家|制片国家|製片國家|制作国家|製作國家|哪里出品|哪裡出品',
    'P364': r'\b(?:language|languages|spoken|speak|speaks)\b|原声|原聲|原始语言|原始語言|什么语|什麼語|哪种语言|哪種語言|说什么话|說什麼話',
}
UNSUPPORTED = (
    r'\b(?:budget|box office|runtime|screenplay|writer|writers|write|wrote|written|cast|actor|actors|producer|producers|'
    r'composer|composed|rating|ratings|award|awards|plot|story|recommend|recommendation|born|filmed|filming|shot|location|'
    r'released|when|why|earn|earned|cost|sequel|starring|played)\b|\b(?:how many|other films|other movies|better than)\b'
    r'|\brelease\b.*\b(?:date|year)\b|演员|演員|主演|编剧|編劇|预算|預算|票房|评分|評分|时长|時長|上映|'
    r'制片人|製片人|配乐|配樂|剧情|劇情|推荐|推薦|好看|拍摄地|拍攝地|取景|在哪拍|在哪裡拍'
)


def fold(text):
    return unicodedata.normalize('NFKC', text).casefold()


class FreeformParser:
    """Conservative, bounded rules; this is not a learned language model."""

    def __init__(self, movie_ids, entities, linker=None):
        self.movie_ids, self.entities = set(movie_ids), entities
        self.linker = linker or EntityLinker(movie_ids, entities)
        self.cjk_names = {}
        for qid in self.movie_ids:
            entity = entities[qid]
            for name in [entity['label'], *entity.get('aliases', [])]:
                name = fold(name)
                if len(name) >= 2 and re.search(r'[\u3400-\u9fff]', name):
                    self.cjk_names.setdefault(name, set()).add(qid)

    def _resolve(self, question):
        text = fold(question)
        qids = set(re.findall(r'(?<![a-z0-9])q\d+(?![a-z0-9])', text, flags=re.I))
        qids = {qid.upper() for qid in qids}
        if qids - self.movie_ids:
            return None, 0., 'not_found', []
        mentions = []
        for name, ids in self.cjk_names.items():
            for match in re.finditer(re.escape(name), text):
                mentions.append((match.start(), match.end(), ids))
        mentions = EntityLinker._non_overlapping(mentions)
        # An unknown numbered sequel must not become the shorter known film.
        if any(re.match(r'\s*\d', text[right:]) for _, right, _ in mentions):
            return None, 0., 'not_found', []
        ids = set().union(*(item[2] for item in mentions)) if mentions else set()
        movie_id, score, status = self.linker.link(question)
        if movie_id:
            ids.add(movie_id)
        if qids:
            chosen = next(iter(qids))
            all_mentions = mentions + self.linker._exact_mentions(normalize(question))
            if (len(qids) > 1 or any(chosen not in item[2] for item in all_mentions)
                    or movie_id and movie_id != chosen):
                return None, 0., 'ambiguous', sorted(ids | qids)
            return chosen, 1., 'qid', []
        if len(ids) > 1:
            return None, 0., 'ambiguous', sorted(ids)
        if ids:
            chosen = next(iter(ids))
            return chosen, 1. if mentions else score, 'alias' if mentions else status, []
        return None, score, status, []

    def parse(self, question):
        if not isinstance(question, str) or not question.strip() or len(question) > 1000:
            raise ValueError('请输入 1—1000 个字符的电影问题。')
        movie, score, method, candidates = self._resolve(question)
        result = {'question': question.strip(), 'method': 'bounded_rules_v1', 'status': 'needs_clarification',
                  'movie_id': movie, 'relation_id': None, 'link_status': method, 'link_score': score,
                  'canonical_question': None, 'message': '', 'suggestions': [],
                  'candidates': [{'id': qid, 'label': self.entities[qid].get('display_label', self.entities[qid]['label'])}
                                 for qid in candidates[:5]]}
        if not movie:
            result['status'] = 'needs_clarification' if method == 'ambiguous' else 'unanswerable'
            result['message'] = '电影存在歧义，请补充年份或使用 QID。' if method == 'ambiguous' else '未能确定已收录的电影，请搜索片名、别名或使用 QID。'
            return result
        entity = self.entities[movie]
        title = entity.get('question_title', entity['label'])
        # English title masking reuses the baseline linker. Chinese matching
        # is kept separately, so its relation words are not discarded as ASCII.
        chinese = fold(question)
        for name in sorted([entity['label'], *entity.get('aliases', [])], key=len, reverse=True):
            if re.search(r'[\u3400-\u9fff]', name):
                chinese = chinese.replace(fold(name), ' ')
        chinese = ' '.join(re.findall(r'[\u3400-\u9fff]+', chinese))
        text = self.linker.mask(question, movie) + ' ' + chinese
        if re.search(UNSUPPORTED, text):
            result.update(status='unanswerable', message='当前支持导演、类型、制作国家和原始语言；这个问题还需要其他事实。')
            return result
        relations = [relation for relation, pattern in PATTERNS.items() if re.search(pattern, text)]
        if (re.match(r'\s*(?:is|are|was|were|does|do|did)\b', fold(question))
                or re.search(r'是不是|是否(?:是|为|為|由)', chinese)):
            result['message'] = '当前提供事实列表；请直接询问导演、类型、制作国家或原始语言。'
            return result
        if len(relations) != 1:
            result['message'] = '请一次询问一种关系。' if relations else '请说明想了解导演、类型、制作国家还是原始语言。'
            result['suggestions'] = [CANONICAL[r].format(title=title) for r in relations or CANONICAL]
            return result
        relation = relations[0]
        result.update(status='ready', relation_id=relation, canonical_question=CANONICAL[relation].format(title=title),
                      message='已将自然问法转换为标准问题；答案仍由当前知识库与所选系统给出。')
        if not entity.get('facts', {}).get(relation):
            result.update(status='unanswerable', message='已识别问题，但当前知识库缺少这项事实。')
        return result


def refusal(interpretation):
    return {'system': '自由问法', 'movie': interpretation['movie_id'], 'answer': REFUSAL,
            'evidence': [], 'relation': interpretation['relation_id'], 'message': interpretation['message']}


def answer(question, data_dir=DEFAULT_DATA, model_path=DEFAULT_MODEL):
    demo = Demo(load_neural=False, data_dir=data_dir, model_path=model_path)
    parsed = FreeformParser(demo.movie_ids, demo.entities, demo.linker).parse(question)
    if parsed['status'] == 'ready':
        if demo.linker.link(parsed['canonical_question'])[0] != parsed['movie_id']:
            parsed.update(status='needs_clarification', message='标准片名仍有歧义，请从电影列表选择明确版本。')
        else:
            return {'question': question, 'mode': 'freeform', 'interpretation': parsed,
                    'result': demo.answer_a0(parsed['canonical_question'])}
    return {'question': question, 'mode': 'freeform', 'interpretation': parsed, 'result': refusal(parsed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question', required=True)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    result = answer(args.question, args.data_dir, args.model_path)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
