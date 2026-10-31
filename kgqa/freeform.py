"""An explicit natural-question adapter, separate from the experiment baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import unicodedata

from .common import REFUSAL, ROOT, write_json
from .chinese import ChineseAdapter
from .demo import Demo
from .linker import EntityLinker, normalize

DEFAULT_DATA = ROOT / 'data/processed/movies_5000_20261029'
DEFAULT_MODEL = ROOT / 'models/pipeline_5000_20261029/model.pkl'
CANONICAL = {
    'P57': 'Who directed {title}?', 'P136': 'What genre is {title}?',
    'P495': 'Which country did {title} originate in?',
    'P364': 'What is the original language of {title}?',
}
METHOD = 'bounded_rules_v2'
PATTERNS = {
    'P57': r'\b(?:director|directors|directed|directing|directorial|helmed|filmmaker|filmmakers)\b|导演|導演|执导|執導|谁拍|誰拍',
    'P136': r'\b(?:genre|genres)\b|\b(?:kind|type|category)\b.*\b(?:movie|film)\b|类型|類型|什么片|什麼片|什么类别|什麼類別',
    'P495': r'\b(?:country|countries|nation|nations|origin|originate|originated)\b|\bwhere\b.*\bfrom\b|哪国|哪國|哪个国家|哪個國家|制片国家|製片國家|制作国家|製作國家|哪里出品|哪裡出品',
    'P364': r'\b(?:language|languages)\b|原声|原聲|原始语言|原始語言|什么语|什麼語|哪种语言|哪種語言',
}
UNSUPPORTED = (
    r'\b(?:budget|box office|runtime|screenplay|writer|writers|write|wrote|written|cast|actor|actors|producer|producers|'
    r'composer|composed|rating|ratings|award|awards|plot|story|recommend|recommendation|born|filmed|filming|shot|location|'
    r'released|when|why|earn|earned|cost|sequel|starring|played|ban|banned|banning|dub|dubbed|dubbing|subtitle|subtitles)\b'
    r'|\b(?:how many|other films|other movies|better than|set in|based in)\b'
    r'|\brelease\b.*\b(?:date|year)\b|演员|演員|主演|编剧|編劇|预算|預算|票房|评分|評分|时长|時長|上映|'
    r'制片人|製片人|配乐|配樂|剧情|劇情|推荐|推薦|好看|拍摄地|拍攝地|取景|在哪拍|在哪裡拍|'
    r'配音|译制|譯製|翻译|翻譯|字幕|禁映|封禁'
)
# These qualifiers cannot be executed by the four-property lookup adapter.
# Check them after masking titles, so e.g. a title containing "After" remains
# a valid entity and is not mistaken for a temporal filter.
CONSTRAINTS = (
    r'\b(?:not|never|without|except|excluding|exclude|before|after|since|until|during|between|'
    r'earlier|later|earliest|latest|first|last|only|count|compare|comparison|compared|versus|vs|'
    r'better|best|worst|top|higher|lower|more|less|over|under|both|neither|either)\b'
    r'|\b(?:isn|aren|wasn|weren|don|doesn|didn|hasn|haven|hadn|won|wouldn|couldn)\s+t\b'
    r'|\b(?:at least|at most|how many|number of|other than|rather than)\b'
    r'|不是|并非|並非|没有|沒有|不要|不属于|不屬於|不由|未由|谁没|誰沒|谁不|誰不|'
    r'排除|除了|除外|之前|之后|之後|以前|以后|以後|早于|晚于|超过|超過|低于|低於|'
    r'至少|至多|最多|最少|只有|仅|僅|比较|比較|相比|共同|同一|分别|分別|多少|几个|幾個|数量|數量|第[一二三四五六七八九十\d]'
)
AMBIGUOUS_SCOPE = (
    r'\b(?:spoken|speak|speaks|speaking|dialogue|dialogues|dialog|dialogs|dialect|dialects)\b'
    r'|说什么话|說什麼話|说什么语言|說什麼語言|对白|對白|台词|臺詞'
    r'|\bbehind the camera\b|掌镜|掌鏡|摄影|攝影'
)


def unconsumed(text):
    """Reject unimplemented residual conditions instead of dropping them.

    This deliberately bounded grammar trades coverage for explicit semantics;
    it is not a claim to understand arbitrary natural-language instructions.
    """
    # Intent detection may contain wide patterns such as "where ... from".
    # Never remove those entire spans here: a region or other qualifier inside
    # the span must remain visible to this residual-condition check.
    for relation in ('P57', 'P364'):
        text = re.sub(PATTERNS[relation], ' ', text)
    text = re.sub(
        r'\b(?:genre|genres|kind|type|category|country|countries|nation|nations|origin|originate|originated)\b'
        r'|类型|類型|什么片|什麼片|什么类别|什麼類別|哪国|哪國|哪个国家|哪個國家|'
        r'制片国家|製片國家|制作国家|製作國家|哪里出品|哪裡出品', ' ', text)
    text = re.sub(r'\bq\d+\b|movie_title', ' ', text)
    text = re.sub(
        r'\b(?:a|an|the|and|or|of|for|to|in|on|at|by|with|from|about|into|as|'
        r'is|are|was|were|be|been|being|do|does|did|has|have|had|'
        r'i|me|my|you|your|it|its|this|that|these|those|they|their|'
        r'who|what|which|where|can|could|would|will|please|tell|give|show|list|name|'
        r'know|find|identify|answer|provide|want|need|like|information|info|'
        r'movie|movies|film|films|title|titles|person|people|individual|individuals|'
        r'credit|credits|credited|receive|receives|received|hold|holds|responsible|original|originally|'
        r'production|produced|made|belongs|following)\b', ' ', text)
    text = re.sub(
        r'请问|請問|请|請|告诉我|告訴我|我想知道|想知道|能否|可以|麻烦|麻煩|'
        r'这部|這部|这个|這個|电影|電影|影片|片子|一下|一位|一部|'
        r'是什么|是什麼|是谁|是誰|哪些|什么|什麼|哪位|哪种|哪種|哪个|哪個|'
        r'属于|屬於|的|是|由|有|谁|誰|和|与|與|及|呢|啊|呀|吧', ' ', text)
    return bool(re.search(r'[a-z0-9\u3400-\u9fff]', text))


def fold(text):
    return unicodedata.normalize('NFKC', text).casefold()


class FreeformParser:
    """Conservative, bounded rules; this is not a learned language model."""

    def __init__(self, movie_ids, entities, linker=None):
        self.movie_ids, self.entities = set(movie_ids), entities
        self.linker = linker or EntityLinker(movie_ids, entities)
        self.chinese = ChineseAdapter(entities, self.movie_ids)
        self.cjk_names = {}
        # Reuse the same frozen optional Chinese metadata and year-qualified
        # aliases as the interactive demo instead of maintaining another list.
        for name, ids in self.chinese.names.items():
            if len(name) >= 2 and re.search(r'[\u3400-\u9fff]', name):
                self.cjk_names.setdefault(name, set()).update(ids)

    def _scope_text(self, question, movie=None):
        """Mask known title spans before reading scope words.

        With no resolved movie, this is used only for an early unsupported-
        intent check. It cannot choose a movie or authorize an answer.
        """
        english = self.linker.mask(question, movie) if movie else normalize(question)
        if movie:
            for left, right, _ in reversed(self.linker._exact_mentions(english, movie)):
                english = english[:left] + 'movie_title' + english[right:]
        if not movie:
            for left, right, _ in reversed(self.linker._exact_mentions(english)):
                english = english[:left] + 'movie_title' + english[right:]
        chinese = fold(question)
        names = self.cjk_names if movie is None else {
            name: ids for name, ids in self.cjk_names.items() if movie in ids}
        for name in sorted(names, key=len, reverse=True):
            chinese = chinese.replace(name, ' ')
        chinese = ' '.join(re.findall(r'[\u3400-\u9fff]+', chinese))
        if movie and self.entities[movie].get('release_year'):
            year = str(self.entities[movie]['release_year'])
            english = re.sub(r'(?<!\d)' + re.escape(year) + r'(?!\d)', ' ', english)
            chinese = re.sub(r'年版|年', ' ', chinese) if year in question else chinese
        return english + ' ' + chinese

    def _resolve(self, question, link_question=None):
        text = fold(question)
        qids = set(re.findall(r'(?<![a-z0-9])q\d+(?![a-z0-9])', text, flags=re.I))
        qids = {qid.upper() for qid in qids}
        if qids - self.movie_ids:
            return None, 0., 'not_found', []
        mentions = []
        for name, ids in self.cjk_names.items():
            start = 0
            while (left := text.find(name, start)) >= 0:
                right = left + len(name)
                mentions.append((left, right, ids))
                start = right
        mentions = EntityLinker._non_overlapping(mentions)
        # An unknown numbered sequel must not become the shorter known film.
        if any(re.match(r'\s*\d', text[right:]) for _, right, _ in mentions):
            return None, 0., 'not_found', []
        ids = set().union(*(item[2] for item in mentions)) if mentions else set()
        movie_id, score, status = self.linker.link(question if link_question is None else link_question)
        if movie_id:
            ids.add(movie_id)
        if qids:
            chosen = next(iter(qids))
            all_mentions = mentions + self.linker._exact_mentions(normalize(question))
            if (len(qids) > 1 or any(chosen not in item[2] for item in all_mentions)
                    or movie_id and movie_id != chosen):
                return None, 0., 'ambiguous', sorted(ids | qids)
            return chosen, 1., 'qid', []
        english_ids = set().union(*(item[2] for item in self.linker._exact_mentions(normalize(question))))
        if len(ids | english_ids) > 1:
            return None, 0., 'ambiguous', sorted(ids | english_ids)
        if status == 'ambiguous':
            return None, score, 'ambiguous', sorted(ids | english_ids)
        if len(ids) > 1:
            return None, 0., 'ambiguous', sorted(ids)
        if ids:
            chosen = next(iter(ids))
            return chosen, 1. if mentions else score, 'alias' if mentions else status, []
        return None, score, status, []

    def parse(self, question):
        if not isinstance(question, str) or not question.strip() or len(question) > 1000:
            raise ValueError('请输入 1—1000 个字符的电影问题。')
        # 中文关系词可紧接完整英文片名；保留原句中的其他电影提及以检查歧义。
        subject = re.fullmatch(
            r'([\x20-\x7e]+)的(?:导演|導演|类型|類型|制作国家|製作國家|制片国家|製片國家|'
            r'国家|國家|原始语言|原始語言|语言|語言)[\u3400-\u9fff]*[?？。.!！]*', question.strip())
        link_question = (subject.group(1).strip()
                         if subject and normalize(subject.group(1)) in self.linker.unicode_names else question)
        prepared = self.chinese.prepare(question)
        if prepared.language == 'zh' and prepared.status == 'success':
            # This adapter already checks every explicit title, movie conflicts
            # and year constraints. Reuse its exact match before fuzzy search,
            # including an English title touching Chinese question particles.
            # All original semantic/residual checks below still apply.
            movie, score, method, candidates = prepared.movie_id, 1., 'exact', []
        elif prepared.language == 'zh' and prepared.status == 'not_found':
            movie, score, method, candidates = None, 0., 'not_found', []
        else:
            movie, score, method, candidates = self._resolve(question, link_question)
        result = {'question': question.strip(), 'method': METHOD, 'status': 'needs_clarification',
                  'movie_id': movie, 'relation_id': None, 'link_status': method, 'link_score': score,
                  'canonical_question': None, 'message': '', 'suggestions': [],
                  'reason_code': '',
                  'candidates': [{'id': qid, 'label': self.entities[qid].get('display_label', self.entities[qid]['label'])}
                                 for qid in candidates[:5]]}
        if not movie:
            # Scope can be rejected without selecting a movie. For example a
            # common-word film title in an unsupported intent must not turn
            # "who composed ..." into a request to compare two movies.
            if re.search(UNSUPPORTED, self._scope_text(question)):
                result.update(status='unanswerable', reason_code='unsupported_property',
                              message='四关系模式仅支持导演、类型、制作国家和原始语言；这个问题需要其他事实，可尝试扩展工具模式。')
                return result
            result['status'] = 'needs_clarification' if method == 'ambiguous' else 'unanswerable'
            result['reason_code'] = 'ambiguous_movie' if method == 'ambiguous' else 'movie_not_found'
            result['message'] = '电影存在歧义，请补充年份或使用 QID。' if method == 'ambiguous' else '未能确定已收录的电影，请搜索片名、别名或使用 QID。'
            return result
        entity = self.entities[movie]
        title = entity.get('question_title', entity['label'])
        # Use the original input for semantic checks even when a separately
        # linked title is used for mixed-language entity resolution.
        text = self._scope_text(question, movie)
        chinese = ' '.join(re.findall(r'[\u3400-\u9fff]+', text))
        if re.search(UNSUPPORTED, text):
            result.update(status='unanswerable', reason_code='unsupported_property',
                          message='四关系模式仅支持导演、类型、制作国家和原始语言；这个问题需要其他事实，可尝试扩展工具模式。')
            return result
        if re.search(CONSTRAINTS, text):
            result.update(reason_code='unimplemented_constraint',
                          message='问题包含否定、筛选、比较或数量条件；四关系模式不能执行这些条件，请改为直接查询一项事实或使用扩展工具模式。')
            return result
        if re.search(AMBIGUOUS_SCOPE, text):
            result.update(reason_code='ambiguous_property_scope',
                          message='请明确询问原始语言或导演；片中对白语言不等于原始语言，“掌镜/behind the camera”也不一定指导演。')
            return result
        if method == 'fuzzy':
            result.update(reason_code='confirm_fuzzy_movie',
                          message='片名为相似匹配，请先确认电影候选，并使用完整片名或 QID 重新提问。',
                          candidates=[{'id': movie, 'label': entity.get('display_label', entity['label'])}])
            return result
        relations = [relation for relation, pattern in PATTERNS.items() if re.search(pattern, text)]
        if (re.match(r'\s*(?:is|are|was|were|does|do|did)\b', fold(question))
                or re.search(r'是不是|是否(?:是|为|為|由)', chinese)):
            result.update(reason_code='yes_no_question', message='当前提供事实列表；请直接询问导演、类型、制作国家或原始语言。')
            return result
        if len(relations) != 1:
            result['reason_code'] = 'multiple_properties' if relations else 'unknown_property'
            result['message'] = '请一次询问一种关系。' if relations else '请说明想了解导演、类型、制作国家还是原始语言。'
            result['suggestions'] = [CANONICAL[r].format(title=title) for r in relations or CANONICAL]
            return result
        if unconsumed(text):
            result.update(reason_code='unconsumed_condition',
                          message='问题中还有未能解释的条件或限定；为避免遗漏原意，请简化为单部电影的一项事实查询。')
            return result
        relation = relations[0]
        result.update(status='ready', relation_id=relation, canonical_question=CANONICAL[relation].format(title=title),
                      message='已将自然问法转换为标准问题；答案仍由当前知识库与所选系统给出。')
        if not entity.get('facts', {}).get(relation):
            result.update(status='unanswerable', reason_code='missing_evidence', message='已识别问题，但当前知识库缺少这项事实。')
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
