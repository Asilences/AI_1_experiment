"""Offline Chinese grounding for the interactive movie demo.

The adapter resolves a movie and one supported intent; it never retrieves facts
or predicts answers. English experiment questions pass through byte for byte.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata


RELATION_LABELS_ZH = {'P57': '导演', 'P136': '类型', 'P495': '制片国家', 'P364': '原始语言'}
_TEMPLATES = {
    'P57': 'Who directed {title}?',
    'P136': 'What genre is {title}?',
    'P495': 'Which country did {title} originate in?',
    'P364': 'What is the original language of {title}?',
}
_CJK = re.compile(r'[\u3400-\u9fff\uf900-\ufaff]')
_QUOTED = re.compile(r'《([^《》]+)》|“([^“”]+)”|"([^"\n]+)"|‘([^‘’]+)’|(?<![A-Za-z0-9])\'([^\'\n]+)\'(?![A-Za-z0-9])')
_QID = re.compile(r'(?<![A-Za-z0-9])Q\d+(?![A-Za-z0-9])', re.I)
_INTENTS = {
    'P57': re.compile(r'导演|導演|执导|執導|谁拍|誰拍|谁导|誰導|哪位拍|什么人拍'),
    'P136': re.compile(r'类型|類型|类别|類別|流派|题材|題材|什么片|甚麼片|哪种片|哪種片'),
    'P495': re.compile(r'国家|國家|哪国(?!语)|哪國(?!語)|制片国|製片國|产地|產地|出品国|出品國|来自哪|來自哪|出自哪'),
    'P364': re.compile(r'语言|語言|原声|原聲|语种|語種|什么话|甚麼話|哪种话|哪種話|哪国语|哪國語'),
}
_NEGATION = re.compile(r'不是|并非|並非|没有|沒有|不要|不属于|不屬於|不由|未由|谁没|誰沒|谁不|誰不|别告诉|別告訴')
_UNSUPPORTED = re.compile(
    r'演员|演員|主演|编剧|編劇|剧本|劇本|票房|预算|預算|成本|上映|发行|發行|剧情|劇情|故事|'
    r'评分|評分|时长|時長|片长|片長|制片人|製片人|制作人|製作人|配乐|配樂|摄影|攝影|'
    r'拍摄地|拍攝地|哪里拍|哪裡拍|哪里取景|哪裡取景|奖项|獎項|获奖|獲獎|'
    r'拍过|拍過|其他电影|其他電影|详细介绍|詳細介紹|是不是|是否|对不对|對不對|'
    r'为什么|為什麼|为何|為何|怎么|怎麼|如何|出生|国籍|國籍|年龄|年齡|其它电影|其它電影|'
    r'\b(?:actor|cast|screenplay|budget|box office|release date|runtime|plot|producer)\b', re.I)


def _key(text):
    text = unicodedata.normalize('NFKC', str(text)).casefold()
    return re.sub(r'\s+', ' ', text).strip()


def _mask(text, spans):
    chars = list(text)
    for start, end in spans:
        chars[start:end] = ' ' * (end - start)
    return ''.join(chars)


@dataclass(frozen=True)
class PreparedQuestion:
    language: str
    original_question: str
    model_question: str
    movie_id: str | None = None
    relation_id: str | None = None
    status: str = 'pass_through'
    reason: str = ''
    matched_title: str | None = None


class ChineseAdapter:
    def __init__(self, entities, movie_ids, label_catalog=None):
        self.entities = entities
        self.movie_ids = frozenset(movie_ids)
        self.display_labels = {}
        self.display_aliases = {}
        for qid, entity in entities.items():
            label = entity.get('label_zh') or next(
                (alias.strip('《》') for alias in entity.get('aliases', []) if _CJK.search(alias)), None)
            if label:
                self.display_labels[qid] = label
        if label_catalog is None:
            label_catalog = Path(__file__).with_name('locales') / 'zh_entities.json'
        if isinstance(label_catalog, (str, Path)):
            path = Path(label_catalog)
            label_catalog = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        for qid, record in label_catalog.get('entities', {}).items():
            # A fixture or a separate knowledge graph can reuse a QID. Match its
            # frozen English label before applying optional display metadata.
            if qid not in entities or record.get('label_en') != entities[qid].get('label'):
                continue
            if record.get('label_zh'):
                self.display_labels[qid] = record['label_zh']
            self.display_aliases[qid] = tuple(value for value in record.get('aliases_zh', [])
                                              if isinstance(value, str) and _CJK.search(value))
        self.names = {}
        for qid in self.movie_ids:
            entity = entities.get(qid, {})
            values = [qid, entity.get('label'), entity.get('label_zh'), entity.get('question_title'),
                      self.display_labels.get(qid), *entity.get('aliases', []),
                      *self.display_aliases.get(qid, ())]
            for value in values:
                if not value:
                    continue
                key = _key(value)
                self.names.setdefault(key, set()).add(qid)
                year = entity.get('release_year')
                if year and not re.search(r'\b(?:19|20)\d{2}\b', key):
                    for alias in (f'{key} ({year})', f'{key}({year})', f'{key} {year}',
                                  f'{key}{year}年版', f'{year}年版{key}'):
                        self.names.setdefault(alias, set()).add(qid)
        self._names_by_prefix = {}
        for name, ids in self.names.items():
            # Single characters and intent words need explicit title delimiters.
            if len(name) >= 2 and not any(pattern.fullmatch(name) for pattern in _INTENTS.values()):
                self._names_by_prefix.setdefault(name[:2], []).append((name, ids))

    def _mentions(self, text, excluded):
        mentions = []
        for start in range(len(text) - 1):
            for name, ids in self._names_by_prefix.get(text[start:start + 2], []):
                if not text.startswith(name, start):
                    continue
                end = start + len(name)
                if any(start < right and left < end for left, right in excluded):
                    continue
                # Latin titles need word boundaries; CJK titles can join Chinese particles.
                if (name[0].isascii() and name[0].isalnum() and start
                        and text[start - 1].isascii() and text[start - 1].isalnum()):
                    continue
                if (name[-1].isascii() and name[-1].isalnum() and end < len(text)
                        and text[end].isascii() and text[end].isalnum()):
                    continue
                if re.match(r'\s*\d', text[end:]) or re.match(r'(?:第[一二三四五六七八九十\d]+部|续集|續集)', text[end:]):
                    continue
                mentions.append((start, end, ids, text[start:end]))
        chosen = []
        for mention in sorted(mentions, key=lambda item: (-(item[1] - item[0]), item[0])):
            if not any(mention[0] < item[1] and item[0] < mention[1] for item in chosen):
                chosen.append(mention)
        return sorted(chosen)

    def prepare(self, question):
        """Return exact movie grounding and canonical English, or a Chinese refusal."""
        if not _CJK.search(question):
            return PreparedQuestion('en', question, question)
        text = _key(question)
        explicit = list(_QUOTED.finditer(text))
        spans = [(match.start(), match.end()) for match in explicit]
        mentions = self._mentions(text, spans)
        outside = _mask(text, spans + [(item[0], item[1]) for item in mentions])
        # An English question with a Chinese title belongs to the unchanged English path.
        if not _CJK.search(outside) and re.search(r'\b(?:who|what|which|name|where|how|is|was|did)\b', outside):
            return PreparedQuestion('en', question, question)

        def refuse(status, reason, movie_id=None, matched_title=None, relation_id=None):
            return PreparedQuestion('zh', question, question, movie_id, relation_id,
                                    status, reason, matched_title)

        if _NEGATION.search(outside):
            return refuse('unsupported', '暂不支持否定、排除或反向查询；请直接询问一部电影的导演、类型、制片国家或原始语言。')
        if _UNSUPPORTED.search(outside):
            return refuse('unsupported', '目前支持单部电影的导演、类型、制片国家和原始语言；请每次询问其中一项。')
        if re.search(r'[吗嗎][?？。.!！\s]*$', outside) and not re.search(r'谁|誰|什么|甚麼|哪', outside):
            return refuse('unsupported', '暂不支持是非判断；请直接询问导演是谁、类型、制片国家或原始语言。')
        relations = [relation for relation, pattern in _INTENTS.items() if pattern.search(outside)]
        if len(relations) != 1:
            return refuse('unsupported', '请每次询问一项：导演、类型、制片国家或原始语言。')
        relation = relations[0]
        # Every explicit title must match in full. Never rescue an unknown title by substring.
        all_mentions = []
        for match in explicit:
            title = next(value for value in match.groups() if value is not None)
            ids = self.names.get(_key(title), set())
            if not ids:
                return refuse('not_found', f'本地图谱中未找到《{title}》；请使用完整片名或电影 QID。', relation_id=relation)
            all_mentions.append((match.start(), match.end(), ids, title))
        all_mentions.extend(mentions)
        qids = list(_QID.finditer(outside))
        for match in qids:
            qid = match.group().upper()
            if qid not in self.movie_ids:
                return refuse('not_found', f'本地图谱中没有电影 {qid}。', relation_id=relation)
            all_mentions.append((match.start(), match.end(), {qid}, qid))
        if not all_mentions:
            return refuse('not_found', '未识别到本地图谱中的电影；请使用《完整片名》或电影 QID。', relation_id=relation)
        # A remake's year can be outside its title delimiters.
        years = set(re.findall(r'(?<!\d)((?:19|20)\d{2})(?!\d)', outside))
        if len(years) == 1:
            year = next(iter(years))
            filtered = []
            for left, right, ids, title in all_mentions:
                matching = {qid for qid in ids if str(self.entities[qid].get('release_year', '')) == year}
                if not matching:
                    return refuse('not_found', f'片名与指定年份 {year} 不匹配；请检查电影版本。', relation_id=relation)
                filtered.append((left, right, matching, title))
            all_mentions = filtered
        unique_ids = set().union(*(item[2] for item in all_mentions))
        intersection = set.intersection(*(set(item[2]) for item in all_mentions))
        if len(all_mentions) > 1 and not intersection:
            return refuse('unsupported', '检测到多部电影；请每次只询问一部电影。', relation_id=relation)
        candidates = intersection if len(all_mentions) > 1 else unique_ids
        if len(candidates) != 1:
            options = '、'.join(f'{self.format_entity(qid)}（{self.entities[qid].get("release_year", qid)}，{qid}）'
                               for qid in sorted(candidates))
            return refuse('ambiguous', f'片名对应多个版本：{options}。请补充年份或电影 QID。', relation_id=relation)
        movie_id = next(iter(candidates))
        matched_title = all_mentions[0][3]
        entity = self.entities[movie_id]
        title = entity.get('question_title') or entity.get('label') or movie_id
        model_question = _TEMPLATES[relation].format(title=title)
        return PreparedQuestion('zh', question, model_question, movie_id, relation,
                                'success', '', matched_title)

    def format_entity(self, qid):
        """Use supplied metadata only; preserve original names when Chinese is unavailable."""
        entity = self.entities.get(qid, {})
        return self.display_labels.get(qid) or entity.get('label') or qid

    def format_answer(self, movie_id, relation_id, answer_ids):
        """Render predicted answer IDs without consulting movie facts."""
        names = '、'.join(self.format_entity(qid) for qid in dict.fromkeys(answer_ids))
        if not names or relation_id not in _TEMPLATES:
            return '现有知识图谱没有足够的信息回答这个问题。'
        title = self.format_entity(movie_id)
        return {
            'P57': f'《{title}》的导演是{names}。',
            'P136': f'《{title}》的类型是{names}。',
            'P495': f'《{title}》的制片国家是{names}。',
            'P364': f'《{title}》的原始语言是{names}。',
        }[relation_id]

    def localize_generated(self, answer, movie_id):
        """Translate complete generated templates, retaining their actual answer payload.

        This deliberately does not repair a wrong relation or replace an unknown
        prediction with a gold fact. Unrecognized templates remain verbatim.
        """
        from .common import REFUSAL

        if answer == REFUSAL:
            return '现有知识图谱没有足够的信息回答这个问题。'
        entity = self.entities.get(movie_id, {})
        titles = {value for value in [entity.get('label'), entity.get('question_title'),
                                      entity.get('label_zh'), self.display_labels.get(movie_id),
                                      *entity.get('aliases', []),
                                      *self.display_aliases.get(movie_id, ())] if value}
        patterns = [
            ('P57', r'{head} was directed by (.+)\.'),
            ('P57', r'The director or directors of {head} are (.+)\.'),
            ('P136', r'{head} belongs to the following genres: (.+)\.'),
            ('P136', r'The genres of {head} are (.+)\.'),
            ('P495', r'{head} originated in (.+)\.'),
            ('P495', r'The countries of origin of {head} are (.+)\.'),
            ('P364', r'The original languages of {head} are (.+)\.'),
            ('P364', r'{head} was originally made in (.+)\.'),
        ]
        for title in sorted(titles, key=len, reverse=True):
            for relation, pattern in patterns:
                match = re.fullmatch(pattern.format(head=re.escape(title)), answer)
                if not match:
                    continue
                if re.search(r'[.!?]\s+[A-Z]', match.group(1)):
                    return answer
                payload = self._localize_payload(match.group(1))
                head = self.format_entity(movie_id)
                return f'《{head}》的{RELATION_LABELS_ZH[relation]}是{payload}。'
        return answer

    def _localize_payload(self, payload):
        if not hasattr(self, '_label_translations'):
            labels = {}
            for qid, entity in self.entities.items():
                translated = self.display_labels.get(qid)
                if not translated:
                    continue
                for label in [entity.get('label'), *entity.get('aliases', [])]:
                    if label:
                        labels.setdefault(label, set()).add(translated)
            self._label_translations = {label: next(iter(values)) for label, values in labels.items()
                                        if len(values) == 1}
            alternatives = '|'.join(re.escape(label) for label in sorted(self._label_translations,
                                                                          key=len, reverse=True))
            self._label_pattern = re.compile(r'(?<!\w)(?:' + alternatives + r')(?!\w)') if alternatives else None
        if self._label_pattern is None:
            return payload
        return self._label_pattern.sub(lambda match: self._label_translations[match.group()], payload)
