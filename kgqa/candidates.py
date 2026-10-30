"""Conservative bilingual title retrieval: approximate candidates need confirmation."""
from __future__ import annotations

from collections import defaultdict, Counter
from difflib import SequenceMatcher
import re
import unicodedata
from opencc import OpenCC

_SIMPLIFIED = OpenCC('t2s')

AMBIGUOUS_WORDS = {'what', 'who', 'when', 'where', 'which', 'how', 'movie', 'movies', 'film', 'films',
                   'cast', 'director', 'actors', 'genre', 'language', 'country', 'original',
                   '电影', '電影', '导演', '導演', '演员', '演員', '类型', '類型', '故事', '国家', '语言', '作品'}


def normalized(text):
    text = _SIMPLIFIED.convert(unicodedata.normalize('NFKC', text)).casefold()
    return re.sub(r'[^\w\u3400-\u9fff]+', ' ', text, flags=re.UNICODE).strip()


def grams(text):
    compact = text.replace(' ', '')
    return {compact[i:i+2] for i in range(len(compact)-1)}


class CandidateIndex:
    def __init__(self, movies):
        self.movies = {m['id']: m for m in movies}
        self.names = defaultdict(set)
        self.postings = defaultdict(set)
        for movie in movies:
            for name in [movie['label'], movie.get('label_zh') or '', *movie.get('aliases', [])]:
                name = normalized(name)
                if len(name) >= 2:
                    self.names[name].add(movie['id'])
        for name in self.names:
            for gram in grams(name):
                self.postings[gram].add(name)
        self.matchers = []
        for name, ids in self.names.items():
            pattern = re.escape(name)
            if re.match(r'[a-z0-9]', name):
                pattern = r'(?<![\w])' + pattern
            if re.search(r'[a-z0-9]$', name):
                pattern += r'(?![\w])'
            self.matchers.append((name, re.compile(pattern), ids))

    def card(self, qid, score=1., method='exact'):
        movie = self.movies[qid]
        return {'id': qid, 'label': movie['label'], 'label_zh': movie.get('label_zh'),
                'year': movie.get('release_year'), 'score': round(score, 4), 'method': method}

    def search(self, query, limit=5):
        query = normalized(query)
        if query.upper() in self.movies:
            return [self.card(query.upper(), method='qid')]
        if len(query) < 2:
            return []
        candidates = Counter()
        for gram in grams(query):
            candidates.update(self.postings.get(gram, []))
        names = [n for n, _ in candidates.most_common(600)]
        names.extend(self.names.get(query) and [query] or [])
        scores = {}
        for name in names:
            ratio = SequenceMatcher(None, name, query, autojunk=False).ratio()
            score = 1. if name == query else max(ratio, .72 + .18 * len(query)/len(name) if query in name else 0.)
            if score < .42:
                continue
            for qid in self.names[name]:
                if score > scores.get(qid, 0.):
                    scores[qid] = score
        return [self.card(qid, scores[qid], 'candidate') for qid in sorted(scores, key=lambda q: (-scores[q], q))[:limit]]

    def mentions(self, question):
        """Exact complete aliases only. Never promote a retrieved typo to a fact."""
        text = normalized(question)
        explicit = list(dict.fromkeys(q.upper() for q in re.findall(r'(?<![a-z0-9])Q[1-9]\d*(?![a-z0-9])', question, re.I)))
        if explicit:
            if any(q not in self.movies for q in explicit):
                return [], [], 'not_found'
            return explicit, [], 'qid'
        spans = []
        quoted_names = {normalized(m) for m in re.findall(r'[《"“](.*?)[》"”]', question)}
        for name, matcher, ids in self.matchers:
            if (name in AMBIGUOUS_WORDS or len(name) < 4 and re.fullmatch(r'[a-z0-9 ]+', name)) and text != name:
                # Short ordinary words in a question are not reliable titles.
                if name not in quoted_names:
                    continue
            for match in matcher.finditer(text):
                spans.append((match.start(), match.end(), ids))
        chosen = []
        for span in sorted(spans, key=lambda s: (-(s[1]-s[0]), s[0])):
            if not any(span[0] < s[1] and s[0] < span[1] for s in chosen):
                chosen.append(span)
        ids = []
        ambiguous = set()
        ambiguous_spans = 0
        for left, right, matching in sorted(chosen):
            name = text[left:right]
            raw_pattern = re.escape(name).replace(r'\ ', r'\W+')
            subtitle = re.search(raw_pattern + r'\s*[:：]\s*\S', unicodedata.normalize('NFKC', question).casefold())
            extension = re.match(r'\s*(?:\d|(?:part|chapter|episode|volume|vol)\b|[ivx]+\b)', text[right:])
            if len(matching) > 1 or extension or subtitle:
                ambiguous.update(matching)
                ambiguous_spans += 1
            else:
                ids.extend(sorted(matching))
        if ambiguous:
            if ambiguous_spans > 1:
                return list(dict.fromkeys(ids)), [], 'needs_clarification'
            return list(dict.fromkeys(ids)), [self.card(q) for q in sorted(ambiguous)[:5]], 'needs_confirmation'
        if ids:
            return list(dict.fromkeys(ids)), [], 'exact'
        fragments = re.findall(r'[《"“](.*?)[》"”]', question)
        if not fragments:
            fragment = re.sub(r'(?i)\b(?:who|what|which|when|where|was|were|is|are|the|of|in|did|directed|director|cast|actors|screenwriter|genre|budget|box|office|release|date|movie|film)\b', ' ', question)
            fragment = re.sub(r'谁|是谁|导演|执导|演员|主演|编剧|票房|预算|成本|上映|日期|什么时候|哪年|拍摄地|类型|原始语言|制作国家|的|是|什么|请问|介绍一下', ' ', fragment)
            fragments = [fragment]
        cards = {}
        for fragment in fragments:
            for card in self.search(fragment):
                if card['score'] > cards.get(card['id'], {}).get('score', 0):
                    cards[card['id']] = card
        return [], sorted(cards.values(), key=lambda c: (-c['score'], c['id']))[:5], 'needs_confirmation' if cards else 'not_found'
