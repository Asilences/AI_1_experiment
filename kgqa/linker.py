from difflib import SequenceMatcher
import re
import unicodedata


def normalize(text):
    return re.sub(r'[^a-z0-9]+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


class EntityLinker:
    """Match complete title spans, and refuse uncertain fuzzy matches.

    ``quick_ratio`` is only an inexpensive upper bound for the actual ratio.
    """

    def __init__(self, movie_ids, entities, fuzzy_threshold=.88, ambiguity_margin=.06):
        self.names = {}
        self.fuzzy_threshold, self.ambiguity_margin = fuzzy_threshold, ambiguity_margin
        for qid in movie_ids:
            entity = entities[qid]
            for name in [entity['label'], *entity.get('aliases', [])]:
                key = normalize(name)
                if len(key) >= 2:
                    self.names.setdefault(key, set()).add(qid)
        self.max_tokens = max((len(name.split()) for name in self.names), default=0)
        self.fuzzy_names = {}
        for name, ids in self.names.items():
            # Short fuzzy titles are too easily confused with ordinary words.
            if len(name) >= 6:
                self.fuzzy_names.setdefault(len(name.split()), []).append((name, ids))
        self.extensions = {}
        for name, ids in self.names.items():
            words = name.split()
            for start in range(len(words)):
                for end in range(start + 1, len(words) + 1):
                    short_name = ' '.join(words[start:end])
                    short_ids = self.names.get(short_name, set())
                    other_ids = ids - short_ids
                    if short_ids and other_ids and short_name != name:
                        self.extensions.setdefault(short_name, []).append(
                            (words[:start], words[end:], other_ids))

    @staticmethod
    def _non_overlapping(mentions):
        chosen = []
        for mention in sorted(mentions, key=lambda item: (-(item[1] - item[0]), item[0])):
            if not any(mention[0] < other[1] and other[0] < mention[1] for other in chosen):
                chosen.append(mention)
        return sorted(chosen, key=lambda item: item[0])

    def _exact_mentions(self, query, movie_id=None):
        tokens = list(re.finditer(r'\S+', query))
        mentions = []
        # Lookup question spans instead of scanning every title and alias.
        for start in range(len(tokens)):
            for count in range(1, min(self.max_tokens, len(tokens) - start) + 1):
                left, right = tokens[start].start(), tokens[start + count - 1].end()
                ids = self.names.get(query[left:right])
                if ids and (movie_id is None or movie_id in ids):
                    mentions.append((left, right, ids))
        return self._non_overlapping(mentions)

    def _extended_match(self, query, exact):
        """Check distinguishing words before accepting an exact nested title.

        Matching the added words separately prevents a long shared title prefix
        from making an unrelated sequel look like a high-confidence typo.
        """
        tokens = list(re.finditer(r'\S+', query))
        starts = {token.start(): i for i, token in enumerate(tokens)}
        ends = {token.end(): i + 1 for i, token in enumerate(tokens)}
        candidates, extension_seen = {}, False
        for left, right, _ in exact:
            anchor_start, anchor_end = starts[left], ends[right]
            for before, after, ids in self.extensions.get(query[left:right], []):
                # A recognizable extension cue with an unknown completion must
                # not silently fall back to the shorter movie.
                cues = []
                if before and anchor_start:
                    cues.append((before[-1], tokens[anchor_start - 1].group()))
                if after and anchor_end < len(tokens):
                    cues.append((after[0], tokens[anchor_end].group()))
                # Adjacent sequel numbers identify different titles, even if
                # the remaining extension words are almost identical.
                if any(expected.isdecimal() and actual.isdecimal() and expected != actual
                       for expected, actual in cues):
                    extension_seen = True
                    continue
                if any(len(expected) >= 4 and SequenceMatcher(
                        None, expected, actual, autojunk=False).ratio() >= self.fuzzy_threshold
                       for expected, actual in cues):
                    extension_seen = True
                prefix_counts = range(max(1, len(before) - 1), len(before) + 2) if before else (0,)
                suffix_counts = range(max(1, len(after) - 1), len(after) + 2) if after else (0,)
                for prefix_count in prefix_counts:
                    for suffix_count in suffix_counts:
                        start, end = anchor_start - prefix_count, anchor_end + suffix_count
                        if start < 0 or end > len(tokens):
                            continue
                        scores = []
                        for expected, added in (
                            (before, tokens[start:anchor_start]),
                            (after, tokens[anchor_end:end]),
                        ):
                            if expected:
                                actual = ' '.join(token.group() for token in added)
                                scores.append(SequenceMatcher(
                                    None, ' '.join(expected), actual, autojunk=False).ratio())
                        score = min(scores)
                        if score < max(0.0, self.fuzzy_threshold - self.ambiguity_margin):
                            continue
                        for qid in ids:
                            if qid not in candidates or score > candidates[qid][0]:
                                candidates[qid] = (score, tokens[start].start(), tokens[end - 1].end())
        if not candidates:
            return (None, 0.0, 'ambiguous', []) if extension_seen else None
        ranked = sorted(candidates, key=lambda qid: (-candidates[qid][0], qid))
        winner = ranked[0]
        score, left, right = candidates[winner]
        if score < self.fuzzy_threshold:
            return (None, score, 'ambiguous', []) if extension_seen else None
        if (len(ranked) > 1 and score - candidates[ranked[1]][0] <= self.ambiguity_margin
                or any(start < left or end > right for start, end, _ in exact)):
            return None, score, 'ambiguous', []
        return winner, score, 'fuzzy', [(left, right, {winner})]

    def _match(self, query):
        exact = self._exact_mentions(query)
        if exact:
            ids = set().union(*(item[2] for item in exact))
            if len(ids) == 1:
                extended = self._extended_match(query, exact)
                if extended is not None:
                    return extended
                return next(iter(ids)), 1.0, 'exact', exact
            return None, 0.0, 'ambiguous', []

        tokens = list(re.finditer(r'\S+', query))
        candidates = {}
        minimum = max(0.0, self.fuzzy_threshold - self.ambiguity_margin)
        for title_tokens, names in self.fuzzy_names.items():
            for count in range(max(1, title_tokens - 1), title_tokens + 2):
                for start in range(len(tokens) - count + 1):
                    left, right = tokens[start].start(), tokens[start + count - 1].end()
                    span = query[left:right]
                    if len(span) < 5:
                        continue
                    for name, ids in names:
                        matcher = SequenceMatcher(None, name, span, autojunk=False)
                        if matcher.real_quick_ratio() < minimum or matcher.quick_ratio() < minimum:
                            continue
                        score = matcher.ratio()
                        if score < minimum:
                            continue
                        for qid in ids:
                            if qid not in candidates or score > candidates[qid][0]:
                                candidates[qid] = (score, left, right)
        if not candidates:
            return None, 0.0, 'not_found', []
        ranked = sorted(candidates, key=lambda qid: (-candidates[qid][0], qid))
        winner = ranked[0]
        score, left, right = candidates[winner]
        if score < self.fuzzy_threshold:
            return None, score, 'not_found', []
        if len(ranked) > 1 and score - candidates[ranked[1]][0] <= self.ambiguity_margin:
            return None, score, 'ambiguous', []
        return winner, score, 'fuzzy', [(left, right, {winner})]

    def link(self, question):
        return self._match(normalize(question))[:3]

    def mask(self, question, movie_id):
        """Mask a title or alias without replacing parts of other words."""
        query = normalize(question)
        if not movie_id:
            return query
        linked_id, _, _, matched = self._match(query)
        # A fuzzy long title can contain an exact short alias of the same
        # entity. Prefer the linker's complete span before the alias fallback.
        mentions = matched if linked_id == movie_id else self._exact_mentions(query, movie_id)
        for left, right, _ in reversed(mentions):
            query = query[:left] + 'movie_title' + query[right:]
        return query

