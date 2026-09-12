from difflib import SequenceMatcher
import re
import unicodedata


def normalize(text):
    return re.sub(r'[^a-z0-9]+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


class EntityLinker:
    def __init__(self, movie_ids, entities):
        self.names = {}
        for qid in movie_ids:
            entity = entities[qid]
            for name in [entity['label'], *entity.get('aliases', [])]:
                key = normalize(name)
                if len(key) >= 2:
                    self.names.setdefault(key, set()).add(qid)

    def link(self, question):
        query = normalize(question)
        padded = f' {query} '
        exact = [(len(name), ids) for name, ids in self.names.items() if f' {name} ' in padded]
        if exact:
            ids = max(exact, key=lambda item: item[0])[1]
            return (next(iter(ids)), 1.0, 'exact') if len(ids) == 1 else (None, 0.0, 'ambiguous')
        best = max(((SequenceMatcher(None, name, query).quick_ratio(), ids) for name, ids in self.names.items()), default=(0, set()))
        return (next(iter(best[1])), best[0], 'fuzzy') if best[0] >= .72 and len(best[1]) == 1 else (None, best[0], 'not_found')

