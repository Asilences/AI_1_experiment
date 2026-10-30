from __future__ import annotations

import argparse
from collections import defaultdict
import statistics
import re
import unicodedata

from rouge_score import rouge_scorer
import sacrebleu

from .common import REFUSAL, ROOT, read_json, read_jsonl, write_json
from .linker import normalize

class EntityNameIndex:
    """An explicit snapshot of entity names for one inference/evaluation pass."""

    def __init__(self, entities):
        name_map = {}
        for qid, entity in entities.items():
            label = normalize(entity.get('label', ''))
            if label:
                name_map.setdefault(label, {'labels': set(), 'aliases': set()})['labels'].add(qid)
            for raw_name in entity.get('aliases', []):
                name = normalize(raw_name)
                if name:
                    name_map.setdefault(name, {'labels': set(), 'aliases': set()})['aliases'].add(qid)
        self.names = [(name, name_map[name]) for name in sorted(name_map, key=len, reverse=True)]

    def _aliases(self, name, entry, text, span):
        return entry['aliases']

    def match(self, text, excluded=None, preferred_ids=None):
        preferred_ids = set(preferred_ids or ())
        padded = f' {normalize(text)} '
        found, occupied = set(), []
        for name, entry in self.names:
            token = f' {name} '; offset = 0
            while True:
                position = padded.find(token, offset)
                if position < 0: break
                span = (position + 1, position + 1 + len(name))
                if not any(span[0] < other[1] and other[0] < span[1] for other in occupied):
                    preferred = entry['labels'] or self._aliases(name, entry, text, span)
                    qids = {qid for qid in preferred if qid != excluded}
                    # Context resolves homonyms only; unrelated generated names
                    # remain false positives under the unchanged scoring rules.
                    if len(qids) > 1 and qids & preferred_ids:
                        qids &= preferred_ids
                    if qids:
                        found.update(qids)
                    occupied.append(span)
                offset = position + 1
        return sorted(found)


class AnswerEntityIndex(EntityNameIndex):
    """Decode generated answer payloads; never consult gold answers or relations."""

    def __init__(self, entities):
        super().__init__(entities)
        self.code_aliases, self.regular_aliases = {}, {}
        self.short_alias_forms, self.name_punctuation = {}, {}
        for qid, entity in entities.items():
            for name in [entity.get('label', ''), *entity.get('aliases', [])]:
                self.name_punctuation.setdefault(normalize(name), set()).add(
                    tuple(re.findall(r'[,;]', name)))
            for alias in entity.get('aliases', []):
                short = re.sub(r'[^A-Za-z0-9]', '', alias)
                sensitive = bool(short) and len(short) <= 3 and alias.isascii()
                target = self.code_aliases if sensitive else self.regular_aliases
                target.setdefault(normalize(alias), set()).add(qid)
                if sensitive:
                    self.short_alias_forms.setdefault(normalize(alias), {}).setdefault(qid, set()).add(alias)
        self.head_names = {name.casefold() for entity in entities.values()
                                  for name in [entity.get('label', ''), *entity.get('aliases', [])]
                                  if name}
        self.name_entries = dict(self.names)
        self.name_rank = {name: rank for rank, (name, _) in enumerate(self.names)}
        self.max_name_tokens = max((len(name.split()) for name in self.name_entries), default=0)

    def _aliases(self, name, entry, text, span):
        regular = entry['aliases'] & self.regular_aliases.get(name, set())
        codes = self.code_aliases.get(name, set())
        if codes:
            allowed = set()
            for qid, forms in self.short_alias_forms[name].items():
                for form in forms:
                    for match in re.finditer(r'(?<![A-Za-z0-9])' + re.escape(form) + r'(?![A-Za-z0-9])', text):
                        prefix = normalize(text[:match.start()])
                        start = len(prefix) + 2 if prefix else 1
                        if span == (start, start + len(name)):
                            allowed.add(qid)
            return regular | allowed
        return regular

    def payload(self, text):
        if normalize(text) == normalize(REFUSAL):
            return ''
        # Fixed infix templates have a uniquely recognizable boundary.
        match = re.fullmatch(r'.+?\s+(?:was directed by|belongs to the following genres:|originated in|was originally made in)\s+(.+)',
                             text.strip(), flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1)
        # Resolve the head from the generated sentence itself. This preserves
        # titles containing "are" and directors whose names begin with Are.
        for prefix in ('The director or directors of ', 'The genres of ',
                       'The countries of origin of ', 'The original languages of '):
            if text.casefold().startswith(prefix.casefold()):
                rest = text[len(prefix):]
                boundaries = list(re.finditer(r'\s+are\s+', rest, flags=re.IGNORECASE))
                for boundary in reversed(boundaries):
                    if rest[:boundary.start()].casefold() in self.head_names:
                        return rest[boundary.end():]
                # Unknown heads: accept only an unambiguous template boundary.
                parts = re.split(r'\s+are\s+', rest, flags=re.IGNORECASE)
                if len(parts) == 2:
                    return parts[1]
        return text

    def match(self, text, excluded=None, preferred_ids=None):
        text = self.payload(text)
        preferred_ids = set(preferred_ids or ())
        padded = f' {normalize(text)} '
        tokens = list(re.finditer(r'\S+', padded))
        raw = unicodedata.normalize('NFKC', text).casefold()
        raw_tokens = list(re.finditer(r'[a-z0-9]+', raw))
        candidates = []
        for start, token in enumerate(tokens):
            for end in range(start, min(len(tokens), start + self.max_name_tokens)):
                span = (token.start(), tokens[end].end())
                name = padded[span[0]:span[1]]
                if name in self.name_entries:
                    punctuation = tuple(re.findall(r'[,;]', raw[raw_tokens[start].start():raw_tokens[end].end()]))
                    if punctuation and punctuation not in self.name_punctuation[name]:
                        continue
                    entry = self.name_entries[name]
                    qids = (entry['labels'] or self._aliases(name, entry, text, span)) - {excluded}
                    if len(qids) > 1 and qids & preferred_ids:
                        qids &= preferred_ids
                    raw_span = (raw_tokens[start].start(), raw_tokens[end].end())
                    candidates.append((self.name_rank[name], span, raw_span, qids))

        def contextual_parts(whole_span, whole_raw_span):
            # A collective name can also spell a list of retained fact tails.
            # Require complete coverage, explicit list boundaries, and at least
            # two distinct contextual IDs; partial matches must remain errors.
            parts = [candidate for candidate in sorted(candidates)
                     if candidate[1] != whole_span
                     and whole_span[0] <= candidate[1][0] < candidate[1][1] <= whole_span[1]
                     and candidate[3] & preferred_ids]

            def cover(position, raw_end, ids):
                if position == whole_span[1]:
                    return ids if len(ids) >= 2 else None
                for _, part_span, part_raw_span, part_ids in parts:
                    if part_span[0] < position:
                        continue
                    if position == whole_span[0]:
                        if part_span[0] != position:
                            continue
                    elif not re.fullmatch(r'\s*(?:(?:[,;&]\s*)+(?:(?:and|or)\s*)?|(?:and|or)\s*)',
                                          raw[raw_end:part_raw_span[0]], flags=re.IGNORECASE):
                        continue
                    result = cover(part_span[1], part_raw_span[1], ids | part_ids)
                    if result is not None:
                        return result
                return None

            return cover(whole_span[0], whole_raw_span[0], set())

        found, occupied = set(), []
        for _, span, raw_span, qids in sorted(candidates):
            if any(span[0] < other[1] and other[0] < span[1] for other in occupied):
                continue
            if preferred_ids and not qids & preferred_ids:
                parts = contextual_parts(span, raw_span)
                if parts is not None:
                    qids = parts
            found.update(qids)
            occupied.append(span)
        return sorted(found)


def entity_ids_from_text(text, entities, excluded=None, *, index=None, preferred_ids=None):
    # A fresh default index sees dictionary edits; repeated callers can explicitly
    # reuse a snapshot instead of relying on a process-global identity cache.
    if index is None:
        index = EntityNameIndex(entities)
    return index.match(text, excluded, preferred_ids=preferred_ids)


def _index_rows(rows, name):
    by_id = {}
    for position, row in enumerate(rows, 1):
        question_id = row.get('question_id')
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f'{name} row {position} must have a nonempty question_id')
        if question_id in by_id:
            raise ValueError(f'Duplicate {name.lower()} question_id: {question_id!r}')
        by_id[question_id] = row
    return by_id


def _validate_alignment(gold_rows, predictions, *, allow_missing=False, allow_extra=False):
    if not gold_rows:
        raise ValueError('Cannot evaluate an empty gold set')
    gold_by_id = _index_rows(gold_rows, 'Gold')
    by_id = _index_rows(predictions, 'Prediction')
    if not by_id:
        raise ValueError('Cannot evaluate empty predictions')
    unknown = by_id.keys() - gold_by_id.keys()
    if unknown and not allow_extra:
        raise ValueError(f'Unknown prediction question IDs: {sorted(unknown)!r}')
    missing = gold_by_id.keys() - by_id.keys()
    if missing and not allow_missing:
        raise ValueError(f'Missing predictions for question IDs: {sorted(missing)!r}')
    return by_id


def score_rows(gold_rows, predictions, entities, *, allow_extra_predictions=False):
    by_id = _validate_alignment(gold_rows, predictions, allow_extra=allow_extra_predictions)
    hypotheses, references = [], []
    exact, f1s, refusal, latencies = [], [], [], []
    answerable_exact, answerable_f1s = [], []
    relation = defaultdict(list)
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_values = []
    name_index = None
    for gold in gold_rows:
        pred = by_id[gold['question_id']]
        if not gold.get('reference_answers'):
            raise ValueError(f"Gold row {gold['question_id']!r} must contain at least one reference answer")
        predicted_ids = pred.get('answer_ids')
        if predicted_ids is None:
            if name_index is None:
                name_index = EntityNameIndex(entities)
            predicted_ids = name_index.match(pred['answer'], gold['movie_id'])
        gold_set, predicted_set = set(gold['answer_ids']), set(predicted_ids)
        if not gold_set:
            correct_refusal = normalize(pred['answer']) == normalize(REFUSAL)
            item_exact = float(correct_refusal); item_f1 = float(correct_refusal); refusal.append(item_exact)
        else:
            item_exact = float(gold_set == predicted_set)
            precision = len(gold_set & predicted_set) / len(predicted_set) if predicted_set else 0
            recall = len(gold_set & predicted_set) / len(gold_set)
            item_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
            answerable_exact.append(item_exact); answerable_f1s.append(item_f1)
        exact.append(item_exact); f1s.append(item_f1); relation[gold['relation_id']].append((item_exact, item_f1))
        hypotheses.append(pred['answer']); references.append(gold['reference_answers'])
        rouge_values.append(max(rouge.score(reference, pred['answer'])['rougeL'].fmeasure for reference in gold['reference_answers']))
        if 'latency_ms' in pred: latencies.append(float(pred['latency_ms']))
    reference_streams = [[refs[min(i, len(refs)-1)] for refs in references] for i in range(max(map(len, references)))]
    result = {'count': len(gold_rows), 'entity_exact': 100 * statistics.mean(exact), 'entity_macro_f1': 100 * statistics.mean(f1s),
              'bleu': sacrebleu.corpus_bleu(hypotheses, reference_streams, tokenize='13a').score,
              'rouge_l': 100 * statistics.mean(rouge_values),
              'by_relation': {key: {'exact': 100 * statistics.mean(v[0] for v in values),
                                    'f1': 100 * statistics.mean(v[1] for v in values), 'count': len(values)} for key, values in relation.items()}}
    result['answerable_count'] = len(answerable_exact); result['refusal_count'] = len(refusal)
    if answerable_exact:
        result['answerable_entity_exact'] = 100 * statistics.mean(answerable_exact)
        result['answerable_entity_macro_f1'] = 100 * statistics.mean(answerable_f1s)
    if refusal: result['refusal_accuracy'] = 100 * statistics.mean(refusal)
    if latencies:
        ordered = sorted(latencies); result['latency_mean_ms'] = statistics.mean(latencies); result['latency_p95_ms'] = ordered[min(len(ordered)-1, int(.95 * len(ordered)))]
    return result


def evaluate(run_id, split='test'):
    predictions = read_jsonl(ROOT / 'runs' / run_id / f'predictions_{split}.jsonl')
    all_gold = [row for row in read_jsonl(ROOT / 'data' / 'processed' / 'qa.jsonl') if row['split'] == split]
    # Validate against the complete split before selecting an explicit smoke
    # subset, so filtering cannot conceal duplicate or unknown predictions.
    predicted_ids = _validate_alignment(all_gold, predictions, allow_missing=True).keys()
    gold = [row for row in all_gold if row['question_id'] in predicted_ids]
    if len(gold) != len(all_gold):
        config = read_json(ROOT / 'runs' / run_id / 'config.json')
        if not config.get('smoke'):
            raise ValueError(f'Prediction coverage is incomplete: {len(gold)}/{len(all_gold)}')
    result = score_rows(gold, predictions, read_json(ROOT / 'data' / 'processed' / 'entities.json'))
    result['gold_total'] = len(all_gold); result['prediction_coverage'] = 100 * len(gold) / max(1, len(all_gold))
    write_json(ROOT / 'runs' / run_id / f'metrics_{split}.json', result); return result


def evaluate_robustness(run_id):
    gold = read_jsonl(ROOT / 'data' / 'processed' / 'robustness.jsonl')
    predictions = read_jsonl(ROOT / 'runs' / run_id / 'predictions_robustness.jsonl')
    entities = read_json(ROOT / 'data' / 'processed' / 'entities.json')
    # Extra rows are only legitimate within a variant after the entire file has
    # passed strict coverage and uniqueness validation.
    _validate_alignment(gold, predictions)
    result = {}
    for variant in sorted({row['variant'] for row in gold}):
        subset = [row for row in gold if row['variant'] == variant]
        result[variant] = score_rows(subset, predictions, entities, allow_extra_predictions=True)
    write_json(ROOT / 'runs' / run_id / 'metrics_robustness.json', result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('run_id'); parser.add_argument('--split', default='test'); parser.add_argument('--robustness', action='store_true'); args = parser.parse_args()
    print(evaluate_robustness(args.run_id) if args.robustness else evaluate(args.run_id, args.split))


