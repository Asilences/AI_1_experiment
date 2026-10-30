"""Restore uniquely identifiable tokenizer-damaged names from retrieved facts.

This postprocessor does not choose an answer entity or fill in omitted answers.
The caller supplies only entity IDs from the facts actually retrieved at inference.
"""
from __future__ import annotations

from collections import defaultdict
import re


def _matches(text, name):
    return re.finditer(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text, re.IGNORECASE)


def _overlaps(start, end, spans):
    return any(start < other_end and other_start < end for other_start, other_end in spans)


class NameRecovery:
    """Cache exact label roundtrips; recover only unambiguous unknown-token loss."""

    def __init__(self, entities, tokenizer):
        self.entities = entities
        self.tokenizer = tokenizer
        self._cache = {}

    def _form(self, qid):
        if qid not in self._cache:
            label = self.entities.get(qid, {}).get('label', '')
            decoded, damaged = label, False
            if label:
                ids = self.tokenizer(text_target=label, truncation=False,
                                     add_special_tokens=True)['input_ids']
                decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
                unknown = getattr(self.tokenizer, 'unk_token_id', None)
                damaged = unknown is not None and unknown in ids and decoded != label
            self._cache[qid] = (label, decoded, damaged)
        return self._cache[qid]

    def recover(self, raw_answer, candidate_ids):
        """Return answer plus replacements, whose offsets refer to raw_answer.

        Candidate IDs must come from the linked movie's retained facts, never
        reference answers. Empty candidates (including failed linking) are inert.
        Three alphanumeric characters are required in a damaged form to avoid
        interpreting a stray letter or common two-letter fragment as a name.
        """
        forms = {qid: self._form(qid) for qid in sorted(set(candidate_ids))}
        owners = defaultdict(set)
        protected = []
        for qid, (label, decoded, _) in forms.items():
            for name in (label, decoded):
                if name:
                    owners[name.casefold()].add(qid)
            if label:
                protected.extend(match.span() for match in _matches(raw_answer, label))

        matches = []
        for qid, (label, decoded, damaged) in forms.items():
            if (not damaged or sum(char.isalnum() for char in decoded) < 3
                    or owners[decoded.casefold()] != {qid}):
                continue
            for match in _matches(raw_answer, decoded):
                if not _overlaps(*match.span(), protected):
                    matches.append((match.start(), match.end(), qid, label))

        # Longest first; every match is found in the original text, not replacements.
        selected = []
        occupied = []
        for start, end, qid, label in sorted(matches, key=lambda x: (-(x[1] - x[0]), x[0], x[2])):
            if not _overlaps(start, end, occupied):
                selected.append((start, end, qid, label))
                occupied.append((start, end))

        replacements, parts, cursor = [], [], 0
        for start, end, qid, label in sorted(selected):
            parts.extend((raw_answer[cursor:start], label))
            replacements.append({'entity_id': qid, 'before': raw_answer[start:end],
                                 'after': label, 'start': start, 'end': end})
            cursor = end
        parts.append(raw_answer[cursor:])
        return {'raw_answer': raw_answer, 'answer': ''.join(parts), 'replacements': replacements}
