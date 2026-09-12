from __future__ import annotations

import argparse
import random

from .common import ROOT, read_json, read_jsonl, write_jsonl

PARAPHRASES = {
    'P57': 'Can you tell me who helmed {title}?', 'P136': 'How would you classify {title} by genre?',
    'P495': 'Where was {title} produced originally?', 'P364': 'In which language or languages was {title} first made?'}


def typo(text, seed):
    rng = random.Random(seed); positions = [i for i, char in enumerate(text) if char.isalpha()]
    if len(positions) < 2: return text
    index = rng.choice(positions[:-1]); chars = list(text); chars[index], chars[index + 1] = chars[index + 1], chars[index]
    return ''.join(chars)


def build(limit=100):
    entities = read_json(ROOT / 'data' / 'processed' / 'entities.json')
    rows = [row for row in read_jsonl(ROOT / 'data' / 'processed' / 'qa.jsonl') if row['split'] == 'test' and row['is_answerable']]
    selected = sorted(rows, key=lambda row: row['question_id'])[:limit]
    output = []
    for index, row in enumerate(selected):
        title = entities[row['movie_id']]['label']
        variants = {'clean': row['question'], 'paraphrase': PARAPHRASES[row['relation_id']].format(title=title), 'typo': typo(row['question'], index)}
        aliases = [name for name in entities[row['movie_id']].get('aliases', []) if name.casefold() != title.casefold()]
        if aliases: variants['alias'] = row['question'].replace(title, sorted(aliases, key=len)[0])
        for kind, question in variants.items(): output.append({**row, 'question_id': f"robust_{kind}_{row['question_id']}", 'source_question_id': row['question_id'], 'variant': kind, 'question': question})
    unanswerable = sorted([row for row in read_jsonl(ROOT / 'data' / 'processed' / 'qa.jsonl') if row['split'] == 'test' and not row['is_answerable']], key=lambda row: row['question_id'])[:50]
    for row in unanswerable:
        output.append({**row, 'question_id': f"robust_unanswerable_{row['question_id']}", 'source_question_id': row['question_id'], 'variant': 'unanswerable'})
    write_jsonl(ROOT / 'data' / 'processed' / 'robustness.jsonl', output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--limit', type=int, default=100); args = parser.parse_args(); print(len(build(args.limit)))
