from __future__ import annotations

import argparse
from collections import Counter
import csv
import logging
import random

from .common import ROOT, REFUSAL, answer_text, read_json, sha256, setup_logging, utc_now, write_json, write_jsonl

QUESTIONS = {
    'P57': ['Who directed {title}?', 'Who was the director of {title}?', 'Name the director or directors of {title}.'],
    'P136': ['What genre is {title}?', 'Which genres does {title} belong to?', 'Name the genre or genres of {title}.'],
    'P495': ['Which country did {title} originate in?', 'What is the country of origin of {title}?', 'Name the countries of origin of {title}.'],
    'P364': ['What is the original language of {title}?', 'Which languages was {title} originally made in?', 'Name the original language or languages of {title}.'],
}
UNKNOWN_QUESTIONS = ['Who wrote the screenplay for {title}?', 'What was the budget of {title}?', 'Which company distributed {title}?']


def split_movies(movies, seed=42):
    ids = [movie['id'] for movie in movies]
    random.Random(seed).shuffle(ids)
    n = len(ids)
    return {'train': ids[:int(n * .70)], 'valid': ids[int(n * .70):int(n * .85)], 'test': ids[int(n * .85):]}


def build(snapshot_path):
    snapshot = read_json(snapshot_path)
    entities, movies = snapshot['entities'], snapshot['movies']
    splits = split_movies(movies)
    split_for = {qid: split for split, ids in splits.items() for qid in ids}
    qa, triples = [], []
    for movie in movies:
        title, split = movie['label'], split_for[movie['id']]
        for relation, answer_ids in movie['facts'].items():
            triples.extend((movie['id'], relation, tail) for tail in answer_ids if tail in entities and entities[tail]['label'])
            if relation not in movie['qa_relations']:
                continue
            names = [entities[qid]['label'] for qid in answer_ids]
            for variant, template in enumerate(QUESTIONS[relation]):
                qa.append({'question_id': f"{movie['id']}_{relation}_{variant}", 'movie_id': movie['id'],
                           'relation_id': relation, 'answer_ids': answer_ids, 'question': template.format(title=title),
                           'reference_answers': [answer_text(title, relation, names, 0), answer_text(title, relation, names, 1)],
                           'template_family': variant, 'split': split, 'is_answerable': True})
    for split, ids in splits.items():
        count = max(1, round(sum(row['split'] == split for row in qa) * .10))
        for index in range(count):
            movie = next(m for m in movies if m['id'] == ids[index % len(ids)])
            qa.append({'question_id': f"{movie['id']}_UNKNOWN_{index}", 'movie_id': movie['id'], 'relation_id': 'UNKNOWN',
                       'answer_ids': [], 'question': UNKNOWN_QUESTIONS[index % 3].format(title=movie['label']),
                       'reference_answers': [REFUSAL], 'template_family': index % 3, 'split': split, 'is_answerable': False})
    processed = ROOT / 'data' / 'processed'; processed.mkdir(parents=True, exist_ok=True)
    write_jsonl(processed / 'qa.jsonl', qa); write_json(processed / 'entities.json', entities); write_json(processed / 'splits.json', splits)
    with (processed / 'triples.tsv').open('w', newline='', encoding='utf-8') as stream:
        csv.writer(stream, delimiter='\t', lineterminator='\n').writerows(sorted(set(triples)))
    stats = {'created_at': utc_now(), 'snapshot': str(snapshot_path), 'snapshot_sha256': sha256(snapshot_path),
             'movies': len(movies), 'entities': len(entities), 'triples': len(set(triples)), 'qa': len(qa),
             'qa_by_split': dict(Counter(row['split'] for row in qa)), 'qa_by_relation': dict(Counter(row['relation_id'] for row in qa)),
             'movie_split_sizes': {key: len(value) for key, value in splits.items()}}
    write_json(processed / 'manifest.json', stats); logging.info('COMPLETE %s', stats); return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--snapshot', required=True); args = parser.parse_args()
    setup_logging('prepare'); build(args.snapshot)
