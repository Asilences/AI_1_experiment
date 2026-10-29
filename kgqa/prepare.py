from __future__ import annotations

import argparse
from collections import Counter
import csv
import logging
from pathlib import Path
import random

from .common import ROOT, REFUSAL, answer_text, read_json, sha256, setup_logging, utc_now, write_json, write_jsonl

QUESTIONS = {
    'P57': ['Who directed {title}?', 'Who was the director of {title}?', 'Name the director or directors of {title}.'],
    'P136': ['What genre is {title}?', 'Which genres does {title} belong to?', 'Name the genre or genres of {title}.'],
    'P495': ['Which country did {title} originate in?', 'What is the country of origin of {title}?', 'Name the countries of origin of {title}.'],
    'P364': ['What is the original language of {title}?', 'Which languages was {title} originally made in?', 'Name the original language or languages of {title}.'],
}
UNKNOWN_QUESTIONS = ['Who wrote the screenplay for {title}?', 'What was the budget of {title}?', 'Which company distributed {title}?']
PRESERVED_SPLIT_POLICY = 'Preserve all previous movie assignments; seed-42 shuffle additions to reach 70/15/15.'


def split_movies(movies, seed=42):
    ids = [movie['id'] for movie in movies]
    random.Random(seed).shuffle(ids)
    n = len(ids)
    return {'train': ids[:int(n * .70)], 'valid': ids[int(n * .70):int(n * .85)], 'test': ids[int(n * .85):]}


def extend_splits(movies, previous, seed=42):
    ids = [movie['id'] for movie in movies]
    if len(ids) != len(set(ids)):
        raise ValueError('Movie IDs must be unique')
    if set(previous) != {'train', 'valid', 'test'} or any(not isinstance(v, list) for v in previous.values()):
        raise ValueError('Previous splits must contain train, valid and test lists')
    old_ids = sum(previous.values(), [])
    if len(old_ids) != len(set(old_ids)) or not set(old_ids) <= set(ids):
        raise ValueError('Previous movie splits must be disjoint and included in the new snapshot')
    size = len(ids)
    targets = {'train': int(size * .70), 'valid': int(size * .85) - int(size * .70), 'test': size - int(size * .85)}
    needed = {split: targets[split] - len(previous[split]) for split in targets}
    if any(n < 0 for n in needed.values()):
        raise ValueError('Previous split exceeds the expanded target size')
    old_set = set(old_ids)
    additions = [qid for qid in ids if qid not in old_set]
    random.Random(seed).shuffle(additions)
    result, cursor = {}, 0
    for split, count in needed.items():
        result[split] = list(previous[split]) + additions[cursor:cursor + count]
        cursor += count
    return result


def build(snapshot_path, output_dir=None, base_splits=None):
    snapshot = read_json(snapshot_path)
    entities, movies = snapshot['entities'], snapshot['movies']
    splits = extend_splits(movies, read_json(base_splits)) if base_splits else split_movies(movies)
    processed = Path(output_dir) if output_dir else ROOT / 'data' / 'processed'
    if output_dir is not None and processed.exists() and any(processed.iterdir()):
        raise FileExistsError(f'Refusing to overwrite processed dataset: {processed}')
    split_for = {qid: split for split, ids in splits.items() for qid in ids}
    qa, triples = [], []
    for movie in movies:
        title, split = movie['label'], split_for[movie['id']]
        question_title = movie.get('question_title', title)
        for relation, answer_ids in movie['facts'].items():
            triples.extend((movie['id'], relation, tail) for tail in answer_ids if tail in entities and entities[tail]['label'])
            if relation not in movie['qa_relations']:
                continue
            names = [entities[qid]['label'] for qid in answer_ids]
            for variant, template in enumerate(QUESTIONS[relation]):
                qa.append({'question_id': f"{movie['id']}_{relation}_{variant}", 'movie_id': movie['id'],
                           'relation_id': relation, 'answer_ids': answer_ids, 'question': template.format(title=question_title),
                           'reference_answers': [answer_text(title, relation, names, 0), answer_text(title, relation, names, 1)],
                           'template_family': variant, 'split': split, 'is_answerable': True})
    for split, ids in splits.items():
        count = max(1, round(sum(row['split'] == split for row in qa) * .10))
        for index in range(count):
            movie = next(m for m in movies if m['id'] == ids[index % len(ids)])
            qa.append({'question_id': f"{movie['id']}_UNKNOWN_{index}", 'movie_id': movie['id'], 'relation_id': 'UNKNOWN',
                       'answer_ids': [], 'question': UNKNOWN_QUESTIONS[index % 3].format(title=movie.get('question_title', movie['label'])),
                       'reference_answers': [REFUSAL], 'template_family': index % 3, 'split': split, 'is_answerable': False})
    processed.mkdir(parents=True, exist_ok=True)
    write_jsonl(processed / 'qa.jsonl', qa); write_json(processed / 'entities.json', entities); write_json(processed / 'splits.json', splits)
    with (processed / 'triples.tsv').open('w', newline='', encoding='utf-8') as stream:
        csv.writer(stream, delimiter='\t', lineterminator='\n').writerows(sorted(set(triples)))
    stats = {'created_at': utc_now(), 'snapshot': str(snapshot_path), 'snapshot_sha256': sha256(snapshot_path),
             'movies': len(movies), 'entities': len(entities), 'triples': len(set(triples)), 'qa': len(qa),
             'qa_by_split': dict(Counter(row['split'] for row in qa)), 'qa_by_relation': dict(Counter(row['relation_id'] for row in qa)),
             'movie_split_sizes': {key: len(value) for key, value in splits.items()}}
    if base_splits:
        stats.update(base_splits_sha256=sha256(base_splits), split_policy=PRESERVED_SPLIT_POLICY)
    write_json(processed / 'manifest.json', stats); logging.info('COMPLETE %s', stats); return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output-dir', type=Path, help='New isolated output directory; existing files are never overwritten.')
    parser.add_argument('--base-splits', type=Path, help='Preserve movie assignments from a previous splits.json.')
    args = parser.parse_args()
    setup_logging('prepare'); build(args.snapshot, args.output_dir, args.base_splits)
