from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from itertools import combinations
from pathlib import Path
import random
import re

from .common import ROOT, read_jsonl, write_json


VISIBLE_FIELDS = ['question_id', 'question', 'reference', 'answer_position', 'answer',
                  'accuracy_0_2', 'completeness_0_2', 'fluency_1_5', 'unsupported_claim_0_1', 'notes']
KEY_FIELDS = ['question_id', 'answer_position', 'system']
SCORE_RANGES = {'accuracy_0_2': (0, 2), 'completeness_0_2': (0, 2),
                'fluency_1_5': (1, 5), 'unsupported_claim_0_1': (0, 1)}


def _index_rows(rows, label):
    """Index once, rejecting IDs that would silently overwrite another answer."""
    indexed = {}
    for row in rows:
        qid = row.get('question_id')
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f'{label}: question_id must be a non-empty string')
        if qid in indexed:
            raise ValueError(f'{label}: duplicate question_id {qid!r}')
        indexed[qid] = row
    return indexed


def _sample_questions(gold, limit, seed):
    """Seeded movie rounds with greedy relation balancing, not a probability sample.

    Each round visits each remaining movie at most once. For that movie it picks
    an available relation with the smallest sample count so far; seeded tie
    orders choose between relations and between paraphrases. Sorting before
    shuffling makes the result independent of JSONL input order. This encourages
    movie coverage and relation diversity, without claiming representativeness.
    """
    rng = random.Random(seed)
    by_movie = defaultdict(lambda: defaultdict(list))
    for qid in sorted(gold):
        row = gold[qid]
        by_movie[row['movie_id']][row['relation_id']].append(qid)
    priorities = {}
    for movie_id in sorted(by_movie):
        relations = sorted(by_movie[movie_id])
        for relation in relations:
            rng.shuffle(by_movie[movie_id][relation])
        rng.shuffle(relations)
        priorities[movie_id] = {relation: i for i, relation in enumerate(relations)}
    selected, relation_counts = [], Counter()
    active = sorted(by_movie)
    while active and len(selected) < limit:
        rng.shuffle(active)
        remaining = []
        for movie_id in active:
            groups = by_movie[movie_id]
            candidates = [relation for relation, ids in groups.items() if ids]
            relation = min(candidates, key=lambda item: (relation_counts[item], priorities[movie_id][item]))
            selected.append(groups[relation].pop())
            relation_counts[relation] += 1
            if any(groups.values()):
                remaining.append(movie_id)
            if len(selected) == limit:
                break
        active = sorted(remaining)
    return selected


def build(neural_run, limit=100, seed=42):
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError('limit must be a positive integer')
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('seed must be an integer')
    all_gold = _index_rows(read_jsonl(ROOT / 'data' / 'processed' / 'qa.jsonl'), 'QA data')
    gold = {qid: row for qid, row in all_gold.items() if row['split'] == 'test' and row['is_answerable']}
    if not gold:
        raise ValueError('No answerable test questions are available for human evaluation')
    systems = {
        'X': _index_rows(read_jsonl(ROOT / 'runs' / 'A0' / 'predictions_test.jsonl'), 'A0 predictions'),
        'Y': _index_rows(read_jsonl(ROOT / 'runs' / neural_run / 'predictions_test.jsonl'), f'{neural_run} predictions'),
    }
    # Validate the whole eligible pool before sampling, so a seed cannot hide
    # incomplete predictions. Matching below is an O(1) dictionary lookup.
    for system, predictions in systems.items():
        missing = sorted(set(gold) - set(predictions))
        if missing:
            preview = ', '.join(missing[:5])
            raise ValueError(f'{system} predictions: missing {len(missing)} eligible question IDs: {preview}')
        for qid in gold:
            if not isinstance(predictions[qid].get('answer'), str):
                raise ValueError(f'{system} predictions: answer must be a string for {qid!r}')

    ids = _sample_questions(gold, min(limit, len(gold)), seed)
    rows, keys = [], []
    for qid in ids:
        order = ['X', 'Y']
        random.Random(f'human-evaluation-order:{seed}:{qid}').shuffle(order)
        for position, system in enumerate(order, 1):
            rows.append({'question_id': qid, 'question': gold[qid]['question'], 'reference': ' || '.join(gold[qid]['reference_answers']),
                         'answer_position': position, 'answer': systems[system][qid]['answer'],
                         'accuracy_0_2': '', 'completeness_0_2': '', 'fluency_1_5': '', 'unsupported_claim_0_1': '', 'notes': ''})
            keys.append({'question_id': qid, 'answer_position': position, 'system': system})

    output = ROOT / 'output'
    output.mkdir(parents=True, exist_ok=True)
    for filename, fields, records in [('human_evaluation.csv', VISIBLE_FIELDS, rows),
                                      ('human_evaluation_key.csv', KEY_FIELDS, keys)]:
        with (output / filename).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    manifest = {
        'seed': seed, 'requested_limit': limit, 'eligible_questions': len(gold),
        'eligible_movies': len({row['movie_id'] for row in gold.values()}),
        'sampled_questions': len(ids), 'sampled_movies': len({gold[qid]['movie_id'] for qid in ids}),
        'sampling_algorithm': 'seeded_movie_round_robin_with_greedy_relation_balance_v1',
        'sampling_note': 'One question per remaining movie per round; prefer less-sampled relations. '
                         'This is a coverage-oriented diagnostic sample, not a statistically representative sample.',
        'sampled_by_relation': dict(sorted(Counter(gold[qid]['relation_id'] for qid in ids).items())),
        'sampled_by_movie': dict(sorted(Counter(gold[qid]['movie_id'] for qid in ids).items())),
        'question_ids': ids, 'system_key': {'X': 'A0', 'Y': neural_run},
        'scores': 'blank; must be completed independently by human reviewers',
    }
    write_json(output / 'human_evaluation_manifest.json', manifest)
    return manifest


def _read_score_csv(path, required):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{Path(path).name}: missing CSV columns: {sorted(missing)}')
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f'{Path(path).name}: duplicate CSV columns')
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f'{Path(path).name}: row contains more values than CSV columns')
    return rows


def _answer_id(row, label):
    question_id, raw_position = row.get('question_id'), row.get('answer_position')
    if not isinstance(question_id, str) or not question_id.strip():
        raise ValueError(f'{label}: question_id must be a non-empty string')
    if not isinstance(raw_position, str) or not re.fullmatch(r'[0-9]+', raw_position.strip()):
        raise ValueError(f'{label}: answer_position must be a positive integer')
    position = int(raw_position)
    if position < 1:
        raise ValueError(f'{label}: answer_position must be a positive integer')
    return question_id, position


def summarize_scores(score_files, key_file, summary_output):
    """Aggregate complete independent human score sheets without changing them."""
    files = [Path(path) for path in score_files]
    if not files or len({path.resolve() for path in files}) != len(files):
        raise ValueError('At least one distinct score file is required; repeated files are not independent')
    if Path(summary_output).resolve() in {path.resolve() for path in [*files, Path(key_file)]}:
        raise ValueError('Summary destination must differ from the score and key files')
    keys = {}
    for row in _read_score_csv(key_file, KEY_FIELDS):
        answer_id = _answer_id(row, 'Answer key')
        if answer_id in keys:
            raise ValueError(f'Answer key: duplicate answer ID {answer_id!r}')
        system = row['system']
        if not isinstance(system, str) or not system.strip():
            raise ValueError('Answer key: system must be a non-empty string')
        keys[answer_id] = system
    if not keys:
        raise ValueError('Answer key must not be empty')

    reviews = []
    for path in files:
        indexed = {}
        for row in _read_score_csv(path, ['question_id', 'answer_position', *SCORE_RANGES]):
            answer_id = _answer_id(row, path.name)
            if answer_id in indexed:
                raise ValueError(f'{path.name}: duplicate answer ID {answer_id!r}')
            if answer_id not in keys:
                raise ValueError(f'{path.name}: unknown answer ID {answer_id!r}')
            scores = {}
            for field, (low, high) in SCORE_RANGES.items():
                raw = row.get(field)
                if not isinstance(raw, str) or not re.fullmatch(r'[+-]?[0-9]+', raw.strip()):
                    raise ValueError(f'{path.name}: {answer_id!r} {field} must be an integer in [{low}, {high}]')
                value = int(raw)
                if not low <= value <= high:
                    raise ValueError(f'{path.name}: {answer_id!r} {field} must be an integer in [{low}, {high}]')
                scores[field] = value
            indexed[answer_id] = scores
        missing = set(keys) - set(indexed)
        if missing:
            raise ValueError(f'{path.name}: missing {len(missing)} answer IDs from the key')
        reviews.append(indexed)

    systems = {}
    for system in sorted(set(keys.values())):
        answer_ids = [answer_id for answer_id, owner in keys.items() if owner == system]
        ratings = [review[answer_id] for review in reviews for answer_id in answer_ids]
        means = {field: sum(rating[field] for rating in ratings) / len(ratings) for field in SCORE_RANGES}
        systems[system] = {'question_count': len({answer_id[0] for answer_id in answer_ids}),
                           'answer_count': len(answer_ids), 'rating_count': len(ratings), 'means': means,
                           'unsupported_claim_percent': 100 * means['unsupported_claim_0_1']}
    agreement = []
    for left, right in combinations(range(len(reviews)), 2):
        items = {}
        for field in SCORE_RANGES:
            equal = sum(reviews[left][answer_id][field] == reviews[right][answer_id][field] for answer_id in keys)
            items[field] = {'compared_count': len(keys), 'agreement_count': equal,
                            'disagreement_count': len(keys) - equal, 'agreement_percent': 100 * equal / len(keys)}
        agreement.append({'reviewers': [f'reviewer_{left + 1}', f'reviewer_{right + 1}'], 'items': items})
    result = {'reviewer_count': len(reviews), 'question_count': len({answer_id[0] for answer_id in keys}),
              'answer_count': len(keys), 'rating_count': len(keys) * len(reviews),
              'reviewers': [{'reviewer_id': f'reviewer_{index + 1}', 'score_file': path.name}
                            for index, path in enumerate(files)],
              'key_file': Path(key_file).name, 'score_ranges': {field: list(bounds) for field, bounds in SCORE_RANGES.items()},
              'systems': systems, 'pairwise_agreement': agreement,
              'aggregation': 'All validated original ratings, without replacing disagreements or changing source CSVs.'}
    write_json(Path(summary_output), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--neural-run', help='Export blank A0/neural blind-review sheets.')
    mode.add_argument('--score-file', type=Path, action='append', help='Completed independent CSV; repeat for each reviewer.')
    parser.add_argument('--key-file', type=Path, help='Answer key matching every reviewed answer.')
    parser.add_argument('--summary-output', type=Path, help='Destination for the validated scoring summary JSON.')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.score_file:
        if args.key_file is None or args.summary_output is None:
            parser.error('--score-file requires --key-file and --summary-output')
        try:
            summarize_scores(args.score_file, args.key_file, args.summary_output)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    else:
        if args.key_file is not None or args.summary_output is not None:
            parser.error('--key-file and --summary-output require --score-file')
        build(args.neural_run, args.limit, args.seed)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
