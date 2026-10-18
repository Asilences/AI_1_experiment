from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import random

from .common import ROOT, read_jsonl, write_json


VISIBLE_FIELDS = ['question_id', 'question', 'reference', 'answer_position', 'answer',
                  'accuracy_0_2', 'completeness_0_2', 'fluency_1_5', 'unsupported_claim_0_1', 'notes']
KEY_FIELDS = ['question_id', 'answer_position', 'system']


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neural-run', required=True)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    build(args.neural_run, args.limit, args.seed)
