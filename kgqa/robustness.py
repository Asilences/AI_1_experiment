from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import random
import re

from .common import ROOT, read_json, read_jsonl, write_json, write_jsonl

PARAPHRASES = {
    'P57': 'Can you tell me who helmed {title}?', 'P136': 'How would you classify {title} by genre?',
    'P495': 'Where was {title} produced originally?', 'P364': 'In which language or languages was {title} first made?'}


def typo(text, seed):
    """Swap distinct adjacent letters; return text unchanged if no swap exists."""
    rng = random.Random(seed)
    positions = [i for i in range(len(text) - 1)
                 if text[i].isalpha() and text[i + 1].isalpha() and text[i] != text[i + 1]]
    if not positions: return text
    index = rng.choice(positions); chars = list(text); chars[index], chars[index + 1] = chars[index + 1], chars[index]
    return ''.join(chars)


def _sample_rows(rows, limit, seed):
    """Cover movies in seeded rounds, preferring less-sampled relations."""
    rng = random.Random(seed)
    groups = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows, key=lambda row: row['question_id']):
        groups[row['movie_id']][row['relation_id']].append(row)
    priorities = {}
    for movie in sorted(groups):
        relations = sorted(groups[movie])
        for relation in relations:
            rng.shuffle(groups[movie][relation])
        rng.shuffle(relations)
        priorities[movie] = {relation: index for index, relation in enumerate(relations)}
    selected, counts = [], Counter()
    active = sorted(groups)
    while active and len(selected) < limit:
        rng.shuffle(active)
        remaining = []
        for movie in active:
            relations = [relation for relation, pool in groups[movie].items() if pool]
            relation = min(relations, key=lambda item: (counts[item], priorities[movie][item]))
            selected.append(groups[movie][relation].pop())
            counts[relation] += 1
            if any(groups[movie].values()):
                remaining.append(movie)
            if len(selected) == limit:
                break
        active = sorted(remaining)
    return selected


def _replace_title(question, title, replacement):
    if not title:
        return question
    # A short title such as "It" must not change letters inside "written".
    return re.sub(r'(?<!\w)' + re.escape(title) + r'(?!\w)', lambda match: replacement,
                  question, flags=re.IGNORECASE)


def _distribution(rows):
    return {
        'count': len(rows), 'movies': len({row['movie_id'] for row in rows}),
        'by_relation': dict(sorted(Counter(row['relation_id'] for row in rows).items())),
        'by_movie': dict(sorted(Counter(row['movie_id'] for row in rows).items())),
        'source_question_ids': [row.get('source_question_id', row['question_id']) for row in rows],
    }


def build(limit=100, seed=42):
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError('limit must be a positive integer')
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('seed must be an integer')
    entities = read_json(ROOT / 'data' / 'processed' / 'entities.json')
    all_rows = read_jsonl(ROOT / 'data' / 'processed' / 'qa.jsonl')
    if not all_rows:
        raise ValueError('QA data is empty')
    seen, movie_splits = set(), defaultdict(set)
    for row in all_rows:
        qid = row.get('question_id')
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError('QA data: question_id must be a non-empty string')
        if qid in seen:
            raise ValueError(f'QA data: duplicate question_id {qid!r}')
        seen.add(qid)
        for field in ('movie_id', 'split', 'relation_id', 'question'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'QA data: {qid!r} requires a non-empty {field}')
        if not isinstance(row.get('is_answerable'), bool):
            raise ValueError(f'QA data: {qid!r} requires boolean is_answerable')
        if row['is_answerable'] and row['relation_id'] not in PARAPHRASES:
            raise ValueError(f"QA data: unsupported answerable relation {row['relation_id']!r}")
        movie_splits[row['movie_id']].add(row['split'])
    leaking = sorted(movie for movie, splits in movie_splits.items() if len(splits) > 1)
    if leaking:
        raise ValueError(f'QA data: movies occur in multiple splits: {leaking!r}')
    all_rows.sort(key=lambda row: row['question_id'])
    rows = [row for row in all_rows if row['split'] == 'test' and row['is_answerable']]
    if not rows:
        raise ValueError('No answerable test questions are available for robustness evaluation')
    selected = _sample_rows(rows, min(limit, len(rows)), seed)
    output = []
    alias_unavailable = defaultdict(list)
    typo_unavailable = defaultdict(list)
    unchanged_typos = []
    for row in selected:
        qid = row['question_id']
        entity = entities.get(row['movie_id'], {})
        title = entity.get('label')
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"QA data: movie {row['movie_id']!r} requires an entity label")
        typo_title = typo(title, f'robustness-typo:{seed}:{qid}')
        typo_question = _replace_title(row['question'], title, typo_title)
        if typo_question == row['question']:
            unchanged_typos.append(qid)
            reason = 'no_swappable_title_letters' if typo_title == title else 'title_not_in_question'
            typo_unavailable[reason].append(qid)
        variants = {'clean': row['question'], 'paraphrase': PARAPHRASES[row['relation_id']].format(title=title),
                    'typo': typo_question}
        aliases = sorted({name.strip() for name in entity.get('aliases', [])
                          if name.strip() and name.strip().casefold() != title.strip().casefold()},
                         key=lambda name: (len(name), name.casefold(), name))
        if aliases:
            alias_question = _replace_title(row['question'], title, aliases[0])
            if alias_question != row['question']:
                variants['alias'] = alias_question
            else:
                alias_unavailable['title_not_in_question'].append(qid)
        else:
            alias_unavailable['no_distinct_alias'].append(qid)
        for kind, question in variants.items():
            output.append({**row, 'question_id': f'robust_{kind}_{qid}', 'source_question_id': qid,
                           'paired_clean_question_id': f'robust_clean_{qid}', 'variant': kind,
                           'question': question, 'perturbation_applied': kind != 'clean' and question != row['question']})
    negative_pool = [row for row in all_rows if row['split'] == 'test' and not row['is_answerable']]
    unanswerable = _sample_rows(negative_pool, min(50, len(negative_pool)), seed + 1)
    for row in unanswerable:
        output.append({**row, 'question_id': f"robust_unanswerable_{row['question_id']}",
                       'source_question_id': row['question_id'], 'paired_clean_question_id': None,
                       'variant': 'unanswerable', 'perturbation_applied': False})
    by_variant = {kind: [row for row in output if row['variant'] == kind]
                  for kind in ('clean', 'paraphrase', 'typo', 'alias', 'unanswerable')}
    clean_by_source = {row['source_question_id']: row for row in by_variant['clean']}
    manifest = {
        'seed': seed, 'requested_limit': limit, 'source_split': 'test',
        'sampling_algorithm': 'seeded_movie_round_robin_with_greedy_relation_balance_v1',
        'sampling_note': 'One question per remaining movie per round; prefer less-sampled relations. '
                         'This is a coverage-oriented diagnostic sample, not a statistically representative sample.',
        'eligible_answerable': _distribution(rows), 'sampled_answerable': _distribution(selected),
        'eligible_unanswerable': _distribution(negative_pool),
        'unanswerable_limit': 50, 'unanswerable_seed': seed + 1,
        'generated_questions': len(output),
        'variants': {kind: {**_distribution(pool),
                            'perturbations_applied': sum(row['perturbation_applied'] for row in pool)}
                     for kind, pool in by_variant.items()},
        'comparisons': {
            kind: {'matched_count': len(by_variant[kind]),
                   'all_clean_denominator': len(by_variant['clean']),
                   'variant_denominator': len(by_variant[kind]),
                   'matched_clean_denominator': len(by_variant[kind]),
                   'matched_source_question_ids': [row['source_question_id'] for row in by_variant[kind]],
                   'matched_clean_records': [clean_by_source[row['source_question_id']]
                                             for row in by_variant[kind]],
                   'pairs': [{'clean_question_id': row['paired_clean_question_id'],
                              'variant_question_id': row['question_id']} for row in by_variant[kind]]}
            for kind in ('paraphrase', 'typo', 'alias')},
        'alias_unavailable': dict(sorted(alias_unavailable.items())),
        'alias_comparison_note': 'Compare alias only with clean rows listed in its matched pairs; '
                                 'the all-clean denominator may be larger. Missing aliases are not fabricated.',
        'typo_scope': 'Adjacent distinct letters within the movie title; references and answer IDs are unchanged.',
        'typo_unchanged_source_question_ids': unchanged_typos,
        'typo_unavailable': dict(sorted(typo_unavailable.items())),
    }
    write_jsonl(ROOT / 'data' / 'processed' / 'robustness.jsonl', output)
    write_json(ROOT / 'data' / 'processed' / 'robustness_manifest.json', manifest)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42); args = parser.parse_args(); print(len(build(args.limit, args.seed)))
