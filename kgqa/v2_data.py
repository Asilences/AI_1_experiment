"""Build a separate, qualified movie catalogue without changing V1 data."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import random

from .common import ROOT, read_json, sha256, utc_now, write_json

BASE = ROOT / 'data/raw/expansion_5000_20261029/snapshot_5000.json'
DEST = ROOT / 'data/processed/movies_v2_20261030'


def select_pilot(movies, count=200, seed=42):
    if not 1 <= count <= len(movies):
        raise ValueError('Pilot size must be between 1 and movie count')
    groups = defaultdict(list)
    for movie in sorted(movies, key=lambda row: row['id']):
        facts = movie.get('facts', {})
        year = movie.get('release_year')
        key = (tuple(sorted(facts.get('P495', []))), tuple(sorted(facts.get('P364', []))),
               year // 10 * 10 if year else 'unknown')
        groups[key].append(movie['id'])
    rng = random.Random(seed)
    groups = [groups[key] for key in sorted(groups, key=str)]
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(group)
    chosen = []
    while len(chosen) < count:
        for group in groups:
            if group:
                chosen.append(group.pop())
                if len(chosen) == count:
                    return chosen
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    select = sub.add_parser('select', help='Select a reproducible stratified pilot, not an evaluation result.')
    select.add_argument('--base', type=Path, default=BASE)
    select.add_argument('--count', type=int, default=200)
    select.add_argument('--output', type=Path, default=ROOT / 'output/catalog_v2/selection.json')
    args = parser.parse_args(argv)
    base = read_json(args.base)
    ids = select_pilot(base['movies'], args.count)
    write_json(args.output, {'created_at': utc_now(), 'base_sha256': sha256(args.base),
                            'seed': 42, 'movie_ids': ids, 'count': len(ids),
                            'sampling': 'country/language/known-decade strata round-robin; unknown decade separate; not global random sample'})
    print(f'Selected {len(ids)} movies: {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
