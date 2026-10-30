"""Build a separate, qualified movie catalogue without changing V1 data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random
import re

from .common import ROOT, read_json, sha256, utc_now, write_json, write_jsonl
from .expand import EntityApiClient, SparqlClient, release_year, values

BASE = ROOT / 'data/raw/expansion_5000_20261029/snapshot_5000.json'
DEST = ROOT / 'data/processed/movies_v2_20261030'
PROPERTIES = {
    'P57': ('导演', 'director', 'entity'), 'P136': ('类型', 'genre', 'entity'),
    'P495': ('制作国', 'country of origin', 'entity'), 'P364': ('原始语言', 'original language', 'entity'),
    'P161': ('演员', 'cast member', 'entity'), 'P58': ('编剧', 'screenwriter', 'entity'),
    'P577': ('上映/发行日期', 'publication date', 'time'), 'P915': ('拍摄地', 'filming location', 'entity'),
    'P840': ('故事地点', 'narrative location', 'entity'), 'P2130': ('制作成本', 'production cost', 'quantity'),
    'P2142': ('票房', 'box office', 'quantity'), 'P725': ('配音演员', 'voice actor', 'entity'),
}


def statements(entity, provenance):
    """Keep scope, precision, unknown snaks and references, not just triples."""
    result = []
    for prop in PROPERTIES:
        for claim in entity.get('claims', {}).get(prop, []):
            if claim.get('rank') == 'deprecated':
                continue
            snak = claim.get('mainsnak', {})
            kind = snak.get('snaktype')
            value = snak.get('datavalue', {}).get('value')
            if snak.get('datatype') == 'wikibase-item' and isinstance(value, dict):
                value = value.get('id')
            result.append({'id': claim.get('id'), 'subject': entity['id'], 'property': prop,
                           'type': PROPERTIES[prop][2], 'snaktype': kind, 'value': value,
                           'rank': claim.get('rank', 'normal'), 'datatype': snak.get('datatype'),
                           'qualifiers': claim.get('qualifiers', {}), 'references': claim.get('references', []),
                           'revision': entity.get('lastrevid'),
                           'source_url': f'https://www.wikidata.org/wiki/{entity["id"]}',
                           **provenance})
    return result


def compact_source(entity):
    return {key: entity[key] for key in ('id', 'lastrevid', 'labels', 'aliases', 'missing', 'redirects') if key in entity} | {
        'claims': {p: entity.get('claims', {}).get(p, []) for p in PROPERTIES}}


def cached_entities(paths, wanted, claims=True):
    found, sources = {}, {}
    for path in sorted(set(map(Path, paths))):
        envelope = read_json(path)
        props = envelope.get('params', {}).get('props', '')
        if claims and 'claims' not in props:
            continue
        payload = envelope.get('response', {}).get('entities', {})
        matched = wanted.intersection(payload)
        if not matched:
            continue
        for qid in matched:
            entity = payload[qid]
            if entity.get('id') != qid or 'missing' in entity:
                continue
            if qid in found and (found[qid].get('lastrevid') or 0) > (entity.get('lastrevid') or 0):
                continue
            found[qid] = compact_source(entity)
            sources[qid] = {'retrieved_at': envelope.get('retrieved_at'), 'cache_file': str(path.resolve()),
                            'request_url': envelope.get('url'), 'request_params': envelope.get('params')}
    return found, sources


def referenced_ids(rows):
    ids = set()
    for row in rows:
        if row['type'] == 'entity' and row['snaktype'] == 'value' and isinstance(row['value'], str):
            ids.add(row['value'])
        # Qualifiers and units need labels too; references remain verbatim.
        for snaks in row['qualifiers'].values():
            for snak in snaks:
                value = snak.get('datavalue', {}).get('value', {})
                if isinstance(value, dict) and value.get('id'):
                    ids.add(value['id'])
        if row['type'] == 'quantity' and isinstance(row['value'], dict):
            unit = row['value'].get('unit', '')
            if re.fullmatch(r'https?://www.wikidata.org/entity/Q[1-9]\d*', unit):
                ids.add(unit.rsplit('/', 1)[-1])
    return ids


def coverage(movies, rows, sources):
    by_movie = defaultdict(list)
    for row in rows:
        by_movie[row['subject']].append(row)
    props = {}
    for prop, (zh, en, kind) in PROPERTIES.items():
        matching = [r for r in rows if r['property'] == prop]
        values = [r for r in matching if r['snaktype'] == 'value' and r['value'] is not None]
        counts = Counter(r['subject'] for r in values)
        props[prop] = {'label_zh': zh, 'label_en': en, 'type': kind, 'movies_with_value': len(counts),
                       'coverage': len(counts) / len(movies), 'statements': len(matching),
                       'with_references': sum(bool(r['references']) for r in values),
                       'with_qualifiers': sum(bool(r['qualifiers']) for r in values),
                       'multi_value_movies': sum(n > 1 for n in counts.values()),
                       'unknown_snaks': sum(r['snaktype'] != 'value' for r in matching),
                       'without_qualifiers': sum(not r['qualifiers'] for r in values)}
    return {'movie_count': len(movies), 'fetched_movie_count': len(sources),
            'missing_movie_ids': sorted({m['id'] for m in movies} - sources.keys()),
            'statement_count': len(rows), 'properties': props,
            'known_release_year_count': sum(bool(m.get('release_year')) for m in movies),
            'country_distribution': dict(Counter(c for m in movies for c in m.get('base_facts', {}).get('P495', []))),
            'language_distribution': dict(Counter(c for m in movies for c in m.get('base_facts', {}).get('P364', []))),
            'note': 'Non-deprecated statements including preferred/normal and unknown snaks; multi-values are not automatically conflicts. Missing scope is not global scope.'}


def fetch_names(ids, cache_dir, proxy=None):
    """Use bounded official WDQS label queries after observed item-API 429s."""
    client = SparqlClient(cache_dir, proxy=proxy)
    names, provenance = {}, []
    for start in range(0, len(ids), 250):
        batch = ids[start:start+250]
        query = ('SELECT ?item ?en ?zh WHERE { VALUES ?item { ' + values(batch) +
                 ' } OPTIONAL { ?item rdfs:label ?en FILTER(LANG(?en)="en") } '
                 'OPTIONAL { ?item rdfs:label ?zh FILTER(LANG(?zh)="zh") } }')
        for row in client.query(query):
            qid = row['item']['value'].rsplit('/', 1)[-1]
            names[qid] = {'id': qid, 'label': row.get('en', {}).get('value', qid),
                          'label_zh': row.get('zh', {}).get('value'), 'aliases': [],
                          'label_source': 'cached_wikidata_sparql'}
        provenance.append(query)
        print(f'WDQS labels: {min(start+250,len(ids))}/{len(ids)}', flush=True)
    return names, provenance


def build(base_path, output_dir, count=200, proxy=None, offline=False, offline_labels=False):
    base_path, output_dir = Path(base_path), Path(output_dir)
    if (output_dir / 'manifest.json').exists():
        raise ValueError('Output is already frozen; choose a new output directory')
    base = read_json(base_path)
    if not 1 <= count <= len(base['movies']):
        raise ValueError('Count must be between 1 and original movie count')
    selected = select_pilot(base['movies'], count) if count < len(base['movies']) else [m['id'] for m in base['movies']]
    wanted = set(selected)
    cache_dir = ROOT / 'data/raw/catalog_v2_20261030/api_requests'
    cache_paths = [* (ROOT / 'data/raw').glob('entities_*.json'),
                   * (ROOT / 'data/raw/expansion_5000_20261029/api_requests').glob('*.json'),
                   *cache_dir.glob('*.json')]
    print(f'Reading cached claims for {count} movies...', flush=True)
    found, sources = cached_entities(cache_paths, wanted)
    missing = sorted(wanted - found.keys())
    client = EntityApiClient(cache_dir, proxy=proxy)
    if missing and not offline:
        client.entities(missing)
        extra, provenance = cached_entities(cache_dir.glob('*.json'), set(missing))
        found.update(extra)
        sources.update(provenance)
    if wanted - found.keys():
        write_json(output_dir / 'incomplete.json', {'missing_movie_ids': sorted(wanted - found.keys()), 'requested': count})
        raise ValueError('Not all selected movie claims available; progress saved; not freezing incomplete catalogue')
    print(f'Claims available: {len(found)}. Extracting qualified statements and labels...', flush=True)
    rows = [r for qid in selected for r in statements(found[qid], sources[qid])]
    labels = dict(base.get('entities', {}))
    needed = referenced_ids(rows) | wanted
    cached_labels, _ = cached_entities(cache_paths, needed, claims=False)
    for qid, entity in {**cached_labels, **found}.items():
        old = labels.get(qid, {})
        names = entity.get('labels', {})
        aliases = set(old.get('aliases', []))
        aliases.update(v['value'] for language in ('en', 'zh') for v in entity.get('aliases', {}).get(language, []))
        aliases.update(v['value'] for v in names.values())
        labels[qid] = {'id': qid, 'label': names.get('en', {}).get('value', old.get('label', qid)),
                       'label_zh': names.get('zh', {}).get('value', old.get('label_zh')),
                       'aliases': sorted(aliases), 'revision': entity.get('lastrevid')}
    name_queries = []
    for path in sorted((ROOT / 'data/raw/catalog_v2_20261030/name_queries').glob('*.json')):
        cached = read_json(path)
        query = cached.get('params', {}).get('query', '')
        if not query.startswith('SELECT ?item ?en ?zh WHERE'):
            continue
        name_queries.append(query)
        for row in cached.get('response', {}).get('results', {}).get('bindings', []):
            qid = row['item']['value'].rsplit('/', 1)[-1]
            if qid not in needed:
                continue
            old = labels.get(qid, {})
            labels[qid] = {**old, 'id': qid, 'label': row.get('en', {}).get('value', old.get('label', qid)),
                           'label_zh': row.get('zh', {}).get('value', old.get('label_zh')),
                           'aliases': old.get('aliases', []), 'label_source': 'cached_wikidata_sparql'}
    label_missing = sorted(q for q in needed if q not in labels or labels[q].get('label') == q)
    print(f'Entity names to fetch: {len(label_missing)}', flush=True)
    # Newly observed names are fetched once; dates/amounts never become fake entities.
    if label_missing and not offline and not offline_labels:
        fetched_names, fresh_queries = fetch_names(label_missing, ROOT / 'data/raw/catalog_v2_20261030/name_queries', proxy)
        labels.update(fetched_names)
        name_queries.extend(fresh_queries)
    for qid in needed:
        labels.setdefault(qid, {'id': qid, 'label': qid, 'label_zh': None, 'aliases': [], 'name_status': 'unresolved'})
    movies = []
    for movie in base['movies']:
        if movie['id'] not in wanted:
            continue
        qid = movie['id']
        movies.append({**labels[qid], 'base_facts': movie['facts'],
                       'release_year': release_year(found[qid]), 'source_revision': found[qid].get('lastrevid')})
    audit = coverage(movies, rows, sources)
    audit['unresolved_entity_labels'] = sorted(q for q in needed if q not in labels or labels[q].get('label') == q)
    write_json(output_dir / 'movies.json', movies)
    write_json(output_dir / 'entities.json', {q: labels[q] for q in sorted(needed) if q in labels})
    write_jsonl(output_dir / 'statements.jsonl', rows)
    write_json(output_dir / 'coverage.json', audit)
    write_json(output_dir / 'provenance.json', sources)
    files = {name: sha256(output_dir / name) for name in ('movies.json', 'entities.json', 'statements.jsonl', 'coverage.json', 'provenance.json')}
    write_json(output_dir / 'manifest.json', {'schema': 'movie-catalog-v2/1', 'created_at': utc_now(),
               'name': f'电影扩展事实 V2（{len(movies)}部）', 'movie_count': len(movies), 'statement_count': len(rows),
               'base_sha256': sha256(base_path), 'properties': list(PROPERTIES), 'files': files,
               'license': 'Wikidata structured data CC0; reference links do not confer redistribution rights to linked content',
               'source': 'Cached Wikidata Entity API statements; per-item revisions and actual retrieval timestamps in provenance.json',
               'label_queries': name_queries, 'label_query_source': 'https://query.wikidata.org/sparql; cached separately with actual retrieved_at',
               'unresolved_entity_label_count': len(audit['unresolved_entity_labels']),
               'sampling': 'full original 5000 movie IDs' if count == len(base['movies']) else 'stratified pilot, seed42; see selection algorithm'})
    print(f'Frozen {len(movies)} movies, {len(rows)} statements: {output_dir}', flush=True)
    return audit


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
    collect = sub.add_parser('build', help='Reuse qualified cached statements, fetching missing items if allowed.')
    collect.add_argument('--base', type=Path, default=BASE)
    collect.add_argument('--count', type=int, default=200)
    collect.add_argument('--output-dir', type=Path, default=DEST)
    collect.add_argument('--proxy')
    collect.add_argument('--offline', action='store_true')
    collect.add_argument('--offline-labels', action='store_true', help='Keep cached names/QIDs without remote label lookups; missing names are counted.')
    args = parser.parse_args(argv)
    if args.command == 'build':
        build(args.base, args.output_dir, args.count, args.proxy, args.offline, args.offline_labels)
        return 0
    base = read_json(args.base)
    ids = select_pilot(base['movies'], args.count)
    write_json(args.output, {'created_at': utc_now(), 'base_sha256': sha256(args.base),
                            'seed': 42, 'movie_ids': ids, 'count': len(ids),
                            'sampling': 'country/language/known-decade strata round-robin; unknown decade separate; not global random sample'})
    print(f'Selected {len(ids)} movies: {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
