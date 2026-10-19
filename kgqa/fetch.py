"""Read-only access to Wikidata; every response is cached before processing."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import time
import urllib.parse
import urllib.request

from .common import ROOT, RELATIONS, read_json, setup_logging, sha256, utc_now, write_json

RAW = ROOT / 'data' / 'raw'


def request_json(url, params, cache_name):
    cache = RAW / cache_name
    if cache.exists():
        return read_json(cache)['response']
    full_url = url + '?' + urllib.parse.urlencode(params)
    for attempt in range(6):
        try:
            req = urllib.request.Request(full_url, headers={'User-Agent': 'AI1MovieKGQA/1.0 (educational experiment; low-rate cached queries)', 'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=65) as response:
                result = json.loads(response.read().decode('utf-8'))
            write_json(cache, {'retrieved_at': utc_now(), 'url': url, 'params': params, 'response': result})
            time.sleep(0.4)
            return result
        except Exception as error:
            if attempt == 5:
                raise
            delay = min(30, 2 ** (attempt + 1))
            if hasattr(error, 'headers'):
                try:
                    delay = max(delay, min(60, int(error.headers.get('Retry-After', delay))))
                except (ValueError, TypeError):
                    pass
            logging.warning('request retry %s/%s %s in %ss', attempt + 1, 6, error, delay)
            time.sleep(delay)


def candidates():
    collected = set()
    # A bounded candidate pool with equal query caps per release year; not a random sample of all Wikidata.
    for year in range(1980, 2025, 10):
        query = f'''SELECT DISTINCT ?film WHERE {{
          ?film wdt:P31 wd:Q11424; wdt:P577 ?date; wdt:P57 ?director.
          FILTER(?date >= "{year}-01-01T00:00:00Z"^^xsd:dateTime && ?date < "{min(year+10, 2025)}-01-01T00:00:00Z"^^xsd:dateTime)
        }} LIMIT 800'''
        response = request_json('https://query.wikidata.org/sparql', {'query': query, 'format': 'json'}, f'candidates_decade_{year}.json')
        ids = [row['film']['value'].rsplit('/', 1)[-1] for row in response['results']['bindings']]
        collected.update(ids)
        logging.info('Candidates year=%s returned=%s unique=%s', year, len(ids), len(collected))
    ids = sorted(collected, key=lambda x: int(x[1:]))
    random.Random(42).shuffle(ids)
    write_json(RAW / 'candidate_ids.json', ids)
    return ids


def get_entities(ids, claims=True):
    result = {}
    for offset in range(0, len(ids), 40):
        batch = ids[offset:offset + 40]
        mode = 'claims' if claims else 'labels'
        digest = hashlib.sha256((mode + '|' + '|'.join(batch)).encode()).hexdigest()[:16]
        payload = request_json('https://www.wikidata.org/w/api.php', {
            'action': 'wbgetentities', 'ids': '|'.join(batch), 'props': 'labels|aliases|claims|info' if claims else 'labels|aliases|info',
            'languages': 'en', 'format': 'json', 'maxlag': 5}, f'entities_{digest}.json')
        if 'error' in payload:
            raise RuntimeError(payload['error'])
        result.update(payload['entities'])
        logging.info('Entity batch %s/%s', min(offset + 40, len(ids)), len(ids))
    return result


def truthy(entity, relation):
    claims = [claim for claim in entity.get('claims', {}).get(relation, []) if claim.get('rank') != 'deprecated']
    if any(claim.get('rank') == 'preferred' for claim in claims):
        claims = [claim for claim in claims if claim.get('rank') == 'preferred']
    result = set()
    for claim in claims:
        snak = claim['mainsnak']
        if snak.get('snaktype') == 'value' and snak.get('datatype') == 'wikibase-item':
            value = snak['datavalue']['value']
            if 'id' in value:
                result.add(value['id'])
    return sorted(result, key=lambda x: int(x[1:]))


def compact(entity):
    return {'id': entity['id'], 'label': entity.get('labels', {}).get('en', {}).get('value', ''),
            'aliases': sorted({a['value'] for a in entity.get('aliases', {}).get('en', [])}),
            'revision': entity.get('lastrevid'),
            'facts': {rel: truthy(entity, rel) for rel in RELATIONS}}


def _candidate_ids(values):
    if not isinstance(values, list) or not values:
        raise ValueError('Candidate cache must contain a non-empty list of film IDs')
    if any(not isinstance(value, str) or not re.fullmatch(r'Q[1-9]\d*', value) for value in values):
        raise ValueError('Candidate cache contains a malformed Wikidata film ID')
    return list(dict.fromkeys(values))


def _cached_candidates(path):
    """Read a saved request_json envelope without contacting either Wikidata API."""
    path = Path(path)
    try:
        cached = read_json(path)
        bindings = cached['response']['results']['bindings']
        params = cached['params']
        query, retrieved_at = params['query'], cached['retrieved_at']
        if not isinstance(bindings, list) or not isinstance(params, dict):
            raise ValueError('Invalid response or params')
        if not isinstance(query, str) or not query.strip() or not isinstance(retrieved_at, str) or not retrieved_at.strip():
            raise ValueError('Missing source query or retrieval timestamp')
        values = []
        for binding in bindings:
            value = binding['film']['value']
            if not isinstance(value, str) or not re.fullmatch(r'https?://www\.wikidata\.org/entity/Q[1-9]\d*', value):
                raise ValueError('Invalid film entity URI')
            values.append(value.rsplit('/', 1)[-1])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'Malformed cached SPARQL candidate response: {path}') from error
    ids = sorted(_candidate_ids(values), key=lambda qid: int(qid[1:]))
    random.Random(42).shuffle(ids)
    provenance = {
        'source_path': str(path.resolve()), 'source_sha256': sha256(path),
        'source_url': cached.get('url'), 'query_params': params, 'query': query,
        'retrieved_at': retrieved_at, 'binding_count': len(bindings), 'unique_count': len(ids),
        'ordering': 'sorted by numeric Wikidata ID, then shuffled with seed 42', 'shuffle_seed': 42,
    }
    return ids, provenance


def run(target, candidate_cache=None):
    if not isinstance(target, int) or isinstance(target, bool) or target < 1:
        raise ValueError('target must be a positive integer')
    if candidate_cache is not None:
        ids, candidate_pool = _cached_candidates(candidate_cache)
        sampling = ('cached SPARQL candidate response; limited scope defined by source query; '
                    'not all Wikidata; sorted numeric IDs then shuffled seed 42; first eligible movies')
    else:
        id_cache = RAW / 'candidate_ids.json'
        existing = id_cache.exists()
        ids = _candidate_ids(read_json(id_cache) if existing else candidates())
        candidate_pool = {
            'source_path': str(id_cache.resolve()), 'source_sha256': sha256(id_cache),
            'unique_count': len(ids), 'ordering': 'cached ID list order retained',
            'source_query_provenance': 'not verified from the candidate ID list alone',
        }
        sampling = ('cached candidate ID list; original source-query scope not verified; not all Wikidata; '
                    'cached order retained; first eligible movies') if existing else (
                    'bounded SPARQL candidate pool for 1980-2024 release dates, up to 800 results per decade '
                    '(2020-2024 partial decade); numeric IDs shuffled seed 42; first eligible movies; not all Wikidata')
    setup_logging(f'fetch_{target}')
    selected, entity_map = [], {}
    offset = 0
    while offset < len(ids) and len(selected) < target:
        batch = ids[offset:offset + min(80, target - len(selected))]
        offset += len(batch)
        records = get_entities(batch)
        possible = []
        for qid in batch:
            entity = records.get(qid, {})
            if not entity.get('labels', {}).get('en'):
                continue
            row = compact(entity)
            if sum(1 <= len(values) <= 3 for values in row['facts'].values()) >= 3:
                possible.append(row)
        tail_ids = sorted({t for row in possible for values in row['facts'].values() for t in values}, key=lambda x: int(x[1:]))
        tail_data = get_entities([t for t in tail_ids if t not in entity_map], claims=False)
        for qid, entity in tail_data.items():
            if 'missing' not in entity:
                entity_map[qid] = compact(entity)
        for row in possible:
            valid = {r: vs for r, vs in row['facts'].items() if 1 <= len(vs) <= 3 and all(entity_map.get(t, {}).get('label') for t in vs)}
            if len(valid) >= 3:
                row['qa_relations'] = sorted(valid)
                selected.append(row)
                entity_map[row['id']] = row
            if len(selected) == target:
                break
        logging.info('Eligible movies=%s/%s', len(selected), target)
        if len(selected) >= target:
            break
    if len(selected) < target:
        raise RuntimeError(f'Only {len(selected)} eligible movies; requested {target}. Cached responses preserved.')
    needed = {row['id'] for row in selected} | {t for row in selected for vs in row['facts'].values() for t in vs}
    final = {'created_at': utc_now(), 'target': target, 'movies': selected,
             'entities': {qid: entity_map[qid] for qid in sorted(needed) if qid in entity_map},
             'sampling': sampling, 'candidate_pool': candidate_pool,
             'source': 'Wikidata, CC0, best-rank entity statements; English labels; query caps induce selection bias'}
    write_json(ROOT / 'data' / 'raw' / f'snapshot_{target}.json', final)
    logging.info('COMPLETE snapshot_%s.json movies=%s entities=%s', target, len(selected), len(final['entities']))
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=50)
    parser.add_argument('--candidate-cache', type=Path,
                        help='Use one saved raw SPARQL response; its query defines the dataset scope')
    args = parser.parse_args()
    run(args.target, candidate_cache=args.candidate_cache)
