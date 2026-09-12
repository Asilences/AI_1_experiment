"""Read-only access to Wikidata; every response is cached before processing."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import time
import urllib.parse
import urllib.request

from .common import ROOT, RELATIONS, read_json, setup_logging, utc_now, write_json

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


def run(target):
    setup_logging(f'fetch_{target}')
    ids = read_json(RAW / 'candidate_ids.json') if (RAW / 'candidate_ids.json').exists() else candidates()
    selected, entity_map = [], {}
    for offset in range(0, len(ids), 80):
        records = get_entities(ids[offset:offset + 80])
        possible = []
        for qid in ids[offset:offset + 80]:
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
             'sampling': '1980-2024 release years, up to 800 query results per decade (2020-2024 partial decade); shuffled seed 42; first eligible movies',
             'source': 'Wikidata, CC0, best-rank entity statements; English labels; query caps induce selection bias'}
    write_json(ROOT / 'data' / 'raw' / f'snapshot_{target}.json', final)
    logging.info('COMPLETE snapshot_%s.json movies=%s entities=%s', target, len(selected), len(final['entities']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=50)
    run(parser.parse_args().target)
