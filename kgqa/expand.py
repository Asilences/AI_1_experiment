"""Extend a frozen movie snapshot using cached Wikidata truthy SPARQL facts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .common import RELATIONS, ROOT, read_json, sha256, utc_now, write_json
from .linker import normalize
from .fetch import compact, truthy

ENDPOINT = 'https://query.wikidata.org/sparql'
FILM_QUERY = '''SELECT DISTINCT ?film ?links WHERE {
  ?film wdt:P31 wd:Q11424; wdt:P57 ?director; wikibase:sitelinks ?links.
  FILTER(?links >= 15)
} ORDER BY DESC(?links) ASC(?film) LIMIT 8500'''


def entity_id(uri):
    if not isinstance(uri, str) or not re.fullmatch(r'https?://www\.wikidata\.org/entity/Q[1-9]\d*', uri):
        raise ValueError('Invalid Wikidata entity URI')
    return uri.rsplit('/', 1)[-1]


def values(ids):
    if any(not re.fullmatch(r'Q[1-9]\d*', qid) for qid in ids):
        raise ValueError('Invalid Wikidata item ID')
    return ' '.join('wd:' + qid for qid in ids)


class SparqlClient:
    def __init__(self, cache_dir, proxy=None):
        self.cache_dir = Path(cache_dir)
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({'https': proxy, 'http': proxy}) if proxy else urllib.request.ProxyHandler())

    def query(self, query):
        digest = hashlib.sha256(query.encode('utf-8')).hexdigest()
        cache = self.cache_dir / (digest + '.json')
        if cache.exists():
            document = read_json(cache)
            if document['params']['query'] != query:
                raise ValueError('Cached query does not match its key')
            return document['response']['results']['bindings']
        url = ENDPOINT + '?' + urllib.parse.urlencode({'query': query, 'format': 'json'})
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={
                    'User-Agent': 'AI1MovieKGQA/2.0 (educational research; low-rate cached requests)',
                    'Accept': 'application/sparql-results+json',
                })
                with self.opener.open(request, timeout=55) as response:
                    payload = json.loads(response.read().decode('utf-8'))
                rows = payload['results']['bindings']
                write_json(cache, {'retrieved_at': utc_now(), 'url': ENDPOINT,
                                   'params': {'query': query, 'format': 'json'}, 'response': payload})
                time.sleep(.6)
                return rows
            except (OSError, ValueError, KeyError) as error:
                if attempt == 3:
                    raise
                delay = 3 * (attempt + 1)
                if isinstance(error, urllib.error.HTTPError):
                    retry = error.headers.get('Retry-After', '')
                    if retry.isdigit():
                        delay = max(delay, min(60, int(retry)))
                print(f'Wikidata retry {attempt + 1}/3: {type(error).__name__} {getattr(error, "code", "")}; wait {delay}s', flush=True)
                time.sleep(delay)
        raise RuntimeError('Unreachable retry state')


class EntityApiClient:
    """The official item API is independent of WDQS and includes item revisions."""
    def __init__(self, cache_dir, proxy=None):
        self.cache_dir = Path(cache_dir)
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({'https': proxy, 'http': proxy}) if proxy else urllib.request.ProxyHandler())

    def entities(self, ids, claims=True):
        result = {}
        for start in range(0, len(ids), 40):
            batch = ids[start:start + 40]
            values(batch)
            params = {'action': 'wbgetentities', 'ids': '|'.join(batch),
                      'props': 'labels|aliases|claims|info' if claims else 'labels|aliases|info',
                      'languages': 'en|zh', 'format': 'json', 'maxlag': 5}
            digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
            cache = self.cache_dir / (digest + '.json')
            if cache.exists():
                document = read_json(cache)
                if document['params'] != params:
                    raise ValueError('Cached item request does not match its key')
                payload = document['response']
            else:
                url = 'https://www.wikidata.org/w/api.php'
                for attempt in range(5):
                    try:
                        request = urllib.request.Request(url + '?' + urllib.parse.urlencode(params), headers={
                            'User-Agent': 'AI1MovieKGQA/2.0 (educational research; low-rate cached requests)',
                            'Accept': 'application/json',
                        })
                        with self.opener.open(request, timeout=55) as response:
                            payload = json.loads(response.read().decode('utf-8'))
                        if payload.get('error'):
                            raise RuntimeError(str(payload['error'].get('code', 'Wikidata error')))
                        if 'entities' not in payload:
                            raise ValueError('Missing entities in API response')
                        write_json(cache, {'retrieved_at': utc_now(), 'url': url, 'params': params, 'response': payload})
                        time.sleep(.6)
                        break
                    except (OSError, ValueError, RuntimeError) as error:
                        if attempt == 4:
                            raise
                        delay = 3 * (attempt + 1)
                        if isinstance(error, urllib.error.HTTPError):
                            retry = error.headers.get('Retry-After', '')
                            if retry.isdigit():
                                delay = max(delay, min(60, int(retry)))
                        print(f'Item API retry {attempt + 1}/4: {type(error).__name__} {getattr(error, "code", "")}; wait {delay}s', flush=True)
                        time.sleep(delay)
            result.update(payload['entities'])
            print(f'Item API {"facts" if claims else "labels"}: {min(start + 40, len(ids))}/{len(ids)}', flush=True)
        return result


def release_year(entity):
    statements = [row for row in entity.get('claims', {}).get('P577', []) if row.get('rank') != 'deprecated']
    if any(row.get('rank') == 'preferred' for row in statements):
        statements = [row for row in statements if row.get('rank') == 'preferred']
    years = []
    for row in statements:
        snak = row.get('mainsnak', {})
        date = snak.get('datavalue', {}).get('value', {})
        matched = re.match(r'^\+?(\d{4})-', date.get('time', '') if isinstance(date, dict) else '')
        if matched:
            years.append(int(matched[1]))
    return min(years) if years else None


def fetch_api_batch(client, ids, known_entities):
    heads = client.entities(ids)
    rows, rejected = [], Counter()
    for qid in ids:
        entity = heads.get(qid, {})
        if entity.get('id') != qid or not entity.get('labels', {}).get('en', {}).get('value'):
            rejected['missing_or_redirected_english_item'] += 1
            continue
        # Direct P31 films are required by the frozen candidate query; recheck
        # the live claims so a changed/redirected candidate cannot become a film.
        if 'Q11424' not in truthy(entity, 'P31'):
            rejected['no_longer_direct_instance_of_film'] += 1
            continue
        movie = compact(entity)
        aliases = set(movie['aliases'])
        aliases.update(a['value'] for a in entity.get('aliases', {}).get('zh', []))
        chinese = entity.get('labels', {}).get('zh', {}).get('value')
        if chinese:
            aliases.add(chinese)
        movie.update(aliases=sorted(aliases), label_zh=chinese, release_year=release_year(entity),
                     source_kind='cached_wikidata_item_api')
        rows.append(movie)
    tail_ids = sorted({tail for movie in rows for tails in movie['facts'].values() for tail in tails
                       if tail not in known_entities}, key=lambda qid: int(qid[1:]))
    tails = client.entities(tail_ids, claims=False) if tail_ids else {}
    additions = {qid: compact(entity) for qid, entity in tails.items()
                 if entity.get('id') == qid and entity.get('labels', {}).get('en', {}).get('value')}
    names = {**known_entities, **additions}
    movies = []
    for movie in rows:
        qa_relations = [r for r, answers in movie['facts'].items()
                        if 1 <= len(answers) <= 3 and all(names.get(qid, {}).get('label') for qid in answers)]
        if len(qa_relations) < 3:
            rejected['fewer_than_three_complete_qa_relations'] += 1
            continue
        movie['facts'] = {r: [qid for qid in answers if names.get(qid, {}).get('label')]
                          for r, answers in movie['facts'].items()}
        movie['qa_relations'] = sorted(qa_relations)
        movies.append(movie)
        additions[movie['id']] = movie
    return movies, additions, rejected


def fetch_batch(client, ids, known_entities):
    properties = ' '.join('wdt:' + relation for relation in RELATIONS)
    facts = {qid: {r: [] for r in RELATIONS} for qid in ids}
    rows = client.query(f'''SELECT DISTINCT ?film ?property ?value WHERE {{
      VALUES ?film {{ {values(ids)} }} VALUES ?property {{ {properties} }}
      ?film ?property ?value. FILTER(isIRI(?value))
    }}''')
    for row in rows:
        qid, tail = entity_id(row['film']['value']), entity_id(row['value']['value'])
        relation = row['property']['value'].rsplit('/', 1)[-1]
        if qid in facts and relation in RELATIONS:
            facts[qid][relation].append(tail)
    needed = sorted(set(ids) | {tail for f in facts.values() for tails in f.values() for tail in tails})
    labels = {qid: entity['label'] for qid, entity in known_entities.items() if entity.get('label')}
    chinese = {}
    missing = [qid for qid in needed if qid not in labels]
    for start in range(0, len(missing), 200):
        for row in client.query(f'''SELECT ?item ?label WHERE {{
          VALUES ?item {{ {values(missing[start:start + 200])} }}
          ?item rdfs:label ?label. FILTER(LANG(?label) IN ("en", "zh"))
        }}'''):
            qid = entity_id(row['item']['value'])
            label = row['label']
            if label.get('xml:lang') == 'en':
                labels[qid] = label['value']
            elif label.get('xml:lang') == 'zh':
                chinese[qid] = label['value']
    aliases = defaultdict(set)
    for row in client.query(f'''SELECT ?item ?alias WHERE {{
      VALUES ?item {{ {values(ids)} }} ?item skos:altLabel ?alias.
      FILTER(LANG(?alias) IN ("en", "zh"))
    }}'''):
        aliases[entity_id(row['item']['value'])].add(row['alias']['value'])
    years = {}
    for row in client.query(f'''SELECT ?film (MIN(?date) AS ?firstDate) WHERE {{
      VALUES ?film {{ {values(ids)} }} ?film wdt:P577 ?date.
    }} GROUP BY ?film'''):
        matched = re.match(r'^\+?(\d{4})-', row.get('firstDate', {}).get('value', ''))
        if matched:
            years[entity_id(row['film']['value'])] = int(matched[1])
    additions = {qid: {'id': qid, 'label': labels[qid], 'aliases': [], 'facts': {}, 'revision': None}
                 for qid in needed if qid in labels and qid not in known_entities}
    movies, rejected = [], Counter()
    for qid in ids:
        if qid not in labels:
            rejected['missing_english_label'] += 1
            continue
        full_facts = {r: sorted(set(tails), key=lambda x: int(x[1:])) for r, tails in facts[qid].items()}
        qa_relations = [r for r, tails in full_facts.items()
                        if 1 <= len(tails) <= 3 and all(tail in labels for tail in tails)]
        if len(qa_relations) < 3:
            rejected['fewer_than_three_complete_qa_relations'] += 1
            continue
        if qid in chinese:
            aliases[qid].add(chinese[qid])
        movie = {'id': qid, 'label': labels[qid], 'aliases': sorted(aliases[qid]),
                 'release_year': years.get(qid), 'label_zh': chinese.get(qid), 'revision': None,
                 'facts': {r: [tail for tail in tails if tail in labels] for r, tails in full_facts.items()},
                 'qa_relations': sorted(qa_relations), 'source_kind': 'cached_wdqs_truthy'}
        movies.append(movie)
        additions[qid] = movie
    return movies, additions, rejected


def disambiguate_titles(movies, entities):
    """Keep official labels, but provide unique title/year aliases for questions."""
    duplicate_labels = Counter(normalize(movie['label']) for movie in movies)
    display_counts = Counter()
    name_owners = defaultdict(set)
    for movie in movies:
        title = movie['label']
        year = movie.get('release_year')
        display = f'{title} ({year})' if (duplicate_labels[normalize(title)] > 1 or len(normalize(title)) < 2) and year else title
        movie['display_label'] = display
        display_counts[normalize(display)] += 1
        names = [movie['label'], *movie.get('aliases', [])]
        if year:
            names.append(f'{title} ({year})')
        for name in names:
            if normalize(name):
                name_owners[normalize(name)].add(movie['id'])
    for movie in movies:
        display = movie['display_label']
        ambiguous_without_year = duplicate_labels[normalize(movie['label'])] > 1 and not movie.get('release_year')
        question_title = (f"{display} ({movie['id']})"
                          if display_counts[normalize(display)] > 1 or ambiguous_without_year else display)
        if name_owners.get(normalize(question_title), {movie['id']}) != {movie['id']}:
            year_title = f"{movie['label']} ({movie['release_year']})" if movie.get('release_year') else None
            question_title = (year_title if year_title and name_owners.get(normalize(year_title)) == {movie['id']}
                              else f"{display} ({movie['id']})")
        if len(normalize(question_title)) < 2:
            question_title = f"{display} ({movie['id']})"
        movie['question_title'] = question_title
        aliases = set(movie.get('aliases', []))
        if movie.get('release_year'):
            aliases.add(f"{movie['label']} ({movie['release_year']})")
        if question_title != movie['label']:
            aliases.add(question_title)
        movie['aliases'] = sorted(aliases)
        entities[movie['id']] = movie


def collect(base_snapshot, candidate_cache, work_dir, target=5000, batch_size=150, proxy=None, client=None, backend='api'):
    base_snapshot, candidate_cache, work_dir = map(Path, (base_snapshot, candidate_cache, work_dir))
    base = read_json(base_snapshot)
    if not isinstance(target, int) or target <= len(base['movies']):
        raise ValueError('Expansion target must exceed the base movie count')
    if not 1 <= batch_size <= 200:
        raise ValueError('batch_size must be between 1 and 200')
    candidate_document = read_json(candidate_cache)
    candidate_ids = list(dict.fromkeys(entity_id(row['film']['value'])
                                      for row in candidate_document['response']['results']['bindings']))
    if not candidate_document.get('retrieved_at') or not candidate_document.get('params', {}).get('query'):
        raise ValueError('Candidate query and retrieval timestamp are required')
    if backend not in {'api', 'sparql'}:
        raise ValueError('Unsupported data backend')
    identity = {'base_sha256': sha256(base_snapshot), 'candidates_sha256': sha256(candidate_cache), 'target': target, 'backend': backend}
    work_dir.mkdir(parents=True, exist_ok=True)
    destination = work_dir / f'snapshot_{target}.json'
    if destination.exists():
        snapshot = read_json(destination)
        if snapshot.get('expansion_identity') != identity or len(snapshot['movies']) != target:
            raise ValueError('Existing snapshot does not match this expansion request')
        return destination
    progress_path = work_dir / f'progress_{target}.json'
    base_ids = {movie['id'] for movie in base['movies']}
    candidates = [qid for qid in candidate_ids if qid not in base_ids]
    if progress_path.exists():
        progress = read_json(progress_path)
        if progress['identity'] != identity:
            raise ValueError('Resume input hashes changed; use a new work directory')
        movies, entities = progress['movies'], progress['entities']
        cursor, rejected = progress['cursor'], Counter(progress['rejected'])
    else:
        movies, entities = base['movies'], base['entities']
        # Remove historical dangling/unlabelled facts only in the new version.
        for movie in movies:
            movie['facts'] = {r: sorted({tail for tail in tails if entities.get(tail, {}).get('label')},
                                        key=lambda x: int(x[1:])) for r, tails in movie['facts'].items()}
            entities[movie['id']] = movie
        cursor, rejected = 0, Counter()
    client = client or (EntityApiClient(work_dir / 'api_requests', proxy) if backend == 'api'
                        else SparqlClient(work_dir / 'requests', proxy))
    while len(movies) < target and cursor < len(candidates):
        ids = candidates[cursor:cursor + min(batch_size, target - len(movies))]
        added, records, skipped = (fetch_api_batch if backend == 'api' else fetch_batch)(client, ids, entities)
        entities.update(records)
        movies.extend(added)
        rejected.update(skipped)
        cursor += len(ids)
        write_json(progress_path, {'identity': identity, 'cursor': cursor, 'movies': movies,
                                   'entities': entities, 'rejected': dict(rejected), 'updated_at': utc_now()})
        print(f'Collected {len(movies)}/{target} movies; scanned {cursor}/{len(candidates)} new candidates; rejected {sum(rejected.values())}', flush=True)
    if len(movies) != target:
        raise RuntimeError(f'Only {len(movies)} eligible movies; checkpoint retained. Supply a larger candidate pool in a new version.')
    disambiguate_titles(movies, entities)
    needed = {m['id'] for m in movies} | {tail for m in movies for tails in m['facts'].values() for tail in tails}
    snapshot = {'created_at': utc_now(), 'target': target, 'movies': movies,
                'entities': {qid: entities[qid] for qid in sorted(needed)}, 'expansion_identity': identity,
                'base': {'snapshot': base_snapshot.as_posix(), 'sha256': identity['base_sha256'],
                         'movies': len(base_ids), 'created_at': base.get('created_at')},
                'candidate_source': {'path': candidate_cache.as_posix(), 'sha256': identity['candidates_sha256'],
                                     'query': candidate_document['params']['query'],
                                     'retrieved_at': candidate_document['retrieved_at'], 'count': len(candidate_ids)},
                'source': f'Wikidata CC0; preserved base snapshot plus cached {backend} best-rank statements and English labels.',
                'sampling': 'Keep all base movies; append first eligible candidates ordered by descending sitelinks. Popularity-biased, not a representative random sample.',
                'revision_note': 'Item API records include item revision IDs. WDQS records have no item revision IDs. Exact requests, responses and retrieval times are cached.',
                'release_year_note': 'Minimum P577 date among truthy statements, when available; not an independently verified premiere year.',
                'rejected': dict(rejected)}
    write_json(destination, snapshot)
    print(f'Frozen snapshot: {destination}; SHA256={sha256(destination)}', flush=True)
    return destination


def prepare_and_train(snapshot_path, output_dir, model_dir, base_splits=None, activate=False):
    """Build an isolated dataset/model, evaluate it, then optionally select it for UI."""
    from . import prepare, pipeline
    from .common import read_jsonl, write_jsonl
    from .linker import EntityLinker
    from .metrics import score_rows
    from .personal_experiment import validate_inputs

    snapshot_path, output_dir, model_dir = map(Path, (snapshot_path, output_dir, model_dir))
    manifest_path = output_dir / 'manifest.json'
    if manifest_path.exists():
        if read_json(manifest_path)['snapshot_sha256'] != sha256(snapshot_path):
            raise ValueError('Existing processed dataset belongs to a different snapshot')
    else:
        prepare.build(snapshot_path, output_dir, base_splits)
        manifest = read_json(manifest_path)
        manifest.update(name=f"扩展电影库 · {manifest['movies']} 部", dataset_role='expanded UI knowledge base; not an independent held-out research dataset',
                        prepare_source_sha256=sha256(Path(prepare.__file__)))
        write_json(manifest_path, manifest)
    entities, splits = read_json(output_dir / 'entities.json'), read_json(output_dir / 'splits.json')
    qa = read_jsonl(output_dir / 'qa.jsonl')
    grouped = validate_inputs(qa, entities, splits)
    hashes = {name: sha256(output_dir / name) for name in ('qa.jsonl', 'entities.json', 'splits.json')}
    linker = EntityLinker(sum(splits.values(), []), entities)
    for qid in sum(splits.values(), []):
        question_title = entities[qid].get('question_title', entities[qid]['label'])
        if linker.link('Who directed ' + question_title + '?')[0] != qid:
            raise ValueError(f'Movie question title remains ambiguous: {qid} {question_title}')
    model_path = model_dir / 'model.pkl'
    if model_path.exists():
        config = read_json(model_dir / 'config.json')
        if config['input_hashes'] != hashes or config['model_sha256'] != sha256(model_path):
            raise ValueError('Existing model is not paired with this dataset')
    else:
        print('Training the isolated A0 classifier; original models remain untouched.', flush=True)
        pipeline.train(output_dir, model_dir)
    classifier, threshold = pipeline.load_model(model_path)
    system = pipeline.PipelineQA(entities, sum(splits.values(), []), classifier, threshold)
    predictions = []
    for index, row in enumerate(grouped['test'], 1):
        prediction = system.predict(row['question'])
        prediction['question_id'] = row['question_id']
        predictions.append(prediction)
        if index % 1000 == 0:
            print(f'Evaluated {index}/{len(grouped["test"])} expanded test questions', flush=True)
    metrics = score_rows(grouped['test'], predictions, entities)
    metrics.update(prediction_coverage=100.0, scope='Shared-template questions on a known frozen knowledge base; not free-form generalization.',
                   model_sha256=sha256(model_path), input_hashes=hashes, evaluated_at=utc_now())
    write_json(model_dir / 'metrics_test.json', metrics)
    write_jsonl(model_dir / 'predictions_test.jsonl', predictions)
    if activate:
        root = ROOT.resolve()
        data_relative = output_dir.resolve().relative_to(root).as_posix()
        model_relative = model_path.resolve().relative_to(root).as_posix()
        write_json(root / '.work/ui-dataset.json', {
            'schema_version': 1, 'activated_at': utc_now(), 'data_dir': data_relative,
            'model_path': model_relative, 'input_hashes': hashes, 'model_sha256': sha256(model_path),
            'note': 'Expanded A0 only. Original data, TransE and neural models remain available via --legacy.',
        })
    print(f'Expanded A0 ready: movies={len(sum(splits.values(), []))}; test={metrics["count"]}; F1={metrics["entity_macro_f1"]:.4f}', flush=True)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-snapshot', type=Path, default=ROOT / 'data/raw/snapshot_1000.json')
    parser.add_argument('--target', type=int, default=5000)
    parser.add_argument('--candidate-cache', type=Path)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=150)
    parser.add_argument('--proxy', help='Optional HTTP proxy, e.g. http://127.0.0.1:7890')
    parser.add_argument('--backend', choices=['api', 'sparql'], default='api')
    parser.add_argument('--output-dir', type=Path, help='New isolated processed-data directory.')
    parser.add_argument('--model-dir', type=Path, help='New isolated A0 model directory; requires --output-dir.')
    parser.add_argument('--base-splits', type=Path, help='Optional original splits.json to preserve film assignments.')
    parser.add_argument('--activate', action='store_true', help='Use the completed paired A0 dataset/model in the browser UI.')
    args = parser.parse_args(argv)
    if args.model_dir and not args.output_dir or args.activate and not args.model_dir:
        parser.error('--model-dir requires --output-dir; --activate also requires --model-dir')
    candidate_cache = args.candidate_cache or args.work_dir / 'candidates.json'
    if not candidate_cache.exists():
        rows = SparqlClient(args.work_dir / 'requests', args.proxy).query(FILM_QUERY)
        write_json(candidate_cache, {'retrieved_at': utc_now(), 'url': ENDPOINT,
                                     'params': {'query': FILM_QUERY, 'format': 'json'},
                                     'response': {'results': {'bindings': rows}}})
    try:
        snapshot = collect(args.base_snapshot, candidate_cache, args.work_dir, args.target, args.batch_size, args.proxy, backend=args.backend)
        if args.output_dir:
            if args.model_dir:
                prepare_and_train(snapshot, args.output_dir, args.model_dir, args.base_splits, args.activate)
            else:
                from .prepare import build
                build(snapshot, args.output_dir, args.base_splits)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f'Expansion stopped: {error}. Cached responses and progress are retained.\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
