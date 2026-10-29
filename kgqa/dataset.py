"""Verify and activate the complete, versioned movie dataset shipped with Git."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re

from .common import ROOT, read_json, read_jsonl, sha256, utc_now, write_json

DEFAULT_MANIFEST = Path('datasets/movies-5000-20261029/manifest.json')


def _inside(root, value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Manifest paths must be project-relative POSIX paths')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Manifest paths must stay inside the project')
    destination = (root / value).resolve()
    if not destination.is_relative_to(root):
        raise ValueError('Manifest paths must stay inside the project')
    return destination


def verify_manifest(manifest_path=DEFAULT_MANIFEST, root=None, smoke=False):
    root = Path(root or ROOT).resolve()
    path = Path(manifest_path)
    manifest = read_json(path if path.is_absolute() else root / path)
    if manifest.get('schema_version') != 1 or not manifest.get('files'):
        raise ValueError('Unsupported or empty dataset manifest')
    files, total = {}, 0
    for entry in manifest['files']:
        name = entry['path']
        destination = _inside(root, name)
        if name in files:
            raise ValueError(f'Duplicate manifest path: {name}')
        if not re.fullmatch('[0-9a-f]{64}', entry['sha256']):
            raise ValueError(f'Invalid SHA-256: {name}')
        if not isinstance(entry['bytes'], int) or isinstance(entry['bytes'], bool) or entry['bytes'] < 0:
            raise ValueError(f'Invalid byte size: {name}')
        if destination.stat().st_size != entry['bytes'] or sha256(destination) != entry['sha256']:
            raise ValueError(f'File size or SHA-256 mismatch: {name}')
        files[name] = entry
        total += entry['bytes']
    if manifest['totals'] != {'files': len(files), 'bytes': total}:
        raise ValueError('Manifest file totals do not match')

    data = _inside(root, manifest['entrypoints']['data_dir'])
    model = _inside(root, manifest['entrypoints']['model_path'])
    snapshot = _inside(root, manifest['entrypoints']['snapshot_path'])
    required = [data / name for name in ('entities.json', 'splits.json', 'qa.jsonl', 'triples.tsv', 'manifest.json')]
    required += [model, model.parent / 'config.json', snapshot]
    if any(item.relative_to(root).as_posix() not in files for item in required):
        raise ValueError('Manifest must cover all required data and model files')
    if read_json(data / 'manifest.json')['snapshot_sha256'] != files[snapshot.relative_to(root).as_posix()]['sha256']:
        raise ValueError('Snapshot and processed dataset pairing mismatch')
    config = read_json(model.parent / 'config.json')
    hashes = {name: files[(data / name).relative_to(root).as_posix()]['sha256']
              for name in ('entities.json', 'splits.json', 'qa.jsonl')}
    model_hash = files[model.relative_to(root).as_posix()]['sha256']
    if config['input_hashes'] != hashes or config['model_sha256'] != model_hash:
        raise ValueError('Model and dataset pairing mismatch')
    entities, splits = read_json(data / 'entities.json'), read_json(data / 'splits.json')
    movies = sum(splits.values(), [])
    if len(movies) != len(set(movies)):
        raise ValueError('Movie splits overlap')
    rows = read_jsonl(data / 'qa.jsonl')
    with (data / 'triples.tsv').open(encoding='utf-8') as stream:
        triples = sum(bool(line.strip()) for line in stream)
    counts = {'movies': len(movies), 'entities': len(entities), 'triples': triples, 'qa': len(rows)}
    if counts != manifest['counts']:
        raise ValueError('Dataset counts do not match the manifest')
    smoke_results = []
    if smoke:
        from .pipeline import PipelineQA, load_model
        classifier, threshold = load_model(model)
        system = PipelineQA(entities, movies, classifier, threshold)
        if not manifest.get('smoke_queries'):
            raise ValueError('Manifest has no smoke queries')
        for case in manifest['smoke_queries']:
            prediction = system.predict(case['question'])
            if (prediction['movie_id'] != case['movie_id'] or prediction['relation_id'] != case['relation_id']
                    or set(prediction['answer_ids']) != set(case['answer_ids'])):
                raise ValueError(f"Smoke question failed: {case['question']}")
            smoke_results.append({'question': case['question'], 'answer': prediction['answer']})
    return {'dataset_id': manifest['dataset_id'], 'files': len(files), 'bytes': total,
            'counts': counts, 'input_hashes': hashes, 'model_sha256': model_hash,
            'data_dir': data.relative_to(root).as_posix(), 'model_path': model.relative_to(root).as_posix(),
            'smoke_results': smoke_results, 'status': 'PASS'}


def activate(manifest_path=DEFAULT_MANIFEST, root=None, smoke=False):
    root = Path(root or ROOT).resolve()
    result = verify_manifest(manifest_path, root, smoke)
    write_json(root / '.work/ui-dataset.json', {
        'schema_version': 1, 'activated_at': utc_now(), 'dataset_id': result['dataset_id'],
        **{name: result[name] for name in ('data_dir', 'model_path', 'input_hashes', 'model_sha256')},
        'note': 'Published expanded A0 dataset. Original experiment paths remain available via --legacy.'})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'activate'))
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--smoke', action='store_true', help='Load the paired A0 model and check recorded example answers.')
    args = parser.parse_args(argv)
    try:
        result = (activate if args.action == 'activate' else verify_manifest)(args.manifest, smoke=args.smoke)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
