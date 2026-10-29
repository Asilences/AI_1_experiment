import copy

import pytest

from kgqa import dataset, web
from kgqa.common import read_json, sha256, write_json


def inventory(root, manifest):
    for entry in manifest['files']:
        path = root / entry['path']
        entry.update(bytes=path.stat().st_size, sha256=sha256(path))
    manifest['totals'] = {'files': len(manifest['files']), 'bytes': sum(item['bytes'] for item in manifest['files'])}


@pytest.fixture
def bundle(tmp_path):
    data, model, snapshot = tmp_path / 'data/new', tmp_path / 'models/new', tmp_path / 'raw/snapshot.json'
    write_json(snapshot, {'movies': ['Q1']})
    write_json(data / 'entities.json', {'Q1': {'label': 'Movie'}})
    write_json(data / 'splits.json', {'train': ['Q1'], 'valid': [], 'test': []})
    (data / 'qa.jsonl').write_text('{"question_id":"q1"}\n', encoding='utf-8')
    (data / 'triples.tsv').write_text('Q1\tP57\tQ1\n', encoding='utf-8')
    write_json(data / 'manifest.json', {'snapshot_sha256': sha256(snapshot)})
    model.mkdir(parents=True)
    (model / 'model.pkl').write_bytes(b'opaque model: verifier never unpickles without --smoke')
    hashes = {name: sha256(data / name) for name in ('entities.json', 'splits.json', 'qa.jsonl')}
    write_json(model / 'config.json', {'input_hashes': hashes, 'model_sha256': sha256(model / 'model.pkl')})
    manifest = {'schema_version': 1, 'dataset_id': 'fixture',
                'entrypoints': {'data_dir': 'data/new', 'model_path': 'models/new/model.pkl', 'snapshot_path': 'raw/snapshot.json'},
                'counts': {'movies': 1, 'entities': 1, 'triples': 1, 'qa': 1},
                'files': [{'path': path.relative_to(tmp_path).as_posix()} for path in
                          [snapshot, *sorted(data.iterdir()), *sorted(model.iterdir())]]}
    inventory(tmp_path, manifest)
    path = tmp_path / 'release-manifest.json'
    write_json(path, manifest)
    return tmp_path, path, manifest


def test_verifies_complete_bundle_without_neural_dependencies_or_unpickling(bundle):
    root, path, manifest = bundle
    result = dataset.verify_manifest(path, root)
    assert result['status'] == 'PASS' and result['counts'] == manifest['counts']
    assert result['bytes'] == manifest['totals']['bytes'] and not result['smoke_results']


def test_activation_creates_profile_consumed_by_existing_ui(bundle, monkeypatch):
    root, path, _ = bundle
    dataset.activate(path, root)
    monkeypatch.setattr(web, 'ROOT', root)
    assert web.startup_paths() == (root / 'data/new', root / 'models/new/model.pkl')


def test_corrupt_file_refuses_activation_and_preserves_existing_profile(bundle):
    root, path, _ = bundle
    write_json(root / '.work/ui-dataset.json', {'previous': True})
    before = (root / '.work/ui-dataset.json').read_bytes()
    with (root / 'data/new/entities.json').open('a', encoding='utf-8') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        dataset.activate(path, root)
    assert (root / '.work/ui-dataset.json').read_bytes() == before


@pytest.mark.parametrize('unsafe', ['../outside.json', 'C:/outside.json', '\\\\server\\outside.json'])
def test_rejects_nonportable_or_escaping_paths(bundle, unsafe):
    root, path, manifest = bundle
    manifest['files'][0]['path'] = unsafe
    write_json(path, manifest)
    with pytest.raises(ValueError, match='paths must'):
        dataset.verify_manifest(path, root)


def test_duplicate_asset_is_rejected(bundle):
    root, path, manifest = bundle
    manifest['files'].append(copy.deepcopy(manifest['files'][0]))
    inventory(root, manifest)
    write_json(path, manifest)
    with pytest.raises(ValueError, match='Duplicate'):
        dataset.verify_manifest(path, root)


@pytest.mark.parametrize('component,error', [('model', 'Model and dataset'), ('snapshot', 'Snapshot and processed')])
def test_hash_valid_but_mismatched_component_is_rejected(bundle, component, error):
    root, path, manifest = bundle
    target = root / ('models/new/config.json' if component == 'model' else 'data/new/manifest.json')
    content = read_json(target)
    content['model_sha256' if component == 'model' else 'snapshot_sha256'] = '0' * 64
    write_json(target, content)
    inventory(root, manifest)
    write_json(path, manifest)
    with pytest.raises(ValueError, match=error):
        dataset.verify_manifest(path, root)


def test_missing_required_file_from_inventory_is_rejected(bundle):
    root, path, manifest = bundle
    manifest['files'] = [item for item in manifest['files'] if item['path'] != 'models/new/config.json']
    inventory(root, manifest)
    write_json(path, manifest)
    with pytest.raises(ValueError, match='required data and model'):
        dataset.verify_manifest(path, root)


def test_reported_scale_must_match_actual_data(bundle):
    root, path, manifest = bundle
    manifest['counts']['qa'] = 2
    write_json(path, manifest)
    with pytest.raises(ValueError, match='counts do not match'):
        dataset.verify_manifest(path, root)
