import hashlib
from pathlib import Path
from zipfile import ZipFile

import pytest

from kgqa.submission_assets import digest, install, safe_target


@pytest.fixture
def bundle(tmp_path):
    path = tmp_path / 'fixture.zip'
    content = b'fixed synthetic data'
    with ZipFile(path, 'w') as stream:
        stream.writestr('data/frozen/test.json', content)
    asset = {'bytes': path.stat().st_size, 'sha256': digest(path),
             'files': {'data/frozen/test.json': {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}}}
    return path, asset, tmp_path / 'project'


def test_restore_is_verified_and_identical_assets_can_be_reused(bundle):
    path, asset, root = bundle
    assert install(path, root, asset) == 1
    assert install(path, root, asset) == 1
    assert (root / 'data/frozen/test.json').read_bytes() == b'fixed synthetic data'


def test_restore_refuses_changed_existing_data_and_corrupt_download(bundle):
    path, asset, root = bundle
    install(path, root, asset)
    target = root / 'data/frozen/test.json'
    target.write_bytes(b'other existing dataset')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        install(path, root, asset)
    assert target.read_bytes() == b'other existing dataset'
    path.write_bytes(b'broken archive')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        install(path, root, asset)


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'C:/absolute', 'a\\..\\outside'])
def test_asset_paths_cannot_escape_project(tmp_path, name):
    with pytest.raises(ValueError):
        safe_target(tmp_path, name)


def test_archive_members_must_match_manifest_before_any_extraction(bundle):
    path, asset, root = bundle
    with ZipFile(path, 'a') as stream:
        stream.writestr('../escaped', b'no')
    asset.update(bytes=path.stat().st_size, sha256=digest(path))
    with pytest.raises(ValueError, match='file list'):
        install(path, root, asset)
    assert not root.exists()
