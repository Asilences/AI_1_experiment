"""按公开manifest下载、校验并恢复冻结复现资产，不覆盖不同版本的文件。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from urllib.request import ProxyHandler, build_opener
from zipfile import ZipFile

from .common import ROOT


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def safe_target(root, name):
    if not isinstance(name, str) or '\\' in name or ':' in name:
        raise ValueError('Invalid manifest path')
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Path escapes project')
    root = Path(root).resolve()
    target = root.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError('Path escapes project')
    return target


def verify_file(path, entry):
    if not Path(path).is_file() or Path(path).stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
        raise ValueError(f'Checksum mismatch or missing file: {path}')


def verify(root, asset):
    for name, entry in asset['files'].items():
        verify_file(safe_target(root, name), entry)
    return len(asset['files'])


def install(archive, root, asset):
    verify_file(archive, asset)
    with ZipFile(archive) as bundle:
        names = [entry.filename for entry in bundle.infolist()]
        if len(names) != len(set(names)) or set(names) != set(asset['files']):
            raise ValueError('Archive file list differs from manifest')
        # Validate every destination and conflict before creating anything.
        for name in names:
            target = safe_target(root, name)
            if (bundle.getinfo(name).external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive symlinks are not accepted')
            if target.exists():
                verify_file(target, asset['files'][name])
        for name in names:
            target = safe_target(root, name)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.asset-partial')
            try:
                with bundle.open(name) as source, temporary.open('xb') as output:
                    shutil.copyfileobj(source, output, 4 * 1024 * 1024)
                verify_file(temporary, asset['files'][name])
                temporary.replace(target)
            except Exception:
                if temporary.exists():
                    temporary.unlink()
                raise
    return verify(root, asset)


def download(asset, destination, proxy=None):
    path = Path(destination)
    if path.exists():
        verify_file(path, asset)
        return path
    if not asset['url'].startswith('https://github.com/Asilences/AI_1_experiment/releases/download/'):
        raise ValueError('Unexpected release host or repository')
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = build_opener(ProxyHandler({'http': proxy, 'https': proxy}) if proxy else ProxyHandler())
    temporary = path.with_suffix(path.suffix + '.partial')
    with opener.open(asset['url'], timeout=120) as source, temporary.open('wb') as output:
        shutil.copyfileobj(source, output, 4 * 1024 * 1024)
    verify_file(temporary, asset)
    temporary.replace(path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'download', 'install'])
    parser.add_argument('--manifest', type=Path, default=ROOT / 'configs/submission-assets-20261031.json')
    parser.add_argument('--asset', choices=['catalog_v2', 'freeform_v2'], required=True)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--cache-dir', type=Path, default=ROOT / '.work/submission-assets')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--proxy')
    args = parser.parse_args(argv)
    asset = json.loads(args.manifest.read_text(encoding='utf-8'))['assets'][args.asset]
    try:
        if args.action == 'verify':
            count = verify(args.project_root, asset)
        else:
            archive = args.archive or download(asset, args.cache_dir / asset['filename'], args.proxy)
            if args.action == 'download':
                verify_file(archive, asset)
                print(archive)
                return 0
            count = install(archive, args.project_root, asset)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f'PASS {args.asset}: {count} frozen files verified')
    return 0


if __name__ == '__main__':
    main()
