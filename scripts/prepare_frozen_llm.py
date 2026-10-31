"""Download a pinned frozen model into an explicitly selected asset directory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests

MODEL = 'Qwen/Qwen2.5-3B-Instruct'
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--proxy')
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    manifest_path = args.destination / 'frozen_model_manifest.json'
    if manifest_path.exists():
        print('Model manifest already exists; verify it before reusing this directory.', flush=True)
        return 0
    session = requests.Session()
    if args.proxy:
        session.proxies.update(http=args.proxy, https=args.proxy)
    response = session.get(f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}', timeout=40)
    response.raise_for_status()
    details = response.json()
    names = [s['rfilename'] for s in details['siblings']
             if '/' not in s['rfilename'] and (s['rfilename'].endswith(('.json', '.safetensors', '.txt'))
                                             or s['rfilename'] in ('LICENSE', 'README.md'))]
    files = {}
    for name in names:
        path = args.destination / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.partial')
            for attempt in range(3):
                try:
                    with session.get(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}',
                                     timeout=(30, 90), stream=True) as stream:
                        stream.raise_for_status()
                        size = 0
                        last = time.monotonic()
                        print(f'Downloading {name}', flush=True)
                        with temporary.open('wb') as output:
                            for chunk in stream.iter_content(4 * 1024 * 1024):
                                if chunk:
                                    output.write(chunk)
                                    size += len(chunk)
                                    if time.monotonic() - last > 20:
                                        print(f'{name}: {size // (1024 * 1024)} MiB', flush=True)
                                        last = time.monotonic()
                    temporary.replace(path)
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise RuntimeError(f'Download failed for {name}; no complete manifest written') from None
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        files[name] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
        print(f'Prepared {name}: {files[name]["bytes"]} bytes', flush=True)
    manifest_path.write_text(json.dumps({'model': MODEL, 'revision': REVISION, 'files': files,
                                        'created_at': datetime.now(timezone.utc).isoformat(),
                                        'trained': False}, indent=2), encoding='utf-8')
    print('Frozen model download complete.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
