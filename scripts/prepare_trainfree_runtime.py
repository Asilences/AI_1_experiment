"""在新目录准备完整5000电影运行包与当前无需训练增强代码。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


PROJECT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    archive, destination = args.archive.resolve(), args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f'保留已有目录，请使用新的目标路径：{destination}')
    with zipfile.ZipFile(archive) as package:
        for item in package.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f'运行包路径越界：{item.filename}')
        destination.mkdir(parents=True)
        package.extractall(destination)
    runtime = destination / 'AI_1_experiment'
    required = ('data/processed/entities.json', 'data/processed/splits.json',
                'models/pipeline/model.pkl', 'output/selected_model.json')
    if not all((runtime / name).is_file() for name in required):
        raise ValueError('运行包缺少配套数据或模型；保留解包目录供检查。')
    packaged = json.loads((runtime / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    for relative, metadata in packaged['files'].items():
        target = (runtime / relative).resolve()
        if not target.is_relative_to(runtime) or not target.is_file():
            raise ValueError(f'运行包清单路径无效：{relative}')
        if target.stat().st_size != metadata['bytes'] or digest(target) != metadata['sha256']:
            raise ValueError(f'运行包校验失败：{relative}')
    source_hashes = {}
    for source in (PROJECT / 'kgqa').rglob('*'):
        if source.is_file() and '__pycache__' not in source.parts and source.suffix != '.pyc':
            relative = source.relative_to(PROJECT)
            target = runtime / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            source_hashes[relative.as_posix()] = digest(source)
    challenge = 'benchmarks/trainfree_challenge_v1.jsonl'
    (runtime / 'benchmarks').mkdir(exist_ok=True)
    shutil.copy2(PROJECT / challenge, runtime / challenge)
    source_hashes[challenge] = digest(PROJECT / challenge)
    (destination / 'overlay_manifest.json').write_text(json.dumps({
        'archive': str(archive), 'archive_sha256': digest(archive), 'project': str(PROJECT),
        'runtime': str(runtime), 'source_sha256': source_hashes,
        'packaged_files_verified': len(packaged['files']),
        'note': '仅覆盖新解包目录的程序代码，数据及模型保持运行包版本，未训练。'
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(runtime)


if __name__ == '__main__':
    main()
