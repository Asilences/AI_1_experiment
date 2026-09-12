from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[1]
RELATIONS = {'P57': 'director', 'P136': 'genre', 'P495': 'country of origin', 'P364': 'original language'}
REFUSAL = 'The available knowledge graph does not contain enough information to answer this question.'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def setup_logging(name):
    directory = ROOT / 'runs' / 'logs'
    directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(directory / f'{name}.log', encoding='utf-8')], force=True)
    logging.info('START %s cwd=%s', name, Path.cwd())


def seed_all(seed=42):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def join_names(names):
    if len(names) < 2:
        return ''.join(names)
    return ', '.join(names[:-1]) + ' and ' + names[-1]


def answer_text(title, relation, names, variant=0):
    values = join_names(names)
    pairs = {
        'P57': (f'{title} was directed by {values}.', f'The director or directors of {title} are {values}.'),
        'P136': (f'{title} belongs to the following genres: {values}.', f'The genres of {title} are {values}.'),
        'P495': (f'{title} originated in {values}.', f'The countries of origin of {title} are {values}.'),
        'P364': (f'The original languages of {title} are {values}.', f'{title} was originally made in {values}.')}
    return pairs[relation][variant]


def utc_now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
