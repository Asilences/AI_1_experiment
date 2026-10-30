"""An explicit natural-question adapter, separate from the experiment baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ROOT, write_json
from .demo import Demo

DEFAULT_DATA = ROOT / 'data/processed/movies_5000_20261029'
DEFAULT_MODEL = ROOT / 'models/pipeline_5000_20261029/model.pkl'


def answer(question, data_dir=DEFAULT_DATA, model_path=DEFAULT_MODEL):
    demo = Demo(load_neural=False, data_dir=data_dir, model_path=model_path)
    return {'question': question, 'mode': 'freeform_spine', 'result': demo.answer_a0(question)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question', required=True)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    result = answer(args.question, args.data_dir, args.model_path)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
