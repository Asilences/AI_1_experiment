from __future__ import annotations

from .common import ROOT, read_json, write_json
from .human_eval import build as build_human_eval
from .metrics import evaluate_robustness
from .neural import infer


def select_best():
    candidates = {}
    for run_id in ('B0', 'B1', 'B2', 'B3'):
        linked_metrics = ROOT / 'runs' / run_id / 'metrics_valid.json'
        if linked_metrics.exists():
            candidates[run_id] = read_json(linked_metrics)['entity_macro_f1']
        else:
            history = read_json(ROOT / 'runs' / run_id / 'history.json')
            candidates[run_id] = max(row['entity_macro_f1'] for row in history)
    best = max(candidates, key=lambda key: (candidates[key], -int(key[1:])))
    write_json(ROOT / 'output' / 'selected_model.json', {'selected_run': best, 'validation_f1': candidates[best], 'candidates': candidates})
    infer(best, ROOT / 'data' / 'processed' / 'robustness.jsonl')
    evaluate_robustness(best)
    build_human_eval(best, 100)
    return best


if __name__ == '__main__': print(select_best())
