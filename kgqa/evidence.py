"""Export saved records, never simulated terminal screenshots."""
from __future__ import annotations

import json

from .common import ROOT, read_json, write_json


def build():
    out = ROOT / 'output' / 'evidence'
    out.mkdir(parents=True, exist_ok=True)
    sources = {
        'environment': ROOT / 'output' / 'environment.json',
        'training_config': ROOT / 'runs' / 'B0' / 'config.json',
        'training_history': ROOT / 'runs' / 'B0' / 'history.json',
        'selected_model': ROOT / 'output' / 'selected_model.json',
    }
    for run in ('A0', 'B0', 'B1', 'B2', 'B3', 'C0'):
        sources[f'{run}_metrics'] = ROOT / 'runs' / run / 'metrics_test.json'
    summary = {'record_type': 'saved_log_summary_not_screenshot', 'records': {}, 'missing': []}
    for name, path in sources.items():
        relative = path.relative_to(ROOT).as_posix()
        if path.exists():
            summary['records'][name] = {'source': relative, 'data': read_json(path)}
        else:
            summary['missing'].append(relative)
    write_json(out / 'run_summary.json', summary)
    (out / 'run_summary.txt').write_text(
        'Saved run summaries. This file is NOT a terminal screenshot.\n'
        'Capture actual terminal windows in output/screenshots.\n\n'
        + json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    build()
