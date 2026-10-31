"""Reproduce train-free acceptance figures from saved summary.json only.

Run with the project .venv. No inference, model loading or training is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'verification/2026-10-31/trainfree/runtime_5000_final_v2/summary.json'
DEFAULT_OUTPUT = ROOT / 'verification/2026-10-31/trainfree/figures'
MODES = ('raw', 'guarded', 'hybrid')
MODE_LABELS = ('Raw', 'Guarded', 'Hybrid')
SYSTEMS = (('a0', 'A0 pipeline'), ('neural', 'B0 neural'))
COLORS = ('#0072B2', '#D55E00')
METRICS = (
    'supported_entity_f1', 'unanswerable_refusal_accuracy',
    'supported_answer_coverage', 'tool_fallback_rate',
)


def load_summary(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    plotted = {}
    for prefix, _ in SYSTEMS:
        for mode in MODES:
            key = f'{prefix}_{mode}'
            row = data[key]
            if (row['count'], row['supported_count'], row['unanswerable_count'], row['clarification_count']) != (48, 24, 12, 12):
                raise ValueError('This labelled figure requires 48 cases: 24 supported, 12 unanswerable, 12 clarification.')
            plotted[key] = {}
            for metric in METRICS:
                value = row[metric]
                if not isinstance(value, (int, float)) or not 0 <= value <= 100:
                    raise ValueError(f'Expected an unscaled 0-100 percentage for {key}.{metric}')
                plotted[key][metric] = value
    return plotted, hashlib.sha256(raw).hexdigest()


def panel(axis, values, metric, title, denominator):
    x = np.arange(len(MODES))
    width = .34
    for offset, ((prefix, label), color) in enumerate(zip(SYSTEMS, COLORS)):
        heights = [values[f'{prefix}_{mode}'][metric] for mode in MODES]
        bars = axis.bar(x + (offset - .5) * width, heights, width,
                        label=label, color=color, edgecolor='white', linewidth=.7)
        for bar, height in zip(bars, heights):
            # Full bars remain inside the 0-100 scale; labels can use margin.
            axis.annotate(f'{height:.2f}', (bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    axis.set_xticks(x, MODE_LABELS)
    axis.set_ylim(0, 100)
    axis.set_yticks(np.arange(0, 101, 20))
    axis.set_ylabel('Percent (%)')
    axis.set_title(f'{title}\n{denominator}', fontsize=11, pad=19)
    axis.set_axisbelow(True)
    axis.grid(axis='y', color='#D9D9D9', linewidth=.65)
    axis.spines[['top', 'right']].set_visible(False)
    axis.margins(x=.10)


def figure(values, panels, title, footer, stem, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 5.0))
    for axis, config in zip(axes, panels):
        panel(axis, values, *config)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=13, y=.975)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .905), ncol=2, frameon=False)
    fig.text(.5, .035, footer, ha='center', va='bottom', fontsize=9)
    fig.subplots_adjust(left=.07, right=.985, top=.72, bottom=.19, wspace=.26)
    paths = []
    for extension in ('png', 'svg'):
        path = output / f'{stem}.{extension}'
        fig.savefig(path, dpi=200, facecolor='white', metadata={'Creator': 'scripts/plot_trainfree_results.py'})
        paths.append(str(path))
    plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    values, source_hash = load_summary(args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.hashsalt': 'trainfree-acceptance-v1'})
    outputs = []
    outputs += figure(values, [
        ('supported_entity_f1', 'Supported-question entity F1', 'Macro F1 on 24 answerable cases'),
        ('unanswerable_refusal_accuracy', 'Unanswerable refusal accuracy', '12 out-of-scope / unknown-entity cases'),
    ], 'Train-free system comparison: development acceptance (48 cases)',
        'Fixed KB; authored development cases, not a held-out benchmark.\n'
        'Hybrid includes explicit KB-template fallback; it is not pure neural generation.',
        'trainfree_quality', args.output_dir)
    outputs += figure(values, [
        ('supported_answer_coverage', 'Supported-question answer coverage', 'Non-refusal answers / 24 answerable cases'),
        ('tool_fallback_rate', 'Explicit tool fallback rate', 'Tool-supplied final answers / all 48 cases'),
    ], 'Coverage and tool intervention: development acceptance (48 cases)',
        'All percentage axes start at 0 and end at 100. Coverage is not correctness.\n'
        f'B0 hybrid tool interventions: {values["neural_hybrid"]["tool_fallback_rate"]:.2f}% of all 48 cases; reported separately from model generation.',
        'trainfree_coverage', args.output_dir)
    evidence = {'summary_path': str(args.summary.resolve()), 'summary_sha256': source_hash,
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'matplotlib': matplotlib.__version__, 'numpy': np.__version__,
                'scope': '48 authored development-acceptance cases; not held-out generalization or human evaluation.',
                'percentage_axis_limits': [0, 100], 'values': values, 'outputs': outputs}
    (args.output_dir / 'plot_manifest.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
