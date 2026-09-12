from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from .common import ROOT, read_json, write_json


def plot():
    output = ROOT / 'output' / 'figures'; output.mkdir(parents=True, exist_ok=True)
    runs = ['B0', 'B1', 'B2', 'B3']; colors = ['#16697A', '#E76F51', '#6A994E', '#7B2CBF']
    available = [run for run in runs if (ROOT / 'runs' / run / 'history.json').exists()]
    if available:
        plt.figure(figsize=(7, 4.5))
        for run, color in zip(available, colors):
            history = read_json(ROOT / 'runs' / run / 'history.json')
            values = [x.get('validation_loss', x['train_loss']) for x in history]
            plt.plot([x['epoch'] for x in history], values, marker='o', label=run, color=color)
        plt.xlabel('Epoch'); plt.ylabel('Validation loss'); plt.legend(); plt.tight_layout()
        plt.savefig(output / 'learning_rate_loss.png', dpi=180); plt.close()

    lr_runs = [run for run in runs if (ROOT / 'runs' / run / 'metrics_test.json').exists()]
    if lr_runs:
        points = sorted([(read_json(ROOT / 'runs' / run / 'config.json')['learning_rate'], run,
                          read_json(ROOT / 'runs' / run / 'metrics_test.json')) for run in lr_runs])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.semilogx([x[0] for x in points], [x[2]['entity_macro_f1'] for x in points], marker='o', label='Entity F1')
        ax.semilogx([x[0] for x in points], [x[2]['rouge_l'] for x in points], marker='s', label='ROUGE-L')
        for lr, run, metric in points:
            ax.annotate(run, (lr, metric['entity_macro_f1']), xytext=(0, 7), textcoords='offset points', ha='center')
        ax.set_xlabel('Learning rate'); ax.set_ylabel('Score'); ax.legend(); fig.tight_layout()
        fig.savefig(output / 'learning_rate_metrics.png', dpi=180); plt.close(fig)

    metric_runs = [run for run in ['A0', 'B0', 'B1', 'B2', 'B3', 'C0'] if (ROOT / 'runs' / run / 'metrics_test.json').exists()]
    if metric_runs:
        metrics = [read_json(ROOT / 'runs' / run / 'metrics_test.json') for run in metric_runs]
        x = range(len(metric_runs)); fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].bar(x, [m['entity_macro_f1'] for m in metrics], color='#16697A'); axes[0].set_title('Entity macro F1'); axes[0].set_xticks(list(x), metric_runs)
        axes[1].bar(x, [m['rouge_l'] for m in metrics], color='#E76F51'); axes[1].set_title('ROUGE-L'); axes[1].set_xticks(list(x), metric_runs)
        fig.tight_layout(); fig.savefig(output / 'system_metrics.png', dpi=180); plt.close(fig)
        write_json(ROOT / 'output' / 'summary_metrics.json', dict(zip(metric_runs, metrics)))

    if all((ROOT / 'runs' / run / 'metrics_test.json').exists() for run in ('B0', 'C0')):
        labels = ['Entity exact', 'Entity F1', 'BLEU', 'ROUGE-L']; keys = ['entity_exact', 'entity_macro_f1', 'bleu', 'rouge_l']
        b0, c0 = (read_json(ROOT / 'runs' / run / 'metrics_test.json') for run in ('B0', 'C0'))
        x = range(len(labels)); width = .36; fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar([i-width/2 for i in x], [b0[k] for k in keys], width, label='B0 with TransE', color='#16697A')
        ax.bar([i+width/2 for i in x], [c0[k] for k in keys], width, label='C0 without TransE', color='#E76F51')
        ax.set_xticks(list(x), labels); ax.set_ylabel('Score'); ax.legend(); fig.tight_layout()
        fig.savefig(output / 'embedding_ablation.png', dpi=180); plt.close(fig)

    selected_path = ROOT / 'output' / 'selected_model.json'
    if selected_path.exists():
        selected = read_json(selected_path)['selected_run']
        if all((ROOT / 'runs' / run / 'metrics_robustness.json').exists() for run in ('A0', selected)):
            variants = ['clean', 'paraphrase', 'alias', 'typo', 'unanswerable']
            a0 = read_json(ROOT / 'runs' / 'A0' / 'metrics_robustness.json')
            neural = read_json(ROOT / 'runs' / selected / 'metrics_robustness.json')
            present = [v for v in variants if v in a0 and v in neural]
            x = range(len(present)); width = .36; fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.bar([i-width/2 for i in x], [a0[v]['entity_macro_f1'] for v in present], width, label='A0', color='#16697A')
            ax.bar([i+width/2 for i in x], [neural[v]['entity_macro_f1'] for v in present], width, label=selected, color='#E76F51')
            ax.set_xticks(list(x), present); ax.set_ylabel('Entity macro F1'); ax.legend(); fig.tight_layout()
            fig.savefig(output / 'robustness.png', dpi=180); plt.close(fig)


if __name__ == '__main__': plot()

