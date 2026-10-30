"""Uniform CPU rescore of immutable predictions with the contextual-list decoder."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kgqa.common import REFUSAL, read_json, read_jsonl, sha256, utc_now, write_json, write_jsonl
from kgqa.linker import EntityLinker, normalize
from kgqa.metrics import AnswerEntityIndex, score_rows
from kgqa.neural_preflight import FACT_LIMIT, _facts


def figures(results, project, destination):
    import matplotlib.pyplot as plt
    destination.mkdir()
    metric = lambda run: results[run]['test']['v2']
    points = sorted((read_json(project / f'runs/{run}/config.json')['learning_rate'], run)
                    for run in ('B0', 'B1', 'B2', 'B3'))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label, marker in [('entity_macro_f1', 'Entity F1', 'o'), ('rouge_l', 'ROUGE-L', 's')]:
        ax.semilogx([p[0] for p in points], [metric(p[1])[key] for p in points], marker=marker, label=label)
    for lr, run in points:
        ax.annotate(run, (lr, metric(run)['entity_macro_f1']), xytext=(0, 7), textcoords='offset points', ha='center')
    ax.set(xlabel='Learning rate', ylabel='Score', title='Raw generation, decoder v2'); ax.legend()
    fig.tight_layout(); fig.savefig(destination / 'learning_rate_metrics.png', dpi=180); plt.close(fig)
    runs = ['A0', 'B0', 'B1', 'B2', 'B3', 'C0', 'B0_recovered']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, title in zip(axes, ['entity_macro_f1', 'rouge_l'], ['Entity macro F1', 'ROUGE-L']):
        ax.bar(runs, [metric(run)[key] for run in runs], color='#16697A')
        ax.set_title(title + ' (decoder v2)'); ax.tick_params(axis='x', labelrotation=35)
    fig.tight_layout(); fig.savefig(destination / 'system_metrics.png', dpi=180); plt.close(fig)
    labels, keys, width = ['Entity exact', 'Entity F1', 'BLEU', 'ROUGE-L'], ['entity_exact', 'entity_macro_f1', 'bleu', 'rouge_l'], .36
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for offset, run, label, color in [(-width/2, 'B0', 'B0 with TransE', '#16697A'), (width/2, 'C0', 'C0 without TransE', '#E76F51')]:
        ax.bar([i+offset for i in range(4)], [metric(run)[key] for key in keys], width, label=label, color=color)
    ax.set_xticks(range(4), labels); ax.set_ylabel('Score (decoder v2)'); ax.legend()
    fig.tight_layout(); fig.savefig(destination / 'embedding_ablation.png', dpi=180); plt.close(fig)
    variants = ['clean', 'paraphrase', 'alias', 'typo', 'unanswerable']
    fig, ax = plt.subplots(figsize=(8, 4.5)); width = .25
    for offset, run, color in [(-width, 'A0', '#16697A'), (0, 'B0', '#E76F51'), (width, 'B0_recovered', '#6A994E')]:
        ax.bar([i+offset for i in range(5)], [results[run]['robustness']['by_variant_v2'][v]['entity_macro_f1'] for v in variants], width, label=run, color=color)
    ax.set_xticks(range(5), variants); ax.set_ylabel('Entity macro F1 (decoder v2)'); ax.legend()
    fig.tight_layout(); fig.savefig(destination / 'robustness.png', dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(project):
        raise ValueError('Use a new output directory outside the frozen project')
    verification = ROOT / 'verification/2026-10-30'
    immutable = {f'project/{p.relative_to(project).as_posix()}': p for p in project.rglob('*') if p.is_file()}
    immutable.update({f'verification_v1/{p.relative_to(verification).as_posix()}': p
                      for p in verification.rglob('*') if p.is_file()
                      and not p.relative_to(verification).parts[0].startswith('decoder_revision_v2')})
    before = {key: sha256(path) for key, path in immutable.items()}
    source_hashes = {name: sha256(ROOT / 'kgqa' / name) for name in ('metrics.py', 'linker.py', 'neural_preflight.py', 'name_recovery.py')}
    spec = importlib.util.spec_from_file_location('kgqa._frozen_decoder_v1', project / 'kgqa/metrics.py')
    v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
    def score_ast(path):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'score_rows'))
    assert score_ast(ROOT / 'kgqa/metrics.py') == score_ast(project / 'kgqa/metrics.py')
    data = project / 'data/processed'
    entities, splits = read_json(data / 'entities.json'), read_json(data / 'splits.json')
    mapping = read_json(project / 'models/transe/mapping.json')
    linker, index, old_index = EntityLinker(sum(splits.values(), []), entities), AnswerEntityIndex(entities), v1.AnswerEntityIndex(entities)
    links, tails, parsed, parsed_old = {}, {}, {}, {}
    def context(question, embedded):
        if question not in links:
            links[question] = linker.link(question)[0]
        movie = links[question]; key = (movie, embedded)
        if key not in tails:
            tails[key] = [tail for _, tail in _facts(movie, entities, mapping if embedded else None)[:FACT_LIMIT]]
        return movie, tails[key]
    def decode(text, movie, preferred, old=False):
        cache, decoder = (parsed_old, old_index) if old else (parsed, index)
        key = (text, movie, tuple(preferred))
        if key not in cache:
            cache[key] = decoder.match(text, movie, preferred_ids=preferred)
        return cache[key]
    output.mkdir(parents=True); provenance = output / 'provenance'; provenance.mkdir()
    write_json(provenance / 'immutable_before.json', before)
    jobs = [(run, split, project / f'runs/{run}/predictions_{split}.jsonl', run != 'C0')
            for run in ('B0', 'B1', 'B2', 'B3', 'C0') for split in ('valid', 'test')]
    jobs += [('A0', split, project / f'runs/A0/predictions_{split}.jsonl', False) for split in ('test', 'robustness')]
    jobs += [('B0', 'robustness', project / 'runs/B0/predictions_robustness.jsonl', True)]
    jobs += [('B0_recovered', split, verification / f'name_recovery/B0_{split}/predictions_recovered.jsonl', True)
             for split in ('valid', 'test', 'robustness')]
    pending = []; started = time.perf_counter()
    # Decode from saved questions/text and retrieved context before loading gold.
    for run, split, source, embedded in jobs:
        raw, revised, discrepancies = read_jsonl(source), [], []
        expected = 439 if split == 'robustness' else {'valid': 8857, 'test': 8917}[split]
        assert len(raw) == expected and len({p['question_id'] for p in raw}) == expected
        for pred in raw:
            movie, preferred = context(pred['question'], embedded)
            ids = decode(pred['answer'], movie, preferred)
            if run == 'A0' and set(ids) != set(pred['answer_ids']):
                discrepancies.append({'question_id': pred['question_id'], 'direct_ids': pred['answer_ids'], 'parsed_ids': ids})
            revised.append({**pred, 'answer_ids_v1': pred['answer_ids'], 'answer_ids': pred['answer_ids'] if run == 'A0' else ids,
                            'decoder_revision': 'v2', 'decoder_linked_movie_id': movie})
        pending.append((run, split, source, raw, revised, discrepancies))
        print(f'Decoded {run}/{split}: {len(raw)} rows; {len(links)} cached questions', flush=True)
    gold_rows, robust = read_jsonl(data / 'qa.jsonl'), read_jsonl(data / 'robustness.jsonl')
    results = {}
    for run, split, source, raw, revised, discrepancies in pending:
        gold = robust if split == 'robustness' else [r for r in gold_rows if r['split'] == split]
        gold_by_id = {r['question_id']: r for r in gold}
        assert all(p['question'] == gold_by_id[p['question_id']]['question'] for p in raw)
        old_metric, new_metric = score_rows(gold, raw, entities), score_rows(gold, revised, entities)
        changed, fixed, regressed = [], [], []
        for a, b in zip(raw, revised):
            truth = gold_by_id[a['question_id']]
            def correct(p):
                return set(p['answer_ids']) == set(truth['answer_ids']) if truth['answer_ids'] else normalize(p['answer']) == normalize(REFUSAL)
            if set(a['answer_ids']) != set(b['answer_ids']):
                changed.append({'question_id': a['question_id'], 'v1_ids': a['answer_ids'], 'v2_ids': b['answer_ids']})
            if not correct(a) and correct(b): fixed.append(a['question_id'])
            if correct(a) and not correct(b): regressed.append(a['question_id'])
            assert a['answer'] == b['answer'] and a.get('latency_ms') == b.get('latency_ms')
        assert all(old_metric[k] == new_metric[k] for k in ('bleu', 'rouge_l'))
        entry = {'v1': old_metric, 'v2': new_metric, 'changed_ids': changed, 'fixed_question_ids': fixed,
                 'regressed_question_ids': regressed, 'changed_count': len(changed), 'fixed_count': len(fixed),
                 'regressed_count': len(regressed), 'source_sha256': sha256(source), 'a0_canonical_parse_discrepancies': discrepancies,
                 'text_and_generation_latency_unchanged': True, 'bleu_rouge_unchanged_by_decoder': True}
        if split == 'robustness':
            for version, predictions in [('v1', raw), ('v2', revised)]:
                entry[f'by_variant_{version}'] = {variant: score_rows([r for r in gold if r['variant'] == variant], predictions, entities, allow_extra_predictions=True)
                                                 for variant in sorted({r['variant'] for r in gold})}
        destination = output / 'runs' / run; destination.mkdir(parents=True, exist_ok=True)
        write_jsonl(destination / f'predictions_{split}.jsonl', revised)
        write_json(destination / f'metrics_{split}.json', entry.get('by_variant_v2', new_metric))
        write_json(destination / f'changes_{split}.json', entry)
        results.setdefault(run, {})[split] = entry
        print(f'Scored {run}/{split}: F1 {old_metric["entity_macro_f1"]:.8f} -> {new_metric["entity_macro_f1"]:.8f}; fixed {len(fixed)}, regressed {len(regressed)}', flush=True)
    canonical = {}
    for split in ('train', 'valid', 'test'):
        rows = [r for r in gold_rows if r['split'] == split]; errors = {version: [] for version in ('v1', 'v2')}; comparisons = 0
        for row in rows:
            if split == 'train':
                movie = row['movie_id']; preferred = [tail for _, tail in _facts(movie, entities, mapping)[:FACT_LIMIT]]
            else:
                movie, preferred = context(row['question'], True)
            for reference in row['reference_answers']:
                comparisons += 1
                for version in errors:
                    decoded = decode(reference, movie, preferred, old=version == 'v1')
                    if set(decoded) != set(row['answer_ids']):
                        errors[version].append({'question_id': row['question_id'], 'reference': reference, 'parsed_ids': decoded, 'expected_ids': row['answer_ids']})
        canonical[split] = {'rows': len(rows), 'reference_comparisons': comparisons, **{f'{v}_mismatches': len(e) for v, e in errors.items()}, 'examples': errors}
    write_json(provenance / 'canonical_reference_audit.json', {'scope': 'Gold used only for audit comparison; train context is gold movie; evaluation context uses actual question linker.', 'splits': canonical, 'total_reference_comparisons': sum(v['reference_comparisons'] for v in canonical.values())})
    write_json(provenance / 'source_checks.json', {'full_pytest': {'passed': 344, 'seconds': 17.03, 'evidence': 'Completed preceding console run with CPU threads 2 and CUDA hidden; not rerun by this adapter.'}, 'targeted_pytest': {'passed': 32, 'seconds': 2.97}, 'prior_fix_pytest': {'passed': 342, 'seconds': 17.35}, 'previous_v1_pytest': {'passed': 330, 'preserved_evidence': 'verification/2026-10-30/source_checks/pytest.log'}})
    selected_v1 = read_json(project / 'output/selected_model.json')['selected_run']
    selected_v2 = min(('B0', 'B1', 'B2', 'B3'), key=lambda run: (-results[run]['valid']['v2']['entity_macro_f1'], run))
    figures(results, project, output / 'figures')
    after = {key: sha256(path) for key, path in immutable.items()}
    write_json(provenance / 'immutable_after.json', after)
    assert before == after, 'Frozen project or original v1 evidence changed'
    assert source_hashes == {name: sha256(ROOT / 'kgqa' / name) for name in source_hashes}
    summary = {'created_at': utc_now(), 'decoder_revision': 'v2', 'discovered_after_test': True,
               'scope': 'Post-training parser correction uniformly applied to saved outputs; no training, parameter change, or NameRecovery rerun.',
               'runs': results, 'selected_v1': selected_v1, 'selected_v2_raw_validation': selected_v2,
               'selection_unchanged': selected_v1 == selected_v2, 'review_required': selected_v1 != selected_v2,
               'selection_policy': 'Maximum raw B-run validation entity F1, then ascending run ID; retain original selection pending review if it changes.',
               'old_decoder_sha256': sha256(project / 'kgqa/metrics.py'), 'new_decoder_sha256': source_hashes['metrics.py'],
               'source_sha256': source_hashes, 'adapter_sha256': sha256(Path(__file__)), 'score_rows_ast_unchanged': True,
               'frozen_checkpoints_unaltered': True, 'all_original_v1_files_byte_unchanged': True, 'immutable_file_count': len(before),
               'canonical_reference_audit': canonical, 'cpu_threads': 2, 'gpu_used': False,
               'cached_questions': len(links), 'elapsed_seconds': time.perf_counter() - started,
               'unchanged_figures': ['learning_rate_loss.png', 'TransE loss and embedding figures'],
               'timing_note': 'Saved latency_ms is model-generation latency; decoding CPU time is not substituted.'}
    write_json(output / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('selected_v1', 'selected_v2_raw_validation', 'selection_unchanged', 'new_decoder_sha256', 'immutable_file_count', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
