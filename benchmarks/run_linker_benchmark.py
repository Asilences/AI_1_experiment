"""Offline, hand-written regression fixture, NOT the Wikidata test split.

The legacy implementation is frozen from commit 9b85f49. Each setting runs three
times on the same cases; this is a threshold sensitivity check, not model training.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kgqa.linker import EntityLinker
from legacy_linker import EntityLinker as LegacyEntityLinker


ENTITIES = {
    'Q1': {'label': 'Star Wars', 'aliases': ['A New Hope']},
    'Q2': {'label': 'Film Beta', 'aliases': ['Beta Picture']},
    'Q3': {'label': 'Silver Comet', 'aliases': ['The Comet']},
    'Q4': {'label': 'Ocean Melody', 'aliases': ['Song of the Sea']},
    'Q5': {'label': 'Garden Secrets', 'aliases': []},
    'Q6': {'label': 'Winter Journey', 'aliases': []},
    'Q7': {'label': 'Her', 'aliases': []},
    'Q8': {'label': 'Star Wars Episode Four', 'aliases': []},
    'Q9': {'label': 'Alpha Film', 'aliases': ['Shared Title']},
    'Q10': {'label': 'Gamma Film', 'aliases': ['Shared Title']},
}


def fixture():
    rows = []
    def add(group, question, expected):
        rows.append({'id': f'case_{len(rows)+1:03}', 'group': group,
                     'question': question, 'expected_id': expected})
    for qid, entity in ENTITIES.items():
        title = entity['label']
        add('clean', f'Who directed {title}?', qid)
        add('paraphrase', f'Which country did {title} originate in?', qid)
    for title, qid in [('A New Hope', 'Q1'), ('Beta Picture', 'Q2'), ('The Comet', 'Q3'), ('Song of the Sea', 'Q4')]:
        add('alias', f'Who directed {title}?', qid)
    for title, qid in [('Str Wars', 'Q1'), ('Film Bta', 'Q2'), ('Silver Comt', 'Q3'),
                       ('Ocean Meloy', 'Q4'), ('Gardn Secrets', 'Q5'), ('Winter Journy', 'Q6'),
                       ('Silvr Comt', 'Q3'), ('Ocan Meloy', 'Q4')]:
        add('typo', f'Who directed {title}?', qid)
    for question in ['sraW ratS', 'ateB mliF', 'temoC revliS', 'ydoleM naecO',
                     'Who directed Completely Unknown Picture?', 'Who directed The Avengers?',
                     'Who directed The Unavailable Movie?', 'What is the weather tomorrow?',
                     'Who directed Hre?', 'Who directed Shared Title?',
                     'Compare Star Wars and Film Beta', 'Compare Her and Ocean Melody']:
        add('refusal', question, None)
    return rows


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    cases = fixture()
    fixture_json = json.dumps({'entities': ENTITIES, 'cases': cases}, ensure_ascii=False, sort_keys=True, indent=2)
    (output / 'fixture.json').write_text(fixture_json, encoding='utf-8')
    settings = [('legacy', None)] + [(f'threshold_{t:.2f}', t) for t in (.72, .80, .88, .96)]
    trials, detail = [], []
    for name, threshold in settings:
        for repeat in range(1, 4):
            model = LegacyEntityLinker(ENTITIES, ENTITIES) if threshold is None else EntityLinker(
                ENTITIES, ENTITIES, fuzzy_threshold=threshold, ambiguity_margin=.06)
            correct, positive, negative = 0, 0, 0
            times, groups = [], defaultdict(list)
            # One untimed warm-up; do not include construction or loading in timing.
            model.link(cases[0]['question'])
            for case in cases:
                started = time.perf_counter_ns()
                qid, score, status = model.link(case['question'])
                elapsed = (time.perf_counter_ns() - started) / 1e6
                good = qid == case['expected_id']
                correct += good
                if case['expected_id'] is None: negative += good
                else: positive += good
                times.append(elapsed); groups[case['group']].append(good)
                detail.append({**case, 'setting': name, 'repeat': repeat, 'predicted_id': qid,
                               'score': score, 'status': status, 'correct': good, 'latency_ms': elapsed})
            trial = {'setting': name, 'threshold': threshold, 'repeat': repeat, 'count': len(cases),
                     'correct': correct, 'accuracy': 100 * correct / len(cases),
                     'answerable_accuracy': 100 * positive / sum(c['expected_id'] is not None for c in cases),
                     'refusal_accuracy': 100 * negative / sum(c['expected_id'] is None for c in cases),
                     'mean_linker_ms': statistics.mean(times),
                     'by_group': {g: {'count': len(v), 'accuracy': 100 * statistics.mean(v)} for g, v in groups.items()}}
            trials.append(trial)
            print(f'{name:16} repeat={repeat} correct={correct}/{len(cases)} '
                  f'answerable={trial["answerable_accuracy"]:.2f}% refusal={trial["refusal_accuracy"]:.2f}% '
                  f'linker_mean={trial["mean_linker_ms"]:.4f}ms', flush=True)
    summary = {'scope': 'handwritten_linker_regression_not_formal_wikidata_evaluation',
               'baseline_commit': '9b85f49cd928d715a36444384d8e90e20ce3106b',
               'fixture_sha256': hashlib.sha256(fixture_json.encode()).hexdigest(),
               'case_count': len(cases), 'groups': dict(Counter(c['group'] for c in cases)),
               'ambiguity_margin': .06, 'repeats_per_setting': 3, 'trials': trials,
               'environment': {'python': sys.version, 'executable': sys.executable, 'os': platform.platform(),
                   'cwd': str(Path.cwd()), 'cpu': platform.processor(),
                   'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'scikit-learn', 'pytest', 'sacrebleu', 'rouge-score']}}}
    (output / 'linker_benchmark.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    for filename, records in [('trials.csv', [{k:v for k,v in row.items() if k != 'by_group'} for row in trials]),
                               ('predictions.csv', detail)]:
        with (output / filename).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/verification'))
    run(parser.parse_args().output)
