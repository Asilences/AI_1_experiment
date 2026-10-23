"""Offline acceptance with a synthetic mini KB, never a Wikidata benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from uuid import uuid4

from . import personal_experiment
from .common import REFUSAL, RELATIONS, ROOT, answer_text, read_json, read_jsonl, sha256, write_json, write_jsonl
from .pipeline import PipelineQA, load_model

RUN_ID = 'A0_acceptance'
DATA_PATH = Path('data/processed')
RUN_PATH = Path('runs') / RUN_ID
SCOPE = 'Synthetic offline acceptance fixture; not a measured Wikidata result or a generalization benchmark.'


class AcceptanceError(RuntimeError):
    """The local acceptance workflow completed with an unsuccessful check."""


def _default_output_dir():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    return ROOT / 'output' / 'acceptance' / f'{stamp}_{uuid4().hex[:8]}'


def build_fixture(data_dir):
    """Create ten entirely invented movies with disjoint 6/2/2 movie splits."""
    movies = [
        ('Amber Station', 'Station of Amber'), ('Blue Orchard', 'Orchard in Blue'),
        ('Copper Voyage', 'Voyage of Copper'), ('Distant Harbor', 'Harbor Far Away'),
        ('Emerald Clock', 'Clock of Emerald'), ('Frost Meadow', 'Meadow of Frost'),
        ('Golden Bridge', 'Bridge of Gold'), ('Ivory Comet', 'Comet of Ivory'),
        ('Juniper Trail', 'Trail of Juniper'), ('Lunar Garden', 'Garden Under Moonlight'),
    ]
    entities = {qid: {'label': label, 'aliases': []} for qid, label in (
        ('SYN_PERSON_A', 'Avery Vale'), ('SYN_PERSON_B', 'Blair Rowan'),
        ('SYN_GENRE_A', 'Adventure'), ('SYN_GENRE_B', 'Mystery'),
        ('SYN_COUNTRY_A', 'Northland'), ('SYN_COUNTRY_B', 'Southland'),
        ('SYN_LANGUAGE_A', 'English'), ('SYN_LANGUAGE_B', 'French'))}
    ids = [f'SYN_FILM_{number:02d}' for number in range(1, 11)]
    splits = {'train': ids[:6], 'valid': ids[6:8], 'test': ids[8:]}
    templates = {
        'P57': ('Who directed {title}?', 'Who was the director of {title}?', 'Name the director or directors of {title}.'),
        'P136': ('What genre is {title}?', 'Which genres does {title} belong to?', 'Name the genre or genres of {title}.'),
        'P495': ('Which country did {title} originate in?', 'What is the country of origin of {title}?', 'Name the countries of origin of {title}.'),
        'P364': ('What is the original language of {title}?', 'Which languages was {title} originally made in?', 'Name the original language or languages of {title}.'),
    }
    unknown = ('Who wrote the screenplay for {title}?', 'What was the budget of {title}?', 'Which company distributed {title}?')
    split_for = {movie: split for split, members in splits.items() for movie in members}
    qa = []
    for index, (movie_id, (title, alias)) in enumerate(zip(ids, movies)):
        facts = {'P57': ['SYN_PERSON_A', 'SYN_PERSON_B'] if index % 2 else ['SYN_PERSON_A'],
                 'P136': ['SYN_GENRE_A', 'SYN_GENRE_B'],
                 'P495': ['SYN_COUNTRY_A'], 'P364': ['SYN_LANGUAGE_A', 'SYN_LANGUAGE_B']}
        entities[movie_id] = {'label': title, 'aliases': [alias], 'facts': facts}
        for relation, questions in templates.items():
            names = [entities[tail]['label'] for tail in facts[relation]]
            for family, question in enumerate(questions):
                qa.append({'question_id': f'{movie_id}_{relation}_{family}', 'movie_id': movie_id,
                           'split': split_for[movie_id], 'relation_id': relation, 'question': question.format(title=title),
                           'answer_ids': list(facts[relation]), 'is_answerable': True, 'template_family': family,
                           'reference_answers': [answer_text(title, relation, names, variant) for variant in (0, 1)]})
        for family, question in enumerate(unknown):
            qa.append({'question_id': f'{movie_id}_UNKNOWN_{family}', 'movie_id': movie_id,
                       'split': split_for[movie_id], 'relation_id': 'UNKNOWN', 'question': question.format(title=title),
                       'answer_ids': [], 'is_answerable': False, 'template_family': family, 'reference_answers': [REFUSAL]})
    data_dir = Path(data_dir)
    write_json(data_dir / 'entities.json', entities)
    write_json(data_dir / 'splits.json', splits)
    write_jsonl(data_dir / 'qa.jsonl', qa)
    write_json(data_dir / 'manifest.json', {'source': 'synthetic', 'scope': SCOPE, 'movies': 10,
                                           'qa': len(qa), 'movie_split_sizes': {key: len(value) for key, value in splits.items()},
                                           'template_note': 'Known templates are shared across disjoint movies to test executable behavior.'})
    robust, pairs = [], []
    for row in qa:
        if row['split'] != 'test' or not row['is_answerable'] or row['template_family'] != 0:
            continue
        clean_id, alias_id = f"clean_{row['question_id']}", f"alias_{row['question_id']}"
        clean = {**row, 'question_id': clean_id, 'source_question_id': row['question_id'],
                 'paired_clean_question_id': clean_id, 'variant': 'clean', 'perturbation_applied': False}
        entity = entities[row['movie_id']]
        alias = {**clean, 'question_id': alias_id, 'variant': 'alias', 'perturbation_applied': True,
                 'question': row['question'].replace(entity['label'], entity['aliases'][0])}
        robust.extend((clean, alias))
        pairs.append({'clean_question_id': clean_id, 'variant_question_id': alias_id})
    write_jsonl(data_dir / 'robustness.jsonl', robust)
    write_json(data_dir / 'robustness_manifest.json', {'source': 'synthetic', 'scope': SCOPE,
        'comparisons': {'alias': {'matched_count': len(pairs), 'pairs': pairs}}})


def _portable_metadata(output_dir):
    """Use output-root-relative metadata for this generated acceptance bundle."""
    directory = output_dir / RUN_PATH
    config = read_json(directory / 'config.json')
    config.update(data_dir=DATA_PATH.as_posix(), model_path=(RUN_PATH / 'model.pkl').as_posix(),
                  paths_relative_to='acceptance output directory', experiment='synthetic offline acceptance',
                  data_scope=SCOPE)
    write_json(directory / 'config.json', config)
    manifest = read_json(directory / 'data_manifest.json')
    for name, info in manifest['input_files'].items():
        info['path'] = (DATA_PATH / name).as_posix()
    manifest['paths_relative_to'] = 'acceptance output directory'
    manifest['data_scope'] = SCOPE
    write_json(directory / 'data_manifest.json', manifest)
    environment = read_json(directory / 'environment.json')
    environment['executable'] = Path(environment['executable']).name
    environment['executable_note'] = 'Executable basename only; environment is recorded without an installation path.'
    write_json(directory / 'environment.json', environment)
    hashes = read_json(directory / 'source_hashes.json')
    hashes['acceptance.py'] = sha256(Path(__file__))
    write_json(directory / 'source_hashes.json', hashes)


def _stable(predictions):
    return [{key: value for key, value in row.items() if key != 'latency_ms'} for row in predictions]


def inspect_outputs(output_dir):
    """Check saved outputs and reloaded-model behavior using the synthetic gold."""
    output_dir = Path(output_dir)
    data, directory = output_dir / DATA_PATH, output_dir / RUN_PATH
    entities, splits = read_json(data / 'entities.json'), read_json(data / 'splits.json')
    test = [row for row in read_jsonl(data / 'qa.jsonl') if row['split'] == 'test']
    predictions = read_jsonl(directory / 'predictions_test.jsonl')
    by_id = {row['question_id']: row for row in predictions}
    metrics = read_json(directory / 'metrics_test.json')
    config = read_json(directory / 'config.json')
    sensitivity = read_json(directory / 'sensitivity_summary.json')
    expected_ids = {row['question_id'] for row in test}
    correct = lambda row: set(by_id.get(row['question_id'], {}).get('answer_ids', [])) == set(row['answer_ids'])
    model, threshold = load_model(directory / 'model.pkl')
    system = PipelineQA(entities, [movie for members in splits.values() for movie in members], model, threshold)
    reloaded = [{**system.predict(row['question']), 'question_id': row['question_id']} for row in test]
    alias_gold = next(row for row in test if row['relation_id'] == 'P57')
    entity = entities[alias_gold['movie_id']]
    alias_question = alias_gold['question'].replace(entity['label'], entity['aliases'][0])
    alias = system.predict(alias_question)
    unknown = system.predict('Who directed Zyxwuv Qpkjrt?')
    refusal_rows = [row for row in test if not row['is_answerable']]
    repeats_stable = len(sensitivity['settings']) == 5
    for setting in sensitivity['settings']:
        trials = setting['repeats']
        saved = [read_jsonl(directory / trial['prediction_path']) for trial in trials]
        repeats_stable &= (len(trials) == 3 and all(len(rows) == len(test) and
                            {row['question_id'] for row in rows} == expected_ids for rows in saved)
                           and all(_stable(rows) == _stable(saved[0]) for rows in saved))
    split_sets = [set(members) for members in splits.values()]
    robust_metrics = read_json(directory / 'metrics_robustness.json')
    matched = read_json(directory / 'metrics_robustness_matched.json')['alias']
    checks = {
        'run_complete': read_json(directory / 'status.json').get('status') == 'complete',
        'disjoint_movie_splits': all(not left & right for index, left in enumerate(split_sets) for right in split_sets[index + 1:]),
        'full_test_coverage': (len(by_id) == len(predictions) == len(test) and set(by_id) == expected_ids
                               and metrics.get('prediction_coverage') == 100 and metrics.get('count') == len(test)),
        'all_test_answers_correct': all(correct(row) for row in test),
        'four_relations_answered': all(any(row['relation_id'] == relation and correct(row) and
                                        by_id.get(row['question_id'], {}).get('relation_id') == relation
                                        for row in test) for relation in RELATIONS),
        'known_movie_refusal': bool(refusal_rows) and all(by_id.get(row['question_id'], {}).get('answer') == REFUSAL
                                                        and not by_id.get(row['question_id'], {}).get('answer_ids') for row in refusal_rows),
        'multiple_answer_example': any(len(row['answer_ids']) > 1 and correct(row) for row in test),
        'alias_example': alias['movie_id'] == alias_gold['movie_id'] and set(alias['answer_ids']) == set(alias_gold['answer_ids']),
        'unknown_movie_refusal': unknown['movie_id'] is None and unknown['answer_ids'] == [] and unknown['answer'] == REFUSAL,
        'reloaded_predictions_stable': _stable(reloaded) == _stable(predictions),
        'sensitivity_complete_and_stable': bool(repeats_stable),
        'validation_only_threshold': (config['selection_split'] == 'valid' and config['feature_fit_split'] == 'train'
                                       and config['test_selection'] is False and config['training_runs'] == 1
                                       and threshold == read_json(directory / 'validation_thresholds.json')['selected_threshold']),
        'matched_alias_comparison': (robust_metrics['alias']['entity_exact'] == 100 and matched['matched_count'] == 8
                                      and matched['matched_clean_metrics']['count'] == matched['matched_variant_metrics']['count'] == 8),
    }
    multi = next(row for row in test if len(row['answer_ids']) > 1)
    examples = {'four_relations': {relation: by_id.get(next(row['question_id'] for row in test if row['relation_id'] == relation))
                                   for relation in RELATIONS},
                'multi_answer': by_id.get(multi['question_id']), 'known_movie_refusal': by_id.get(refusal_rows[0]['question_id']),
                'alias': alias, 'unknown_movie_refusal': unknown}
    return checks, examples, metrics


def run_acceptance(output_dir=None):
    """Create a new isolated acceptance directory, raising on any failed check."""
    output_dir = Path(output_dir) if output_dir is not None else _default_output_dir()
    output_dir = output_dir.resolve()
    # An explicit existing destination is refused, even when empty, to preserve it.
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'FAIL', 'source': 'synthetic', 'scope': SCOPE, 'scientific_result': False,
               'paths_relative_to': 'this summary.json directory', 'data_dir': DATA_PATH.as_posix(),
               'run_dir': RUN_PATH.as_posix(), 'model_path': (RUN_PATH / 'model.pkl').as_posix(),
               'examples_path': 'examples.json', 'checks': {}}
    try:
        build_fixture(output_dir / DATA_PATH)
        personal_experiment.run(output_dir / DATA_PATH, output_dir / 'runs', RUN_ID)
        _portable_metadata(output_dir)
        checks, examples, metrics = inspect_outputs(output_dir)
        summary.update(checks=checks, fixture_metrics=metrics, fixture_movies=10,
                       movie_split_sizes={'train': 6, 'valid': 2, 'test': 2}, training_runs=1,
                       test_forward_passes=15, metric_interpretation=SCOPE)
        write_json(output_dir / 'examples.json', {'source': 'synthetic', 'scope': SCOPE, 'examples': examples})
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise AcceptanceError('Checks failed: ' + ', '.join(failed))
        summary['status'] = 'PASS'
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        summary['error'] = str(error).replace(str(output_dir), '.').replace(output_dir.as_posix(), '.')
        write_json(output_dir / 'summary.json', summary)
        raise
    write_json(output_dir / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, help='A new directory; an existing directory is never overwritten.')
    args = parser.parse_args(argv)
    destination = args.output_dir or _default_output_dir()
    try:
        run_acceptance(destination)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f'FAIL synthetic offline acceptance: {error}', file=sys.stderr)
        return 1
    try:
        location = destination.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        location = destination.name
    print(f'PASS synthetic offline acceptance; not a Wikidata benchmark. Summary: {location}/summary.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
