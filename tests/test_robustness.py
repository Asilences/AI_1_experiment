from collections import Counter
import random
import runpy
import sys

import pytest

from kgqa import robustness
from kgqa.common import REFUSAL, read_json, read_jsonl, write_json, write_jsonl


TEMPLATES = {
    'P57': 'Who directed {title}?',
    'P136': 'What genre is {title}?',
    'P495': 'Which country did {title} originate in?',
    'P364': 'What is the original language of {title}?',
}


def source_row(movie='M1', title='Alpha Film', relation='P57', index=0, **changes):
    row = {
        'question_id': f'{movie}_{relation}_{index}', 'movie_id': movie,
        'relation_id': relation, 'question': TEMPLATES.get(relation, 'Who wrote {title}?').format(title=title),
        'split': 'test', 'is_answerable': relation != 'UNKNOWN',
        'answer_ids': ['A1'] if relation != 'UNKNOWN' else [],
        'reference_answers': ['A fixed reference answer.'] if relation != 'UNKNOWN' else [REFUSAL],
        'template_family': index,
    }
    return {**row, **changes}


def movie_rows(count=12, repeats=3):
    entities, rows = {}, []
    for index in range(count):
        movie, title = f'M{index:03}', f'Film Number {index:03}'
        entities[movie] = {'label': title, 'aliases': [f'Movie Number {index:03}']}
        rows.extend(source_row(movie, title, relation, variant)
                    for relation in TEMPLATES for variant in range(repeats))
    return entities, rows


@pytest.fixture
def source_files(tmp_path, monkeypatch):
    monkeypatch.setattr(robustness, 'ROOT', tmp_path)

    def save(rows, entities=None):
        write_jsonl(tmp_path / 'data/processed/qa.jsonl', rows)
        write_json(tmp_path / 'data/processed/entities.json', entities or {
            'M1': {'label': 'Alpha Film', 'aliases': ['Alpha']},
        })
        return tmp_path

    return save


def variants(rows, kind):
    return [row for row in rows if row['variant'] == kind]


def test_seeded_sample_covers_movies_relations_and_is_order_independent(source_files):
    entities, source = movie_rows()
    root = source_files(source, entities)
    state = random.getstate()
    output = robustness.build(limit=13, seed=7)
    assert random.getstate() == state
    clean = variants(output, 'clean')
    assert len(clean) == 13
    assert len({row['movie_id'] for row in clean[:12]}) == 12
    assert max(Counter(row['movie_id'] for row in clean).values()) == 2
    relation_counts = Counter(row['relation_id'] for row in clean)
    assert set(relation_counts) == set(TEMPLATES)
    assert max(relation_counts.values()) - min(relation_counts.values()) <= 1
    assert output == read_jsonl(root / 'data/processed/robustness.jsonl')
    manifest = read_json(root / 'data/processed/robustness_manifest.json')
    source_files(list(reversed(source)), entities)
    assert robustness.build(limit=13, seed=7) == output
    assert read_json(root / 'data/processed/robustness_manifest.json') == manifest
    other_seed = robustness.build(limit=13, seed=8)
    assert [row['source_question_id'] for row in variants(other_seed, 'clean')] != [
        row['source_question_id'] for row in clean]


def test_variants_keep_identical_sources_and_unchanged_gold(source_files):
    entities, source = movie_rows(4)
    source_by_id = {row['question_id']: row for row in source}
    root = source_files(source, entities)
    output = robustness.build(limit=17, seed=123)
    source_ids = [row['source_question_id'] for row in variants(output, 'clean')]
    assert len({row['question_id'] for row in output}) == len(output)
    for kind in ('paraphrase', 'typo', 'alias'):
        assert [row['source_question_id'] for row in variants(output, kind)] == source_ids
    for row in output:
        original = source_by_id[row['source_question_id']]
        for field in ('movie_id', 'relation_id', 'answer_ids', 'reference_answers', 'split', 'is_answerable'):
            assert row[field] == original[field]
        assert row['paired_clean_question_id'] == f"robust_clean_{row['source_question_id']}"
    manifest = read_json(root / 'data/processed/robustness_manifest.json')
    assert manifest['sampled_answerable']['source_question_ids'] == source_ids
    assert manifest['seed'] == 123 and manifest['requested_limit'] == 17
    for kind in ('paraphrase', 'typo', 'alias'):
        comparison = manifest['comparisons'][kind]
        assert comparison['matched_count'] == comparison['matched_clean_denominator'] == 17
        assert comparison['variant_denominator'] == comparison['all_clean_denominator'] == 17
        assert comparison['matched_clean_records'] == variants(output, 'clean')


def test_aliases_use_only_eligible_subset_with_explicit_clean_pairs(source_files):
    entities = {
        'M1': {'label': 'Alpha Film', 'aliases': ['Alpha Film', 'ALPHA FILM', 'Long Alias', 'AF', 'AF', '']},
        'M2': {'label': 'Beta Film', 'aliases': []},
        'M3': {'label': 'Gamma Film', 'aliases': ['gamma film', ' Gamma Film ']},
        'M4': {'label': 'Delta Film', 'aliases': ['DF']},
    }
    source = [source_row(movie, entity['label']) for movie, entity in entities.items()]
    source[-1]['question'] = 'Who directed an unrelated title?'
    root = source_files(source, entities)
    output = robustness.build()
    alias = variants(output, 'alias')
    assert len(alias) == 1
    assert alias[0]['question'] == 'Who directed AF?'
    manifest = read_json(root / 'data/processed/robustness_manifest.json')
    comparison = manifest['comparisons']['alias']
    assert comparison['all_clean_denominator'] == 4
    assert comparison['variant_denominator'] == comparison['matched_clean_denominator'] == 1
    clean = next(row for row in variants(output, 'clean') if row['movie_id'] == 'M1')
    assert comparison['matched_clean_records'] == [clean]
    assert comparison['pairs'] == [{'clean_question_id': clean['question_id'],
                                    'variant_question_id': alias[0]['question_id']}]
    assert set(manifest['alias_unavailable']['no_distinct_alias']) == {'M2_P57_0', 'M3_P57_0'}
    assert manifest['alias_unavailable']['title_not_in_question'] == ['M4_P57_0']


def test_typos_change_only_title_and_record_no_ops(source_files):
    entities = {'M1': {'label': 'It', 'aliases': []},
                'M2': {'label': 'AAA', 'aliases': []},
                'M3': {'label': 'Absent Film', 'aliases': []}}
    source = [source_row('M1', 'It', question='Who directed It and what was written?'),
              source_row('M2', 'AAA'),
              source_row('M3', 'Absent Film', question='Who directed an unrelated film?')]
    root = source_files(source, entities)
    output = robustness.build()
    by_movie = {row['movie_id']: row for row in variants(output, 'typo')}
    assert by_movie['M1']['question'] == 'Who directed tI and what was written?'
    assert by_movie['M1']['perturbation_applied'] is True
    for movie in ('M2', 'M3'):
        assert by_movie[movie]['perturbation_applied'] is False
    manifest = read_json(root / 'data/processed/robustness_manifest.json')
    assert set(manifest['typo_unchanged_source_question_ids']) == {'M2_P57_0', 'M3_P57_0'}
    assert manifest['typo_unavailable'] == {'no_swappable_title_letters': ['M2_P57_0'],
                                            'title_not_in_question': ['M3_P57_0']}
    assert manifest['variants']['typo']['count'] == 3
    assert manifest['variants']['typo']['perturbations_applied'] == 1
    assert all(not row['perturbation_applied'] for row in variants(output, 'clean'))


@pytest.mark.parametrize('text', ['', 'A', 'AAA', '123', 'A A'])
def test_typo_leaves_inputs_without_a_legal_swap_unchanged(text):
    assert robustness.typo(text, 42) == text


def test_unanswerable_sampling_is_seeded_capped_and_separate_from_limit(source_files):
    entities, source = movie_rows(4, repeats=1)
    source.extend(source_row(f'N{i:03}', f'Negative Film {i}', 'UNKNOWN') for i in range(80))
    # A separate training movie must never enter the robustness sample.
    source.append(source_row('TRAIN', 'Training Film', split='train'))
    root = source_files(source, entities)
    output = robustness.build(limit=2, seed=11)
    negative = variants(output, 'unanswerable')
    assert len(variants(output, 'clean')) == 2
    assert len(negative) == 50
    assert len({row['movie_id'] for row in negative}) == 50
    assert all(row['paired_clean_question_id'] is None and not row['perturbation_applied'] for row in negative)
    assert all(row['split'] == 'test' and row['movie_id'] != 'TRAIN' for row in output)
    manifest = read_json(root / 'data/processed/robustness_manifest.json')
    assert manifest['eligible_unanswerable']['count'] == 80
    assert manifest['unanswerable_seed'] == 12 and manifest['unanswerable_limit'] == 50
    assert robustness.build(limit=1, seed=11)[-50:] == negative
    assert variants(robustness.build(limit=2, seed=12), 'unanswerable') != negative


def test_limit_larger_than_pool_uses_all_rows_and_allows_no_negative_rows(source_files):
    root = source_files([source_row()])
    output = robustness.build(limit=100)
    assert len(variants(output, 'clean')) == 1
    assert variants(output, 'unanswerable') == []
    manifest = read_json(root / 'data/processed/robustness_manifest.json')
    assert manifest['variants']['unanswerable']['count'] == 0
    assert manifest['variants']['unanswerable']['source_question_ids'] == []


@pytest.mark.parametrize(('rows', 'message'), [
    ([], 'QA data is empty'),
    ([source_row(split='train')], 'No answerable test questions'),
    ([source_row('M1', relation='UNKNOWN')], 'No answerable test questions'),
    ([source_row(question_id='')], 'question_id must be a non-empty string'),
    ([source_row(question_id=None)], 'question_id must be a non-empty string'),
    ([source_row(), source_row()], 'duplicate question_id'),
    ([source_row(), source_row(index=1, split='train')], 'movies occur in multiple splits'),
    ([source_row(movie_id='')], 'non-empty movie_id'),
    ([source_row(question=' ')], 'non-empty question'),
    ([source_row(is_answerable=1)], 'boolean is_answerable'),
    ([source_row(relation='OTHER', is_answerable=True)], 'unsupported answerable relation'),
])
def test_invalid_source_data_fails_before_writing_outputs(source_files, rows, message):
    root = source_files(rows)
    with pytest.raises(ValueError, match=message):
        robustness.build()
    assert not (root / 'data/processed/robustness.jsonl').exists()
    assert not (root / 'data/processed/robustness_manifest.json').exists()


@pytest.mark.parametrize(('arguments', 'message'), [
    ({'limit': 0}, 'limit must be a positive integer'),
    ({'limit': -1}, 'limit must be a positive integer'),
    ({'limit': True}, 'limit must be a positive integer'),
    ({'limit': 1.5}, 'limit must be a positive integer'),
    ({'seed': True}, 'seed must be an integer'),
    ({'seed': '42'}, 'seed must be an integer'),
])
def test_invalid_sampling_options_fail(arguments, message):
    with pytest.raises(ValueError, match=message):
        robustness.build(**arguments)


def test_missing_movie_label_fails_explicitly(source_files):
    source_files([source_row()], {'M1': {'aliases': []}})
    with pytest.raises(ValueError, match="movie 'M1' requires an entity label"):
        robustness.build()


def test_cli_accepts_existing_limit_and_new_seed(source_files, monkeypatch, capsys):
    from kgqa import common

    root = source_files([source_row()])
    monkeypatch.setattr(common, 'ROOT', root)
    monkeypatch.setattr(sys, 'argv', ['kgqa.robustness', '--limit', '1', '--seed', '19'])
    # runpy executes the real module entry point against the temporary source files.
    with pytest.warns(RuntimeWarning, match='found in sys.modules'):
        runpy.run_module('kgqa.robustness', run_name='__main__')
    assert capsys.readouterr().out.strip() == '4'
    manifest = read_json(root / 'data/processed/robustness_manifest.json')
    assert manifest['seed'] == 19 and manifest['requested_limit'] == 1
