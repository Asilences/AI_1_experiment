import csv
import json

import pytest

from kgqa import human_eval


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def _dataset(movies=150, variants=2):
    return [
        {'question_id': f'Q{movie:04d}_{relation}_{variant}', 'movie_id': f'Q{movie:04d}',
         'relation_id': relation, 'split': 'test', 'is_answerable': True,
         'question': f'Question about Q{movie:04d} {relation} {variant}',
         'reference_answers': ['First reference.', 'Second reference.']}
        for movie in range(movies)
        for relation in ('P57', 'P136', 'P495', 'P364')
        for variant in range(variants)
    ]


def _setup(tmp_path, monkeypatch, gold=None):
    monkeypatch.setattr(human_eval, 'ROOT', tmp_path)
    gold = _dataset() if gold is None else gold
    _write_jsonl(tmp_path / 'data/processed/qa.jsonl', gold)
    for run in ('A0', 'B2'):
        _write_jsonl(tmp_path / f'runs/{run}/predictions_test.jsonl',
                     [{'question_id': row['question_id'], 'answer': f'{run} answer.'} for row in gold])
    return gold


def _read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def test_sample_covers_movies_and_relations_and_keeps_scores_blank(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    manifest = human_eval.build('B2', limit=100, seed=42)
    assert manifest['sampled_questions'] == 100
    assert manifest['sampled_movies'] == 100
    assert set(manifest['sampled_by_relation'].values()) == {25}
    rows = _read_csv(tmp_path / 'output/human_evaluation.csv')
    keys = _read_csv(tmp_path / 'output/human_evaluation_key.csv')
    assert len(rows) == len(keys) == 200
    assert list(rows[0]) == human_eval.VISIBLE_FIELDS
    for row, key in zip(rows, keys):
        assert row['question_id'] == key['question_id']
        assert row['answer_position'] == key['answer_position']
        assert row['answer'] == {'X': 'A0 answer.', 'Y': 'B2 answer.'}[key['system']]
        assert all(row[field] == '' for field in human_eval.VISIBLE_FIELDS[5:])
    for i in range(0, len(keys), 2):
        assert {keys[i]['system'], keys[i + 1]['system']} == {'X', 'Y'}
    stored = json.loads((tmp_path / 'output/human_evaluation_manifest.json').read_text(encoding='utf-8'))
    assert stored == manifest


def test_sampling_is_reproducible_and_independent_of_input_order(tmp_path, monkeypatch):
    gold = _setup(tmp_path, monkeypatch)
    first = human_eval.build('B2', limit=30, seed=7)
    first_csv = (tmp_path / 'output/human_evaluation.csv').read_bytes()
    _write_jsonl(tmp_path / 'data/processed/qa.jsonl', list(reversed(gold)))
    second = human_eval.build('B2', limit=30, seed=7)
    assert second == first
    assert (tmp_path / 'output/human_evaluation.csv').read_bytes() == first_csv
    assert human_eval.build('B2', limit=30, seed=8)['question_ids'] != first['question_ids']


def test_small_and_uneven_pool_samples_every_question_once(tmp_path, monkeypatch):
    gold = _dataset(movies=3, variants=1)
    gold = gold[:1] + gold[4:6] + gold[8:]
    _setup(tmp_path, monkeypatch, gold)
    manifest = human_eval.build('B2', limit=100)
    assert set(manifest['question_ids']) == {row['question_id'] for row in gold}
    assert len(manifest['question_ids']) == len(gold)
    first_round = manifest['question_ids'][:3]
    assert len({qid.split('_')[0] for qid in first_round}) == 3


@pytest.mark.parametrize('run', ['A0', 'B2'])
def test_duplicate_prediction_ids_fail_before_export(tmp_path, monkeypatch, run):
    gold = _setup(tmp_path, monkeypatch, _dataset(movies=2))
    _write_jsonl(tmp_path / f'runs/{run}/predictions_test.jsonl',
                 [{'question_id': gold[0]['question_id'], 'answer': 'One.'}] * 2)
    with pytest.raises(ValueError, match='duplicate question_id'):
        human_eval.build('B2')
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('run', ['A0', 'B2'])
def test_missing_predictions_in_unsampled_pool_also_fail(tmp_path, monkeypatch, run):
    gold = _setup(tmp_path, monkeypatch, _dataset(movies=2))
    _write_jsonl(tmp_path / f'runs/{run}/predictions_test.jsonl',
                 [{'question_id': row['question_id'], 'answer': 'One.'} for row in gold[:-1]])
    with pytest.raises(ValueError, match='missing 1 eligible question IDs'):
        human_eval.build('B2', limit=1)
    assert not (tmp_path / 'output').exists()


def test_duplicate_gold_ids_fail(tmp_path, monkeypatch):
    gold = _dataset(movies=1)
    _setup(tmp_path, monkeypatch, gold + [gold[0]])
    with pytest.raises(ValueError, match='QA data: duplicate question_id'):
        human_eval.build('B2')


@pytest.mark.parametrize('limit', [0, -1, 1.5, True])
def test_invalid_limit_fails(tmp_path, monkeypatch, limit):
    monkeypatch.setattr(human_eval, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='limit must be a positive integer'):
        human_eval.build('B2', limit=limit)


def test_empty_eligible_pool_has_clear_error(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, [])
    with pytest.raises(ValueError, match='No answerable test questions'):
        human_eval.build('B2')


def test_only_answerable_test_questions_are_sampled(tmp_path, monkeypatch):
    gold = _dataset(movies=1)
    gold[0]['split'] = 'train'
    gold[1]['is_answerable'] = False
    _setup(tmp_path, monkeypatch, gold)
    manifest = human_eval.build('B2')
    assert manifest['eligible_questions'] == len(gold) - 2
    assert gold[0]['question_id'] not in manifest['question_ids']
    assert gold[1]['question_id'] not in manifest['question_ids']
