from copy import deepcopy

import pytest

from kgqa import metrics
from kgqa.common import REFUSAL, read_json, write_json, write_jsonl


ENTITIES = {
    'Q1': {'label': 'Example Film', 'aliases': ['Example']},
    'Q2': {'label': 'Jane Director', 'aliases': ['J. Director']},
    'Q3': {'label': 'Other Director', 'aliases': []},
}


def gold_row(question_id='q1', **changes):
    row = {'question_id': question_id, 'movie_id': 'Q1', 'answer_ids': ['Q2'],
           'reference_answers': ['Example Film was directed by Jane Director.'],
           'relation_id': 'P57', 'split': 'test'}
    return {**row, **changes}


def prediction(question_id='q1', **changes):
    row = {'question_id': question_id, 'answer': 'Example Film was directed by Jane Director.',
           'answer_ids': ['Q2']}
    return {**row, **changes}


def test_perfect_scores_and_unchanged_latency_aggregation():
    gold = [gold_row(f'q{i}') for i in range(20)]
    predictions = [prediction(f'q{i}', latency_ms=i + 1) for i in range(20)]
    result = metrics.score_rows(gold, predictions, ENTITIES)
    assert result['count'] == 20
    assert result['entity_exact'] == result['entity_macro_f1'] == 100
    assert result['bleu'] == pytest.approx(100)
    assert result['rouge_l'] == pytest.approx(100)
    assert result['by_relation'] == {'P57': {'exact': 100, 'f1': 100, 'count': 20}}
    assert result['latency_mean_ms'] == 10.5
    assert result['latency_p95_ms'] == 20


def test_answerable_and_refusal_subsets_have_separate_denominators():
    gold = [gold_row(), gold_row('q2', answer_ids=[], reference_answers=[REFUSAL])]
    predictions = [prediction(answer_ids=['Q3']), prediction('q2', answer=REFUSAL, answer_ids=[])]
    result = metrics.score_rows(gold, predictions, ENTITIES)
    assert result['entity_exact'] == result['entity_macro_f1'] == 50
    assert result['answerable_count'] == result['refusal_count'] == 1
    assert result['answerable_entity_exact'] == result['answerable_entity_macro_f1'] == 0
    assert result['refusal_accuracy'] == 100
    assert 'latency_mean_ms' not in result


def test_entity_f1_counts_false_positive_ids():
    result = metrics.score_rows([gold_row()], [prediction(answer_ids=['Q2', 'Q3'])], ENTITIES)
    assert result['entity_exact'] == 0
    assert result['entity_macro_f1'] == pytest.approx(100 * 2 / 3)


def test_entity_extraction_and_best_reference_rouge():
    gold = [gold_row(reference_answers=['A different sentence.', prediction()['answer']])]
    result = metrics.score_rows(gold, [prediction(answer_ids=None)], ENTITIES)
    assert result['entity_exact'] == 100
    assert result['rouge_l'] == 100


@pytest.mark.parametrize(('gold', 'predictions', 'message'), [
    ([], [prediction()], 'empty gold'),
    ([gold_row()], [], 'empty predictions'),
    ([gold_row(), gold_row()], [prediction()], 'Duplicate gold'),
    ([gold_row()], [prediction(), prediction()], 'Duplicate prediction'),
    ([gold_row()], [prediction('unknown')], 'Unknown prediction'),
    ([gold_row(), gold_row('q2')], [prediction()], 'Missing predictions'),
    ([gold_row(question_id='')], [prediction()], 'nonempty question_id'),
    ([gold_row()], [prediction(question_id=None)], 'nonempty question_id'),
    ([gold_row(reference_answers=[])], [prediction()], 'at least one reference'),
])
def test_invalid_metric_inputs_fail_explicitly(gold, predictions, message):
    with pytest.raises(ValueError, match=message):
        metrics.score_rows(gold, predictions, ENTITIES)


def test_extra_predictions_require_explicit_subset_opt_in():
    predictions = [prediction(), prediction('q2')]
    with pytest.raises(ValueError, match='Unknown prediction'):
        metrics.score_rows([gold_row()], predictions, ENTITIES)
    result = metrics.score_rows([gold_row()], predictions, ENTITIES, allow_extra_predictions=True)
    assert result['count'] == 1
    assert result['entity_exact'] == 100
    with pytest.raises(ValueError, match='Duplicate prediction'):
        metrics.score_rows([gold_row()], predictions + [prediction('q2')], ENTITIES,
                           allow_extra_predictions=True)


def test_entity_dictionary_label_and_alias_edits_are_visible():
    entities = deepcopy(ENTITIES)
    assert metrics.entity_ids_from_text('Jane Director', entities) == ['Q2']
    entities['Q2']['label'] = 'Alice Director'
    entities['Q2']['aliases'].append('AD')
    assert metrics.entity_ids_from_text('Jane Director', entities) == []
    assert metrics.entity_ids_from_text('Alice Director and AD', entities) == ['Q2']
    entities['Q2']['aliases'].clear()
    assert metrics.entity_ids_from_text('AD', entities) == []


def test_entity_sources_do_not_share_implicit_cached_names():
    first = {'Q2': {'label': 'Jane Director'}}
    second = {'Q3': {'label': 'Jane Director'}}
    assert metrics.entity_ids_from_text('Jane Director', first) == ['Q2']
    assert metrics.entity_ids_from_text('Jane Director', second) == ['Q3']


def test_explicit_entity_name_index_is_a_reusable_snapshot():
    entities = deepcopy(ENTITIES)
    index = metrics.EntityNameIndex(entities)
    entities['Q2']['label'] = 'Alice Director'
    assert metrics.entity_ids_from_text('Jane Director', entities, index=index) == ['Q2']
    assert metrics.entity_ids_from_text('Jane Director', entities) == []


def test_entity_names_prefer_labels_long_matches_and_movie_exclusion():
    entities = {**deepcopy(ENTITIES), 'Q4': {'label': 'Director', 'aliases': ['Jane Director']}}
    assert metrics.entity_ids_from_text('Example Film was directed by Jane Director.', entities,
                                       excluded='Q1') == ['Q2']


@pytest.fixture
def evaluation_files(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, 'ROOT', tmp_path)
    write_json(tmp_path / 'data/processed/entities.json', ENTITIES)
    return tmp_path


def write_evaluation(root, gold, predictions, *, smoke=False, robustness=False):
    dataset = 'robustness' if robustness else 'qa'
    split = 'robustness' if robustness else 'test'
    write_jsonl(root / f'data/processed/{dataset}.jsonl', gold)
    write_jsonl(root / f'runs/run/predictions_{split}.jsonl', predictions)
    write_json(root / 'runs/run/config.json', {'smoke': smoke})


def test_evaluate_full_split_and_saved_coverage(evaluation_files):
    root = evaluation_files
    write_evaluation(root, [gold_row(), gold_row('train', split='train')], [prediction()])
    result = metrics.evaluate('run')
    assert result['gold_total'] == result['count'] == 1
    assert result['prediction_coverage'] == 100
    assert read_json(root / 'runs/run/metrics_test.json') == result


def test_evaluate_smoke_subset_reports_partial_coverage(evaluation_files):
    write_evaluation(evaluation_files, [gold_row(), gold_row('q2')], [prediction()], smoke=True)
    result = metrics.evaluate('run')
    assert result['count'] == 1
    assert result['gold_total'] == 2
    assert result['prediction_coverage'] == 50


def test_evaluate_incomplete_non_smoke_run_fails(evaluation_files):
    write_evaluation(evaluation_files, [gold_row(), gold_row('q2')], [prediction()])
    with pytest.raises(ValueError, match='coverage is incomplete: 1/2'):
        metrics.evaluate('run')
    assert not (evaluation_files / 'runs/run/metrics_test.json').exists()


@pytest.mark.parametrize(('gold', 'predictions', 'message'), [
    ([gold_row(), gold_row('q2')], [prediction(), prediction()], 'Duplicate prediction'),
    ([gold_row(), gold_row('q2')], [prediction(), prediction('unknown')], 'Unknown prediction'),
    ([gold_row(), gold_row()], [prediction()], 'Duplicate gold'),
    ([gold_row()], [], 'empty predictions'),
    ([gold_row(split='train')], [prediction()], 'empty gold'),
])
def test_smoke_filtering_cannot_hide_invalid_input(evaluation_files, gold, predictions, message):
    write_evaluation(evaluation_files, gold, predictions, smoke=True)
    with pytest.raises(ValueError, match=message):
        metrics.evaluate('run')
    assert not (evaluation_files / 'runs/run/metrics_test.json').exists()


def test_robustness_scores_each_variant_after_full_validation(evaluation_files):
    gold = [gold_row(variant='paraphrase'), gold_row('q2', variant='typo')]
    write_evaluation(evaluation_files, gold, [prediction(), prediction('q2')], robustness=True)
    result = metrics.evaluate_robustness('run')
    assert set(result) == {'paraphrase', 'typo'}
    assert all(row['count'] == 1 and row['entity_exact'] == 100 for row in result.values())
    assert read_json(evaluation_files / 'runs/run/metrics_robustness.json') == result


@pytest.mark.parametrize(('gold', 'predictions', 'message'), [
    ([gold_row(variant='typo')], [prediction(), prediction()], 'Duplicate prediction'),
    ([gold_row(variant='typo')], [prediction(), prediction('unknown')], 'Unknown prediction'),
    ([gold_row(variant='typo'), gold_row('q2', variant='paraphrase')], [prediction()], 'Missing predictions'),
    ([], [prediction()], 'empty gold'),
    ([gold_row(variant='typo')], [], 'empty predictions'),
])
def test_robustness_rejects_invalid_whole_files(evaluation_files, gold, predictions, message):
    write_evaluation(evaluation_files, gold, predictions, robustness=True)
    with pytest.raises(ValueError, match=message):
        metrics.evaluate_robustness('run')
    assert not (evaluation_files / 'runs/run/metrics_robustness.json').exists()
