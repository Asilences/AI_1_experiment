import pytest

from kgqa.planning import PlanValidationError
from test_explorer import explorer as explorer


def typed_plan(**changes):
    value = {'schema': 'kgqa-plan/1', 'language': 'en', 'status': 'ready',
             'movie_ids': ['Q1'], 'properties': ['P161'], 'operation': 'facts',
             'unhandled_constraints': [], 'clarification': None, 'limit': 30}
    value.update(changes)
    return value


def test_semantic_plan_executes_without_reparsing_natural_question(explorer, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('A validated semantic plan must never re-enter rule parsing')
    monkeypatch.setattr(explorer, 'plan', forbidden)
    result = explorer.execute(typed_plan(properties=['P57', 'P161']), allowed_movie_ids=['Q1'],
                              question='Who was at the creative helm, and who appears on screen?')
    assert result['status'] == 'answered'
    assert {e['property'] for e in result['evidence']} == {'P57', 'P161'}
    assert result['plan']['question'].startswith('Who was at the creative helm')
    assert result['validated_plan']['schema'] == 'kgqa-plan/1'


def test_plan_limit_paginates_after_full_set_computation(explorer):
    plan = typed_plan(operation='reverse', properties=['P57'], limit=1)
    result = explorer.execute(plan, allowed_movie_ids=['Q1'])
    assert result['total'] == 2 and result['next_offset'] == 1 and len(result['items']) == 1
    page2 = explorer.execute(plan, offset=1, allowed_movie_ids=['Q1'])
    assert page2['total'] == 2 and page2['next_offset'] is None
    assert page2['items'][0]['movie_id'] != result['items'][0]['movie_id']
    assert all(item['via'] == 'Q4' for item in [*result['items'], *page2['items']])
    assert {'Q1', page2['items'][0]['movie_id']} == {e['movie_id'] for e in page2['evidence']}


def test_semantic_intersection_keeps_both_evidence_sides(explorer):
    result = explorer.execute(typed_plan(operation='intersection', movie_ids=['Q1', 'Q2']),
                              allowed_movie_ids=['Q1', 'Q2'])
    assert result['items'] == [{'property': 'P161', 'entity_id': 'Q5', 'label': 'Actor One'}]
    assert {e['movie_id'] for e in result['evidence']} == {'Q1', 'Q2'}


def test_semantic_amount_comparison_cannot_override_scope_policy(explorer):
    result = explorer.execute(typed_plan(operation='compare', movie_ids=['Q1', 'Q2'], properties=['P2142']),
                              allowed_movie_ids=['Q1', 'Q2'])
    assert result['status'] == 'not_comparable'
    assert 'comparison' not in result


@pytest.mark.parametrize('status', ['unsupported', 'needs_clarification'])
def test_blocked_plan_never_reads_graph_or_returns_facts(explorer, status):
    class NoGraphReads(dict):
        def __getitem__(self, key):
            raise AssertionError('A blocked plan must not read graph facts')
    explorer.by_pair = NoGraphReads()
    plan = typed_plan(status=status, unhandled_constraints=['after 2010'], clarification='Please specify a supported scope.')
    result = explorer.execute(plan, allowed_movie_ids=['Q1'])
    assert result['status'] == status and not result['evidence'] and not result['items']
    assert result['answer'] == 'Please specify a supported scope.'


def test_candidate_qid_validation_cannot_be_bypassed_by_catalogue_membership(explorer):
    with pytest.raises(PlanValidationError, match='allowed candidates'):
        explorer.execute(typed_plan(movie_ids=['Q2']), allowed_movie_ids=['Q1'])


def test_semantic_partial_results_preserve_missing_attributes(explorer):
    result = explorer.execute(typed_plan(properties=['P57', 'P58']), allowed_movie_ids=['Q1'])
    assert result['status'] == 'partial'
    assert result['missing'] == [{'movie_id': 'Q1', 'property': 'P58'}]
    assert len(result['evidence']) == 1


@pytest.mark.parametrize('offset,limit', [(True, None), (-1, None), (0, True), (0, 101)])
def test_execute_page_validation(explorer, offset, limit):
    with pytest.raises(ValueError):
        explorer.execute(typed_plan(), offset=offset, limit=limit, allowed_movie_ids=['Q1'])
