from copy import deepcopy

import pytest

from kgqa.planning import PLAN_SCHEMA, PlanValidationError, validate_plan


@pytest.fixture
def plan():
    return {'schema': 'kgqa-plan/1', 'language': 'zh', 'status': 'ready',
            'movie_ids': ['Q1'], 'properties': ['P161'], 'operation': 'facts',
            'unhandled_constraints': [], 'clarification': None, 'limit': 30}


def test_candidate_membership_is_separate_from_catalogue_membership(plan):
    assert validate_plan(plan, {'Q1'}, catalogue_movie_ids={'Q1', 'Q2'}) == plan
    with pytest.raises(PlanValidationError, match='allowed candidates'):
        validate_plan(plan, {'Q2'}, catalogue_movie_ids={'Q1', 'Q2'})
    with pytest.raises(PlanValidationError, match='frozen catalogue'):
        validate_plan(plan, {'Q1'}, catalogue_movie_ids={'Q2'})


def test_validation_does_not_mutate_or_alias_input(plan):
    original = deepcopy(plan)
    result = validate_plan(plan, ['Q1'])
    result['movie_ids'].append('Q2')
    assert plan == original


@pytest.mark.parametrize('field,value', [
    ('schema', 'kgqa-plan/2'), ('language', 'fr'), ('status', 'execute'),
    ('movie_ids', 'Q1'), ('movie_ids', [1]), ('movie_ids', ['Q1', 'Q1']),
    ('movie_ids', ['Q01']), ('movie_ids', []), ('properties', ['P999']),
    ('properties', ['P161', 'P161']), ('properties', []),
    ('operation', 'python'), ('operation', {'eval': 'x'}),
    ('unhandled_constraints', 'before 2010'), ('unhandled_constraints', ['']),
    ('unhandled_constraints', [7]), ('unhandled_constraints', ['before 2010']),
    ('clarification', ''), ('clarification', 'Which year?'),
    ('limit', True), ('limit', 0), ('limit', 101), ('limit', '3'),
])
def test_invalid_fields_and_unexecuted_conditions_fail(plan, field, value):
    plan[field] = value
    with pytest.raises(PlanValidationError):
        validate_plan(plan, ['Q1'])


def test_unknown_or_missing_fields_are_not_silently_discarded(plan):
    plan['conditions'] = [{'year_after': 2010}]
    with pytest.raises(PlanValidationError, match='fields'):
        validate_plan(plan, ['Q1'])
    del plan['conditions']
    del plan['limit']
    with pytest.raises(PlanValidationError, match='fields'):
        validate_plan(plan, ['Q1'])


def test_nonready_plan_retains_conditions_and_can_omit_entity(plan):
    plan.update(status='unsupported', movie_ids=[], properties=[],
                unhandled_constraints=['after 2010'], clarification='Date filtering is not implemented.')
    assert validate_plan(plan, [])['unhandled_constraints'] == ['after 2010']
    plan.update(status='needs_clarification', clarification=None, unhandled_constraints=[])
    with pytest.raises(PlanValidationError, match='explain'):
        validate_plan(plan, [])


def test_operation_and_property_type_contract(plan):
    plan.update(operation='intersection', movie_ids=['Q1', 'Q2'])
    assert validate_plan(plan, ['Q1', 'Q2'])['operation'] == 'intersection'
    plan['properties'] = ['P577']
    with pytest.raises(PlanValidationError, match='entity-valued'):
        validate_plan(plan, ['Q1', 'Q2'])
    plan.update(operation='reverse', movie_ids=['Q1'], properties=['P161'])
    with pytest.raises(PlanValidationError, match='director'):
        validate_plan(plan, ['Q1'])
    plan['properties'] = ['P57']
    assert validate_plan(plan, ['Q1'])['operation'] == 'reverse'
    plan.update(operation='compare', movie_ids=['Q1', 'Q2'], properties=['P2142', 'P2130'])
    with pytest.raises(PlanValidationError, match='one amount'):
        validate_plan(plan, ['Q1', 'Q2'])
    plan['properties'] = ['P2142']
    assert validate_plan(plan, ['Q1', 'Q2'])['operation'] == 'compare'
    plan['movie_ids'] = ['Q1']
    with pytest.raises(PlanValidationError, match='two movies'):
        validate_plan(plan, ['Q1'])


def test_lookup_budget_is_enforced_before_execution(plan):
    plan.update(movie_ids=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                properties=['P57', 'P136', 'P495', 'P364', 'P161'])
    with pytest.raises(PlanValidationError, match='budget'):
        validate_plan(plan, plan['movie_ids'])


def test_exported_schema_has_no_unbounded_or_arbitrary_tool_fields(plan):
    assert set(PLAN_SCHEMA['required']) == set(plan)
    assert PLAN_SCHEMA['additionalProperties'] is False
    assert 'python' not in PLAN_SCHEMA['properties']['operation']['enum']
