"""A small, typed query-plan contract for frozen-model movie QA.

Validation establishes the plan's executable scope, not that it faithfully
interprets the user's natural-language question. The caller must bind IDs to
the current question's candidates and separately check unresolved conditions.
"""
from __future__ import annotations

from copy import deepcopy
import re

from .v2_data import PROPERTIES

SCHEMA = 'kgqa-plan/1'
OPERATIONS = ('facts', 'intersection', 'reverse', 'compare')
STATUSES = ('ready', 'needs_clarification', 'unsupported')
MAX_MOVIES = 5
MAX_LOOKUPS = 20

PLAN_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['schema', 'language', 'status', 'movie_ids', 'properties',
                 'operation', 'unhandled_constraints', 'clarification', 'limit'],
    'properties': {
        'schema': {'type': 'string', 'enum': [SCHEMA]},
        'language': {'type': 'string', 'enum': ['zh', 'en']},
        'status': {'type': 'string', 'enum': list(STATUSES)},
        'movie_ids': {'type': 'array', 'maxItems': MAX_MOVIES, 'uniqueItems': True,
                      'items': {'type': 'string', 'pattern': '^Q[1-9][0-9]*$'}},
        'properties': {'type': 'array', 'maxItems': len(PROPERTIES), 'uniqueItems': True,
                       'items': {'type': 'string', 'enum': list(PROPERTIES)}},
        'operation': {'type': 'string', 'enum': list(OPERATIONS)},
        'unhandled_constraints': {'type': 'array', 'maxItems': 20,
                                  'items': {'type': 'string', 'minLength': 1, 'maxLength': 1000}},
        'clarification': {'type': ['string', 'null'], 'minLength': 1, 'maxLength': 1000},
        'limit': {'type': 'integer', 'minimum': 1, 'maximum': 100},
    },
}


class PlanValidationError(ValueError):
    """The proposed plan exceeds the typed, candidate-bound tool contract."""


def _ids(value, label):
    if not isinstance(value, (list, tuple, set, frozenset, dict)):
        raise PlanValidationError(f'{label} must be a collection of movie QIDs')
    if any(not isinstance(qid, str) or not re.fullmatch(r'Q[1-9][0-9]*', qid) for qid in value):
        raise PlanValidationError(f'{label} must contain movie QIDs')
    return set(value)


def validate_plan(plan, allowed_movie_ids, *, catalogue_movie_ids=None):
    """Return a copied plan after strict scope/type validation.

    ``allowed_movie_ids`` is mandatory: use only this question's recalled or
    confirmed candidates, not every movie in the catalogue. Membership alone
    does not authorize choosing an ambiguous candidate without confirmation.
    """
    if not isinstance(plan, dict):
        raise PlanValidationError('Plan must be an object')
    required = set(PLAN_SCHEMA['required'])
    if set(plan) != required:
        raise PlanValidationError('Plan fields must exactly match kgqa-plan/1')
    if plan['schema'] != SCHEMA:
        raise PlanValidationError('Unsupported plan schema')
    if plan['language'] not in ('zh', 'en') or plan['status'] not in STATUSES:
        raise PlanValidationError('Unsupported plan language or status')
    if not isinstance(plan['operation'], str) or plan['operation'] not in OPERATIONS:
        raise PlanValidationError('Unsupported operation')
    allowed = _ids(allowed_movie_ids, 'allowed_movie_ids')
    catalogue = _ids(catalogue_movie_ids, 'catalogue_movie_ids') if catalogue_movie_ids is not None else None
    ids, props, constraints = plan['movie_ids'], plan['properties'], plan['unhandled_constraints']
    for values, label, maximum in ((ids, 'movie_ids', MAX_MOVIES), (props, 'properties', len(PROPERTIES))):
        if not isinstance(values, list) or len(values) > maximum or any(not isinstance(v, str) for v in values):
            raise PlanValidationError(f'{label} must be a bounded string list')
        if len(set(values)) != len(values):
            raise PlanValidationError(f'{label} must not contain duplicates')
    if any(not re.fullmatch(r'Q[1-9][0-9]*', qid) for qid in ids) or not set(ids).issubset(allowed):
        raise PlanValidationError('Movie IDs must come from this question\'s allowed candidates')
    if catalogue is not None and not set(ids).issubset(catalogue):
        raise PlanValidationError('Movie ID is not in the frozen catalogue')
    if any(prop not in PROPERTIES for prop in props):
        raise PlanValidationError('Unsupported property')
    if (not isinstance(constraints, list) or len(constraints) > 20 or
            any(not isinstance(c, str) or not c.strip() or len(c) > 1000 for c in constraints)):
        raise PlanValidationError('unhandled_constraints must be a bounded nonempty-text list')
    clarification = plan['clarification']
    if clarification is not None and (not isinstance(clarification, str) or not clarification.strip() or len(clarification) > 1000):
        raise PlanValidationError('clarification must be null or bounded nonempty text')
    limit = plan['limit']
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
        raise PlanValidationError('limit must be an integer from 1 to 100')
    if plan['status'] != 'ready':
        if clarification is None and not constraints:
            raise PlanValidationError('A blocked plan must explain its unresolved scope')
        return deepcopy(plan)
    if constraints or clarification is not None:
        raise PlanValidationError('A ready plan must not contain unresolved conditions or clarification')
    if not ids or not props:
        raise PlanValidationError('A ready plan needs movie IDs and properties')
    if len(ids) * len(props) > MAX_LOOKUPS:
        raise PlanValidationError('Plan exceeds the property-lookup budget')
    op = plan['operation']
    if op in ('intersection', 'compare') and len(ids) != 2:
        raise PlanValidationError(f'{op} requires exactly two movies')
    if op == 'intersection' and any(PROPERTIES[prop][2] != 'entity' for prop in props):
        raise PlanValidationError('intersection requires entity-valued properties')
    if op == 'reverse' and (len(ids) != 1 or props != ['P57']):
        raise PlanValidationError('reverse supports one movie -> director -> other catalogue movies only')
    if op == 'compare' and any(PROPERTIES[prop][2] == 'quantity' for prop in props) and len(props) != 1:
        raise PlanValidationError('Numeric comparison requires exactly one amount property')
    return deepcopy(plan)
