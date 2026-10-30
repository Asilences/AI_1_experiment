"""Context resolves homonyms without suppressing generated false positives."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kgqa.common import RELATIONS
from kgqa.metrics import AnswerEntityIndex, EntityNameIndex, entity_ids_from_text
from kgqa.common import answer_text, REFUSAL
from kgqa.neural_preflight import _facts, FACT_LIMIT


ENTITIES = {
    'country': {'label': 'Brazil'},
    'film': {'label': 'Brazil'},
    'other': {'label': 'Unique Hallucination'},
    'source': {'label': 'Source', 'facts': {'P495': ['country']}},
    'linked': {'label': 'Linked', 'facts': {'P57': ['film']}},
}


def neural_classes():
    tree = ast.parse((Path(__file__).parents[1] / 'kgqa/neural.py').read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)
             and node.name in ('GraphData', 'QADataset')]
    namespace = dict(Dataset=object, RELATIONS=RELATIONS, np=np, _facts=_facts,
                     FACT_LIMIT=FACT_LIMIT, ROOT=Path('.'),
                     read_json=lambda path: {'test': ['linked']},
                     EntityLinker=lambda *args: SimpleNamespace(link=lambda question: ('linked', 1)))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<neural classes>', 'exec'), namespace)
    return namespace['GraphData'], namespace['QADataset']


def test_legacy_default_and_context_fallback():
    index = EntityNameIndex(ENTITIES)
    assert index.match('Brazil') == ['country', 'film']
    assert index.match('Brazil', preferred_ids=['country']) == ['country']
    assert index.match('Brazil', preferred_ids=['film']) == ['film']
    assert index.match('Brazil', preferred_ids=['other']) == ['country', 'film']
    assert index.match('Brazil', preferred_ids=[]) == ['country', 'film']
    assert index.match('Brazil', excluded='film', preferred_ids=['film']) == ['country']


def test_unique_hallucinations_remain_and_wrapper_passes_context():
    assert entity_ids_from_text('Brazil and Unique Hallucination', ENTITIES,
                               preferred_ids=['country']) == ['country', 'other']


@pytest.mark.parametrize('embeddings', [False, True])
def test_tail_ids_share_fact_filter_order_and_limit(embeddings):
    GraphData, _ = neural_classes()
    graph = GraphData.__new__(GraphData)
    graph.entities = {f't{i}': {'label': f'Tail {i}'} for i in range(30)}
    graph.entities.update(source={'label': 'Source', 'facts': {
        'P57': ['missing', 'blank'] + [f't{i}' for i in range(30)], 'P495': ['t0']}},
        blank={'label': ''})
    graph.use_embeddings = embeddings
    graph.entity_to_id = {qid: i for i, qid in enumerate(graph.entities) if qid != 't2'}
    graph.relation_to_id = {'P57': 0, 'P495': 1}
    graph.entity = np.zeros((len(graph.entities), 3)); graph.relation = np.zeros((2, 3))
    ids = graph.fact_tail_ids('source')
    assert len(ids) == 24
    assert ('t2' in ids) is not embeddings
    assert [text for text, _ in graph.facts('source')] == [
        f'Source | director | {graph.entities[tail]["label"]}' for tail in ids]
    assert graph.fact_tail_ids(None) == []


def test_dataset_context_uses_actual_linker_without_answer_or_relation():
    GraphData, QADataset = neural_classes()
    graph = GraphData.__new__(GraphData)
    graph.data_dir = Path('.') / 'data/processed'
    graph.entities = ENTITIES; graph.use_embeddings = False
    # No gold answers/relation are available to context construction.
    rows = [{'question_id': 'q', 'question': 'Which?', 'movie_id': 'source'}]
    linked = QADataset(rows, graph, link_questions=True)
    assert linked.linked_movie_ids == ['linked']
    assert linked.name_context == {'q': ['film']}
    assert QADataset(rows, graph).name_context == {'q': ['country']}


def test_both_generation_paths_pass_dataset_context():
    tree = ast.parse((Path(__file__).parents[1] / 'kgqa/neural.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('predictions', 'validation_outputs'):
            calls = [call for call in ast.walk(node) if isinstance(call, ast.Call)
                     and isinstance(call.func, ast.Name) and call.func.id == 'entity_ids_from_text']
            assert len(calls) == 1
            context = next(kw.value for kw in calls[0].keywords if kw.arg == 'preferred_ids')
            assert ast.unparse(context) == "loader.dataset.name_context[row['question_id']]"


@pytest.mark.parametrize('relation', list(RELATIONS))
@pytest.mark.parametrize('variant', [0, 1])
def test_answer_templates_strip_only_head_and_connectors(relation, variant):
    entities = {**ENTITIES, 'head2': {'label': 'Brazil'},
                'india': {'label': 'India', 'aliases': ['IN']},
                'belarus': {'label': 'Belarus', 'aliases': ['BY']},
                'andorra': {'label': 'Andorra', 'aliases': ['AND']},
                'germany': {'label': 'Germany'}, 'director': {'label': 'Are Director'}}
    idx = AnswerEntityIndex(entities)
    text = answer_text('Brazil', relation, ['Are Director', 'Germany'], variant)
    assert idx.match(text, 'film', ['director', 'germany']) == ['director', 'germany']
    assert idx.match('BY and IN AND') == ['andorra', 'belarus', 'india']
    assert idx.match('by and in') == []
    assert idx.match(REFUSAL) == []
    assert idx.match('Are Director and Unique Hallucination', preferred_ids=['director']) == ['director', 'other']


def test_titles_with_are_and_same_head_tail_and_wrong_director():
    entities = {**ENTITIES, 'a': {'label': 'Australia'}, 'a2': {'label': 'Australia'},
                'h': {'label': 'We Are Here'}, 'd': {'label': 'Are Director'},
                'g': {'label': 'Germany'}, 'wrong': {'label': 'Wrong Director'}}
    idx = AnswerEntityIndex(entities)
    assert idx.match('The director or directors of We Are Here are Are Director.', 'h') == ['d']
    assert idx.match('Australia originated in Australia.', 'a2', ['a']) == ['a']
    assert idx.match('Brazil originated in Brazil.', 'source', ['country']) == ['country']
    assert idx.match('Brazil was directed by Wrong Director and Germany.', 'film', ['d']) == ['g', 'wrong']


def test_canonical_labels_and_noncode_aliases_are_case_insensitive():
    idx = AnswerEntityIndex({'a': {'label': 'IN'}, 'b': {'label': 'Person', 'aliases': ['Nickname']},
                             'c': {'label': 'Country', 'aliases': ['BY']}})
    assert idx.match('in nickname by') == ['a', 'b']
    assert idx.match('BY') == ['c']


def test_accelerated_fragment_matches_legacy_overlap_semantics():
    entities = {**ENTITIES, 'long': {'label': 'Unique Hallucination Extended'},
                'alias': {'label': 'Another Person', 'aliases': ['Director Name']}}
    fast, legacy = AnswerEntityIndex(entities), EntityNameIndex(entities)
    for text in ('Brazil', 'Brazil Brazil Unique Hallucination', 'Director Name',
                 'Unique Hallucination Extended, Brazil', 'unknown', 'Another Person and Brazil'):
        for preferred in (None, ['country'], ['film'], ['other']):
            assert fast.match(text, preferred_ids=preferred) == legacy.match(text, preferred_ids=preferred)


def test_short_alias_case_and_list_punctuation_do_not_shadow_entities():
    entities = {'and': {'label': 'Kinds of Kindness', 'aliases': ['And']},
                'or': {'label': 'Other Film', 'aliases': ['Or']},
                'usa': {'label': 'United States', 'aliases': ['USA']},
                'black': {'label': 'black comedy'}, 'sf': {'label': 'science fiction film'},
                'cross': {'label': 'science fiction comedy', 'aliases': ['comedy science fiction']},
                'country': {'label': 'Trinidad and Tobago'},
                'comma': {'label': 'Name, With Comma'}, 'extra': {'label': 'Germany'}}
    index = AnswerEntityIndex(entities)
    assert index.match('black comedy, science fiction film and Germany') == ['black', 'extra', 'sf']
    assert index.match('black comedy; science fiction film') == ['black', 'sf']
    assert index.match('And Or USA') == ['and', 'or', 'usa']
    assert index.match('and or usa') == []
    assert index.match('Trinidad and Tobago') == ['country']
    assert index.match('Name, With Comma') == ['comma']


@pytest.mark.parametrize('separator', [' and ', ' or ', ' AND ', ' OR ', ' & ', ', ', '; ', ', and '])
def test_contextual_individuals_cover_collective_name(separator):
    text = f'Alice Example{separator}Bob Example'
    entities = {'alice': {'label': 'Alice Example'}, 'bob': {'label': 'Bob Example'},
                'collective': {'label': text}, 'unrelated': {'label': 'Unique Hallucination'}}
    index = AnswerEntityIndex(entities)
    assert index.match(text, preferred_ids=['alice', 'bob']) == ['alice', 'bob']
    assert index.match(text) == ['collective']
    assert index.match(text, preferred_ids=['collective']) == ['collective']
    assert index.match(text, preferred_ids=['alice', 'bob', 'collective']) == ['collective']
    assert index.match(text + ' and Unique Hallucination', preferred_ids=['alice', 'bob']) == [
        'alice', 'bob', 'unrelated']
    # The compatibility index keeps its original longest-name behavior.
    assert EntityNameIndex(entities).match(text, preferred_ids=['alice', 'bob']) == ['collective']


@pytest.mark.parametrize('text, preferred', [
    ('Alice Example and Bob Example', ['alice']),
    ('Alice Example and Bob Example and Extra Person', ['alice', 'bob']),
    ('Prefix Alice Example and Bob Example', ['alice', 'bob']),
    ('Alice Example near Bob Example', ['alice', 'bob']),
    ('Alice Example Bob Example', ['alice', 'bob']),
    ('Alice Example and Alice Example', ['alice', 'bob']),
])
def test_incomplete_or_nonlist_context_does_not_suppress_collective(text, preferred):
    entities = {'alice': {'label': 'Alice Example'}, 'bob': {'label': 'Bob Example'},
                'collective': {'label': text}}
    assert AnswerEntityIndex(entities).match(text, preferred_ids=preferred) == ['collective']
