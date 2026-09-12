from kgqa.common import answer_text
from kgqa.linker import EntityLinker, normalize
from kgqa.metrics import score_rows


ENTITIES = {
    'Q1': {'label': 'Example Film', 'aliases': ['Example'], 'facts': {'P57': ['Q2']}},
    'Q2': {'label': 'Jane Director', 'aliases': []},
}


def test_linker_exact_and_answer():
    linker = EntityLinker(['Q1'], ENTITIES)
    assert linker.link('Who directed Example Film?')[0] == 'Q1'
    assert answer_text('Example Film', 'P57', ['Jane Director']) == 'Example Film was directed by Jane Director.'


def test_metrics_entity_set():
    gold = [{'question_id': 'x', 'movie_id': 'Q1', 'answer_ids': ['Q2'], 'reference_answers': ['Example Film was directed by Jane Director.'], 'relation_id': 'P57'}]
    predictions = [{'question_id': 'x', 'answer': 'Example Film was directed by Jane Director.', 'answer_ids': ['Q2'], 'latency_ms': 1}]
    result = score_rows(gold, predictions, ENTITIES)
    assert result['entity_exact'] == 100
    assert result['entity_macro_f1'] == 100


def test_normalize():
    assert normalize('Spider-Man: Homecoming') == 'spider man homecoming'
