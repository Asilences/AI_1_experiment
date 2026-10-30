import pytest

from kgqa.freeform import FreeformParser, answer
from kgqa.common import REFUSAL


@pytest.fixture
def parser():
    entities = {
        'Q1': {'label': 'The Matrix', 'aliases': ['黑客帝国', '駭客任務'], 'facts': {'P57': ['D'], 'P136': ['G'], 'P495': ['C'], 'P364': ['L']}},
        'Q2': {'label': 'Inception', 'aliases': ['盗梦空间', '盜夢空間'], 'facts': {'P57': ['D'], 'P136': ['G'], 'P364': ['L']}},
        'Q3': {'label': 'Other Film', 'aliases': ['同名电影'], 'facts': {'P57': ['D']}},
        'Q4': {'label': 'Another Film', 'aliases': ['同名电影'], 'facts': {'P57': ['D']}},
        'Q5': {'label': 'The Last Language', 'facts': {'P57': ['D']}},
    }
    return FreeformParser(list(entities), entities)


@pytest.mark.parametrize('question,movie,relation', [
    ('Could you tell me who helmed The Matrix?', 'Q1', 'P57'),
    ('The Matrix — name the filmmaker, please.', 'Q1', 'P57'),
    ('Which languages do people speak in Inception?', 'Q2', 'P364'),
    ('What kind of film is The Matrix?', 'Q1', 'P136'),
    ('Where is The Matrix from?', 'Q1', 'P495'),
    ('《盗梦空间》是谁执导的？', 'Q2', 'P57'),
    ('黑客帝国是什么类型的电影？', 'Q1', 'P136'),
    ('The Matrix的导演是谁？', 'Q1', 'P57'),
    ('駭客任務是哪個國家的電影？', 'Q1', 'P495'),
    ('盜夢空間的原始語言是什麼？', 'Q2', 'P364'),
    ('q1的导演是谁？', 'Q1', 'P57'),
    ('Who directed The Last Language?', 'Q5', 'P57'),
])
def test_supported_natural_questions(parser, question, movie, relation):
    parsed = parser.parse(question)
    assert parsed['status'] == 'ready' and parsed['movie_id'] == movie and parsed['relation_id'] == relation
    assert parsed['canonical_question'] and parsed['method'] == 'bounded_rules_v1'


@pytest.mark.parametrize('question,status', [
    ('Who directed The Matrix and Inception?', 'needs_clarification'),
    ('同名电影是谁导演的？', 'needs_clarification'),
    ('The Matrix的导演和类型分别是什么？', 'needs_clarification'),
    ('Tell me about The Matrix', 'needs_clarification'),
    ('Is The Matrix a comedy?', 'needs_clarification'),
    ('盗梦空间是不是科幻片？', 'needs_clarification'),
    ('The Matrix在哪拍的？', 'unanswerable'),
    ('Who directed and wrote The Matrix?', 'unanswerable'),
    ('What is the budget of Inception?', 'unanswerable'),
    ('Who directed Zyxwuv Qpkjrt?', 'unanswerable'),
    ('Who directed Q999999999?', 'unanswerable'),
    ('黑客帝国999是谁执导的？', 'unanswerable'),
    ('What country is Inception from?', 'unanswerable'),
    ('Who directed 黑客帝国 (The Matrix) and Inception?', 'needs_clarification'),
    ('Which countries banned The Matrix?', 'unanswerable'),
    ('What language is The Matrix dubbed in?', 'unanswerable'),
    ('What country is The Matrix set in?', 'unanswerable'),
    ('黑客帝国在哪些国家被禁映？', 'unanswerable'),
    ('盗梦空间的配音语言是什么？', 'unanswerable'),
])
def test_ambiguous_unsupported_or_missing_facts_do_not_reach_model(parser, question, status):
    parsed = parser.parse(question)
    assert parsed['status'] == status and parsed['message']


def test_qid_can_disambiguate_a_shared_title(parser):
    assert parser.parse('Q3 的导演是谁？')['movie_id'] == 'Q3'
    assert parser.parse('Q3: 同名电影的导演是谁？')['movie_id'] == 'Q3'
    assert parser.parse('Who directed The Matrix Q2?')['status'] == 'needs_clarification'


@pytest.mark.parametrize('question', ['', ' ' * 4, None, 'x' * 1001])
def test_invalid_inputs(parser, question):
    with pytest.raises(ValueError):
        parser.parse(question)


def test_real_5000_a0_understands_adapter_output_without_retraining():
    result = answer('Could you tell me who helmed The Avengers?')
    assert result['interpretation']['status'] == 'ready'
    assert result['result']['answer'] == 'The Avengers was directed by Joss Whedon.'
    assert result['result']['answer'] != REFUSAL
