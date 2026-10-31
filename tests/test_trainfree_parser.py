"""Safety regressions for the train-free adapter, using synthetic film facts."""
import pytest

from kgqa.freeform import FreeformParser


@pytest.fixture
def parser():
    entities = {
        'Q101': {'label': 'Silver Orchard', 'label_zh': '银色果园', 'release_year': 2001,
                 'facts': {'P57': ['D1'], 'P136': ['G1'], 'P495': ['C1'], 'P364': ['L1']}},
        'Q102': {'label': 'Music', 'facts': {'P57': ['D2']}},
        'Q103': {'label': 'After Midnight', 'facts': {'P57': ['D2']}},
        'Q104': {'label': 'Twin Title', 'aliases': ['双生'], 'release_year': 1997, 'facts': {'P57': ['D1']}},
        'Q105': {'label': 'Twin Title', 'aliases': ['双生'], 'release_year': 2017, 'facts': {'P57': ['D2']}},
    }
    return FreeformParser(list(entities), entities)


@pytest.mark.parametrize('question', [
    'Which director did not direct Silver Orchard?',
    'Name the directors of Silver Orchard excluding D1.',
    'Who directed Silver Orchard before 2010?',
    'Who directed Silver Orchard after 1990?',
    'Which director directed Silver Orchard during 2001?',
    'Which director of Silver Orchard is older than 60?',
    'Who is the first director of Silver Orchard?',
    'Only list the American directors of Silver Orchard.',
    'Compare the directors of Silver Orchard.',
    'How many directors does Silver Orchard have?',
    'Silver Orchard不是由谁执导的？',
    'Silver Orchard的导演里有哪些法国人？',
    'Silver Orchard在2000年以前的导演是谁？',
    'Who directed Silver Orchard with a French nationality?',
    'Where in Europe is Silver Orchard from?',
    'What kind of French film is Silver Orchard?',
])
def test_unimplemented_constraints_never_become_unconditional_lookup(parser, question):
    parsed = parser.parse(question)
    assert parsed['status'] != 'ready'
    assert parsed['canonical_question'] is None


@pytest.mark.parametrize('question', [
    'What languages are spoken in Silver Orchard?',
    'Which language do people speak in Silver Orchard?',
    'What is the dialogue language in Silver Orchard?',
    'Who was behind the camera on Silver Orchard?',
    '银色果园的对白语言是什么？',
])
def test_semantically_ambiguous_properties_require_clarification(parser, question):
    parsed = parser.parse(question)
    assert parsed['status'] == 'needs_clarification'
    assert parsed['reason_code'] == 'ambiguous_property_scope'
    assert parsed['canonical_question'] is None


@pytest.mark.parametrize('question,movie', [
    ('Who holds the directorial credit for Silver Orchard?', 'Q101'),
    ('Who directed After Midnight?', 'Q103'),
    ('银色果园的导演是谁？', 'Q101'),
    ('《双生》2017年的导演是谁？', 'Q105'),
    ('《Silver Orchard》的导演是谁？', 'Q101'),
])
def test_known_semantics_title_words_and_chinese_years(parser, question, movie):
    parsed = parser.parse(question)
    assert parsed['status'] == 'ready'
    assert parsed['movie_id'] == movie
    assert parsed['relation_id'] == 'P57'


def test_unsupported_intent_is_not_mistaken_for_two_movie_lookup(parser):
    parsed = parser.parse('Who composed the music for Silver Orchard?')
    assert parsed['status'] == 'unanswerable'
    assert parsed['reason_code'] == 'unsupported_property'
    assert parsed['canonical_question'] is None


def test_true_multiple_movies_are_still_ambiguous(parser):
    parsed = parser.parse('Who directed Silver Orchard and Music?')
    assert parsed['status'] == 'needs_clarification'
    assert parsed['reason_code'] == 'ambiguous_movie'
    assert {item['id'] for item in parsed['candidates']} == {'Q101', 'Q102'}


def test_fuzzy_match_must_be_confirmed(parser):
    parsed = parser.parse('Who directed Silver Orchrad?')
    assert parsed['status'] == 'needs_clarification'
    assert parsed['reason_code'] == 'confirm_fuzzy_movie'
    assert parsed['candidates'][0]['id'] == 'Q101'
    assert parsed['canonical_question'] is None


def test_unknown_year_is_not_silently_discarded(parser):
    parsed = parser.parse('Who directed Silver Orchard (2019)?')
    assert parsed['status'] != 'ready'
    assert parsed['canonical_question'] is None


@pytest.mark.parametrize('question,relation', [
    ('《Silver Orchard》属于什么类型的电影？', 'P136'),
    ('《Silver Orchard》的原始语言是什么？', 'P364'),
    ('Silver Orchard属于什么类型的电影？', 'P136'),
])
def test_exact_english_title_in_chinese_question_reuses_grounding(parser, question, relation):
    parsed = parser.parse(question)
    assert parsed['status'] == 'ready'
    assert parsed['movie_id'] == 'Q101'
    assert parsed['relation_id'] == relation


def test_exact_chinese_grounding_does_not_drop_unknown_title_extensions(parser):
    parsed = parser.parse('Unknown Silver Orchard的导演是谁？')
    assert parsed['status'] != 'ready'
    assert parsed['canonical_question'] is None


def test_exact_chinese_grounding_still_rejects_multiple_movies(parser):
    parsed = parser.parse('《Silver Orchard》和《Music》的导演是谁？')
    assert parsed['status'] == 'needs_clarification'
    assert parsed['canonical_question'] is None
