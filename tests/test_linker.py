import pytest

from kgqa.common import REFUSAL
from kgqa.linker import EntityLinker
from kgqa.pipeline import PipelineQA, mask_question


ENTITIES = {
    'Q1': {'label': 'Star Wars', 'aliases': ['A New Hope'], 'facts': {'P57': ['Q3']}},
    'Q2': {'label': 'Film Beta', 'aliases': []},
    'Q3': {'label': 'Jane Director', 'aliases': []},
}


@pytest.mark.parametrize('question', [
    'Who directed Star Wars?',
    'Can you tell me who helmed Star Wars?',
    'Which country did STAR WARS originate in?',
    'Who directed A New Hope?',
])
def test_exact_title_alias_and_paraphrase(question):
    assert EntityLinker(['Q1', 'Q2'], ENTITIES).link(question) == ('Q1', 1.0, 'exact')


def test_fuzzy_compares_title_span_in_full_question():
    linker = EntityLinker(['Q1', 'Q2'], ENTITIES)
    qid, score, status = linker.link('Who directed Str Wars?')
    assert qid == 'Q1' and status == 'fuzzy' and score > .9
    assert linker.mask('Who directed Str Wars?', qid) == 'who directed movie_title'


@pytest.mark.parametrize('question', [
    'sraW ratS',  # Identical character counts do not imply identical sequences.
    'Who directed Completely Unknown Picture?',
    'Which country produced the unavailable movie?',
    '',
])
def test_unknown_and_scrambled_titles_are_refused(question):
    assert EntityLinker(['Q1', 'Q2'], ENTITIES).link(question)[0] is None


def test_multiple_explicit_movies_are_ambiguous_independent_of_order():
    for movies in (['Q1', 'Q2'], ['Q2', 'Q1']):
        assert EntityLinker(movies, ENTITIES).link('Compare Star Wars and Film Beta')[2] == 'ambiguous'


def test_shared_alias_is_ambiguous():
    entities = {'Q1': {'label': 'Alpha Film', 'aliases': ['Shared Title']},
                'Q2': {'label': 'Beta Film', 'aliases': ['Shared Title']}}
    assert EntityLinker(entities, entities).link('Who directed Shared Title?')[2] == 'ambiguous'


def test_nearby_fuzzy_candidates_need_margin():
    entities = {'Q1': {'label': 'Star Wars'}, 'Q2': {'label': 'Star Cars'}}
    assert EntityLinker(entities, entities).link('Who directed Star Mars?')[2] == 'ambiguous'


def test_equal_fuzzy_candidates_remain_ambiguous_with_zero_margin():
    entities = {'Q1': {'label': 'Star Wars'}, 'Q2': {'label': 'Star Cars'}}
    assert EntityLinker(entities, entities, ambiguity_margin=0).link('Who directed Star Mars?')[2] == 'ambiguous'


def test_longest_nested_exact_title_wins():
    entities = {'Q1': {'label': 'Star Wars'}, 'Q2': {'label': 'Star Wars Episode Four'}}
    assert EntityLinker(entities, entities).link('Who directed Star Wars Episode Four?')[0] == 'Q2'


@pytest.fixture
def nested_linker():
    entities = {'Q1': {'label': 'Star Wars'},
                'Q8': {'label': 'Star Wars Episode Four'}}
    return EntityLinker(entities, entities)


def test_nested_long_title_typo_does_not_select_exact_short_prefix(nested_linker):
    question = 'Who directed Star Wars Episode Foor?'
    qid, score, status = nested_linker.link(question)
    assert qid == 'Q8' and status == 'fuzzy' and score >= .88
    assert nested_linker.mask(question, qid) == 'who directed movie_title'


def test_nested_fuzzy_mask_prefers_complete_span_over_short_exact_alias():
    entities = {'Q1': {'label': 'Star Wars'},
                'Q8': {'label': 'Star Wars Episode Four', 'aliases': ['Star']}}
    linker = EntityLinker(entities, entities)
    question = 'Who directed Star Wars Episode Foor?'
    assert linker.link(question)[0] == 'Q8'
    assert linker.mask(question, 'Q8') == 'who directed movie_title'
    assert mask_question({'question': question, 'movie_id': 'Q8'}, entities, linker) == 'who directed movie_title'


def test_nested_long_title_typo_can_precede_exact_anchor():
    entities = {'Q1': {'label': 'Star Wars'},
                'Q8': {'label': 'Return of Star Wars'}}
    linker = EntityLinker(entities, entities)
    question = 'Who directed Retrun of Star Wars?'
    qid, score, status = linker.link(question)
    assert qid == 'Q8' and status == 'fuzzy' and score >= .88
    assert linker.mask(question, qid) == 'who directed movie_title'


def test_nested_title_requires_both_extensions_to_match():
    entities = {'Q1': {'label': 'Star Wars'},
                'Q8': {'label': 'Return of Star Wars Episode Four'}}
    linker = EntityLinker(entities, entities)
    assert linker.link('Who directed Fall of Star Wars Episode Foor?')[0] is None


@pytest.mark.parametrize('question', [
    'Who directed Star Wars?',
    'Which country did Star Wars originate in?',
    'Who directed Star Wars in France?',
    'Who directed Star Wars during the Episode Four discussion?',
])
def test_unrelated_question_words_do_not_extend_exact_short_title(nested_linker, question):
    assert nested_linker.link(question) == ('Q1', 1.0, 'exact')


def test_exact_nested_long_title_is_preserved(nested_linker):
    assert nested_linker.link('Who directed Star Wars Episode Four?') == ('Q8', 1.0, 'exact')


@pytest.mark.parametrize('question', [
    'Who directed Star Wars Episode Nine?',
    'Who directed Star Wars Episod Nine?',
])
def test_unknown_extension_cannot_fall_back_to_exact_short_title(nested_linker, question):
    assert nested_linker.link(question)[0] is None


@pytest.mark.parametrize('sequel_title,question', [
    ('It 2', 'Who directed It 3?'),
    ('It 2 The Next Chapter', 'Who directed It 3 The Next Chapter?'),
])
def test_unknown_numeric_sequel_cannot_fall_back_or_fuzzy_match(sequel_title, question):
    entities = {'Q1': {'label': 'It'}, 'Q2': {'label': sequel_title}}
    assert EntityLinker(entities, entities).link(question)[2] == 'ambiguous'


@pytest.mark.parametrize('question,expected_id', [
    ('Who directed It?', 'Q1'),
    ('Who directed It 2?', 'Q2'),
    ('Who directed It in 2023?', 'Q1'),
    ('In 2023, who directed It?', 'Q1'),
    ('Who directed It 2 in 2023?', 'Q2'),
])
def test_numeric_sequel_rule_preserves_exact_titles_and_nonadjacent_numbers(question, expected_id):
    entities = {'Q1': {'label': 'It'}, 'Q2': {'label': 'It 2'}}
    assert EntityLinker(entities, entities).link(question) == (expected_id, 1.0, 'exact')


def test_long_extension_candidates_need_their_own_margin():
    entities = {'Q1': {'label': 'Star Wars'},
                'Q8': {'label': 'Star Wars Episode Four'},
                'Q9': {'label': 'Star Wars Episode Foor'}}
    assert EntityLinker(entities, entities).link('Who directed Star Wars Episode Fowr?')[2] == 'ambiguous'


def test_typo_long_title_and_independent_short_title_are_ambiguous(nested_linker):
    assert nested_linker.link('Compare Star Wars Episode Foor and Star Wars')[2] == 'ambiguous'


def test_short_titles_use_word_boundaries_and_exact_match_only():
    entities = {'Q1': {'label': 'Her'}}
    linker = EntityLinker(entities, entities)
    assert linker.mask('Where was Her produced?', 'Q1') == 'where was movie_title produced'
    assert linker.link('Where was the movie produced?')[0] is None
    assert linker.link('Who directed Hre?')[0] is None


def test_mask_training_and_inference_handle_same_alias():
    linker = EntityLinker(['Q1', 'Q2'], ENTITIES)
    question = 'Who directed A New Hope?'
    row = {'question': question, 'movie_id': 'Q1'}
    assert mask_question(row, ENTITIES, linker) == linker.mask(question, 'Q1') == 'who directed movie_title'


class RecordingClassifier:
    classes_ = ['P57']

    def __init__(self):
        self.inputs = []

    def predict_proba(self, rows):
        self.inputs.extend(rows)
        return [[1.0] for _ in rows]


def test_pipeline_masks_alias_and_returns_grounded_evidence():
    classifier = RecordingClassifier()
    system = PipelineQA(ENTITIES, ['Q1', 'Q2'], classifier, .5)
    result = system.predict('Who directed A New Hope?')
    assert classifier.inputs == ['who directed movie_title']
    assert result['answer_ids'] == ['Q3']
    assert result['evidence'] == [['Q1', 'P57', 'Q3']]


def test_pipeline_refuses_ambiguous_title_without_classifier_call():
    classifier = RecordingClassifier()
    system = PipelineQA(ENTITIES, ['Q1', 'Q2'], classifier, .5)
    result = system.predict('Compare Star Wars and Film Beta')
    assert result['answer'] == REFUSAL
    assert result['answer_ids'] == []
    assert classifier.inputs == []


def test_pipeline_refuses_unknown_numeric_sequel_without_querying_original_facts():
    classifier = RecordingClassifier()
    entities = {'Q1': {'label': 'It', 'facts': {'P57': ['Q3']}},
                'Q2': {'label': 'It 2', 'facts': {'P57': ['Q3']}},
                'Q3': {'label': 'Example Director'}}
    system = PipelineQA(entities, ['Q1', 'Q2'], classifier, .5)
    result = system.predict('Who directed It 3?')
    assert result['answer'] == REFUSAL
    assert result['answer_ids'] == result['evidence'] == []
    assert classifier.inputs == []
