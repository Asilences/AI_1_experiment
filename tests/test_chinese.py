"""Chinese grounding is offline and leaves the original English path intact."""
from copy import deepcopy

import pytest

from kgqa.chinese import ChineseAdapter
from kgqa.common import REFUSAL


@pytest.fixture
def adapter():
    entities = {
        'Q1': {'label': 'Shutter Island', 'label_zh': '禁闭岛', 'aliases': ['隔离岛'],
               'release_year': 2010, 'facts': {'P57': ['Q8']}},
        'Q2': {'label': 'King Kong', 'label_zh': '金刚', 'aliases': [], 'release_year': 1933,
               'question_title': 'King Kong (1933)'},
        'Q3': {'label': 'King Kong', 'label_zh': '金刚', 'aliases': ['大金刚'], 'release_year': 2005,
               'question_title': 'King Kong (2005)'},
        'Q4': {'label': 'The Director', 'label_zh': '导演', 'aliases': [], 'release_year': 2020},
        'Q5': {'label': 'Island', 'label_zh': '岛屿', 'aliases': []},
        'Q6': {'label': 'No Country for Old Men', 'label_zh': '老无所依', 'aliases': []},
        'Q7': {'label': 'Inception', 'label_zh': '盗梦空间', 'aliases': []},
        'Q8': {'label': 'Martin Scorsese', 'label_zh': '马丁·斯科塞斯'},
        'Q9': {'label': 'English', 'label_zh': '英语'},
        'Q10': {'label': 'United States', 'label_zh': '美国'},
        'Q11': {'label': 'thriller film', 'label_zh': '惊悚片'},
        'Q12': {'label': 'Other Predicted Director'},
    }
    return ChineseAdapter(entities, [f'Q{i}' for i in range(1, 8)])


@pytest.mark.parametrize('question,relation', [
    ('《禁闭岛》的导演是谁？', 'P57'),
    ('谁执导了隔离岛？', 'P57'),
    ('Shutter Island 是谁拍的？', 'P57'),
    ('禁闭岛属于什么类型？', 'P136'),
    ('《禁闭岛》是什么题材的电影？', 'P136'),
    ('禁闭岛的制片国家是哪里？', 'P495'),
    ('禁闭岛是哪国的电影？', 'P495'),
    ('《禁闭岛》最初用什么语言拍成？', 'P364'),
    ('禁闭岛的原声是什么？', 'P364'),
    ('禁闭岛说哪国语言？', 'P364'),
    ('Q1 的导演是谁？', 'P57'),
    ('《导演》属于什么类型？', 'P136'),
    ('《The Director》的原始语言是什么？', 'P364'),
])
def test_supported_questions(adapter, question, relation):
    result = adapter.prepare(question)
    assert result.status == 'success'
    assert result.language == 'zh'
    assert result.relation_id == relation
    assert result.movie_id == ('Q4' if 'Director' in question or '《导演》' in question else 'Q1')
    assert not any(char in result.model_question for char in '禁闭岛导演')


@pytest.mark.parametrize('question', [
    'Who directed Shutter Island?', 'Who directed 禁闭岛?',
    'What genre is 《禁闭岛》?', '  Who directed Shutter Island?  ',
    'Who directed No Country for Old Men?',
])
def test_english_is_byte_for_byte_pass_through(adapter, question):
    result = adapter.prepare(question)
    assert result.language == 'en'
    assert result.status == 'pass_through'
    assert result.model_question == question == result.original_question
    assert result.movie_id is None and result.relation_id is None


@pytest.mark.parametrize('question,status', [
    ('《禁闭岛2》的导演是谁？', 'not_found'),
    ('《不存在的片名》的导演是谁？', 'not_found'),
    ('《Shutter Island Sequel》的导演是谁？', 'not_found'),
    ('禁闭島的导演是谁？', 'not_found'),
    ('禁闭岛2的导演是谁？', 'not_found'),
    ('Shutter Island 2 的导演是谁？', 'not_found'),
    ('Q999 的导演是谁？', 'not_found'),
    ('禁闭岛和盗梦空间的导演是谁？', 'unsupported'),
    ('《禁闭岛》和《不存在的片名》的导演是谁？', 'not_found'),
    ('禁闭岛的导演和类型是什么？', 'unsupported'),
    ('禁闭岛的导演和主演是谁？', 'unsupported'),
    ('禁闭岛不是谁导演的？', 'unsupported'),
    ('禁闭岛的类型不属于什么？', 'unsupported'),
    ('禁闭岛的导演是不是诺兰？', 'unsupported'),
    ('禁闭岛的导演是诺兰吗？', 'unsupported'),
    ('禁闭岛的导演为什么选择这个题材？', 'unsupported'),
    ('禁闭岛的导演出生在哪里？', 'unsupported'),
    ('禁闭岛的导演的国籍是什么？', 'unsupported'),
    ('禁闭岛是什么时候上映的？', 'unsupported'),
    ('禁闭岛在哪里拍摄的？', 'unsupported'),
    ('禁闭岛的导演拍过什么电影？', 'unsupported'),
    ('请介绍禁闭岛', 'unsupported'),
    ('金刚的导演是谁？', 'ambiguous'),
])
def test_refusals_are_explicit_and_do_not_invent_matches(adapter, question, status):
    result = adapter.prepare(question)
    assert result.language == 'zh'
    assert result.status == status
    assert result.reason
    assert result.movie_id is None


@pytest.mark.parametrize('question', [
    '《金刚（2005）》的导演是谁？', '《金刚》2005年版的导演是谁？',
    '2005年版金刚的导演是谁？', '金刚2005年版的导演是谁？',
    '《金刚》Q3 的导演是谁？',
])
def test_remake_years_and_qids_disambiguate(adapter, question):
    result = adapter.prepare(question)
    assert result.status == 'success'
    assert result.movie_id == 'Q3'
    assert result.model_question == 'Who directed King Kong (2005)?'


def test_wrong_year_is_not_silently_ignored(adapter):
    assert adapter.prepare('《禁闭岛》2005年版的导演是谁？').status == 'not_found'


def test_polite_open_question_is_not_a_yes_no_assertion(adapter):
    assert adapter.prepare('能告诉我《禁闭岛》的导演是谁吗？').status == 'success'


def test_longest_non_overlapping_title_and_intent_outside_title(adapter):
    result = adapter.prepare('《No Country for Old Men》的导演是谁？')
    assert result.status == 'success' and result.movie_id == 'Q6' and result.relation_id == 'P57'
    assert adapter.prepare('Shutter Island的导演是谁？').movie_id == 'Q1'
    assert adapter.prepare('《导演》的导演是谁？').relation_id == 'P57'


def test_answer_rendering_uses_predicted_ids_without_reading_gold_facts(adapter):
    before = deepcopy(adapter.entities)
    answer = adapter.format_answer('Q1', 'P57', ['Q12'])
    assert answer == '《禁闭岛》的导演是Other Predicted Director。'
    assert '斯科塞斯' not in answer
    assert adapter.entities == before
    assert adapter.format_entity('Q8') == '马丁·斯科塞斯'
    assert adapter.format_entity('Q12') == 'Other Predicted Director'
    assert adapter.format_entity('Q999') == 'Q999'
    assert adapter.format_answer('Q1', 'P57', []) == '现有知识图谱没有足够的信息回答这个问题。'


@pytest.mark.parametrize('answer,expected', [
    ('Shutter Island was directed by Martin Scorsese.', '《禁闭岛》的导演是马丁·斯科塞斯。'),
    ('The director or directors of Shutter Island are Martin Scorsese.', '《禁闭岛》的导演是马丁·斯科塞斯。'),
    ('Shutter Island belongs to the following genres: thriller film.', '《禁闭岛》的类型是惊悚片。'),
    ('The genres of Shutter Island are thriller film.', '《禁闭岛》的类型是惊悚片。'),
    ('Shutter Island originated in United States.', '《禁闭岛》的制片国家是美国。'),
    ('The countries of origin of Shutter Island are United States.', '《禁闭岛》的制片国家是美国。'),
    ('The original languages of Shutter Island are English.', '《禁闭岛》的原始语言是英语。'),
    ('Shutter Island was originally made in English.', '《禁闭岛》的原始语言是英语。'),
    ('Shutter Island was directed by Martin Scorsese and Unknown Person.', '《禁闭岛》的导演是马丁·斯科塞斯 and Unknown Person。'),
    ('Shutter Island was directed by Other Predicted Director.', '《禁闭岛》的导演是Other Predicted Director。'),
    ('Inception was directed by Martin Scorsese.', 'Inception was directed by Martin Scorsese.'),
    ('I think Martin Scorsese directed it.', 'I think Martin Scorsese directed it.'),
    ('Shutter Island was directed by Martin Scorsese. Extra claim.', 'Shutter Island was directed by Martin Scorsese. Extra claim.'),
    (REFUSAL, '现有知识图谱没有足够的信息回答这个问题。'),
])
def test_generated_localization_preserves_actual_relation_and_payload(adapter, answer, expected):
    assert adapter.localize_generated(answer, 'Q1') == expected


def test_optional_catalog_updates_display_and_movie_aliases_without_mutation():
    entities = {'Q210364': {'label': 'Shutter Island', 'aliases': [], 'facts': {'P495': ['Q30']}},
                'Q30': {'label': 'United States', 'aliases': []},
                'Q1860': {'label': 'English', 'aliases': []}}
    original = deepcopy(entities)
    catalog = {'entities': {
        'Q210364': {'label_en': 'Shutter Island', 'label_zh': '禁闭岛', 'aliases_zh': ['隔離島']},
        'Q30': {'label_en': 'United States', 'label_zh': '美国', 'aliases_zh': ['美利坚合众国']},
        'Q1860': {'label_en': 'English', 'label_zh': '英语', 'aliases_zh': []}}}
    system = ChineseAdapter(entities, ['Q210364'], label_catalog=catalog)
    assert system.prepare('《禁闭岛》的制片国家是哪里？').movie_id == 'Q210364'
    assert system.prepare('《隔離島》的原始语言是什么？').relation_id == 'P364'
    assert system.format_answer('Q210364', 'P495', ['Q30']) == '《禁闭岛》的制片国家是美国。'
    assert system.localize_generated('The original languages of Shutter Island are English.', 'Q210364') == '《禁闭岛》的原始语言是英语。'
    # A country label is display metadata and cannot introduce another movie.
    assert system.prepare('《美国》的导演是谁？').status == 'not_found'
    assert entities == original


def test_catalog_qids_must_match_the_supplied_english_label():
    entities = {'Q30': {'label': 'A Fixture Film', 'aliases': []}}
    catalog = {'entities': {'Q30': {'label_en': 'United States', 'label_zh': '美国', 'aliases_zh': ['美利坚']}}}
    system = ChineseAdapter(entities, ['Q30'], label_catalog=catalog)
    assert system.format_entity('Q30') == 'A Fixture Film'
    assert system.prepare('《美利坚》的导演是谁？').status == 'not_found'


def test_catalog_is_optional_and_cjk_alias_is_a_display_fallback(tmp_path):
    entities = {'Q1': {'label': 'Example', 'aliases': ['《示例电影》']},
                'Q2': {'label': 'Untranslated Name', 'aliases': []}}
    system = ChineseAdapter(entities, ['Q1'], label_catalog=tmp_path / 'absent.json')
    assert system.format_entity('Q1') == '示例电影'
    assert system.format_entity('Q2') == 'Untranslated Name'
    assert system.format_entity('Q999') == 'Q999'


def test_catalog_does_not_change_english_question_preparation():
    entities = {'Q210364': {'label': 'Shutter Island', 'aliases': []}}
    catalog = {'entities': {'Q210364': {'label_en': 'Shutter Island', 'label_zh': '禁闭岛', 'aliases_zh': []}}}
    before = ChineseAdapter(entities, ['Q210364'], label_catalog={})
    after = ChineseAdapter(entities, ['Q210364'], label_catalog=catalog)
    question = 'Who directed Shutter Island?'
    assert before.prepare(question) == after.prepare(question)
    assert after.prepare('Who directed 禁闭岛?').model_question == 'Who directed 禁闭岛?'


def test_shipped_catalog_has_official_revision_provenance_and_common_labels():
    import json
    from pathlib import Path
    import kgqa.chinese as chinese

    catalog = json.loads((Path(chinese.__file__).with_name('locales') / 'zh_entities.json').read_text(encoding='utf-8'))
    assert catalog['source'] == 'https://www.wikidata.org/w/api.php'
    assert catalog['schema_version'] == 1
    for record in catalog['entities'].values():
        assert record['label_en'] and record['label_zh'] and record['source_label_en']
        assert isinstance(record['revision'], int) and record['revision'] > 0
        assert record['retrieved_at'] and record['source_url'].startswith('https://www.wikidata.org/')
    assert catalog['entities']['Q30']['label_zh'] == '美国'
    assert catalog['entities']['Q1860']['label_zh'] == '英语'
    assert catalog['entities']['Q210364']['label_zh'] == '禁闭岛'
