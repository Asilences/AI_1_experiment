import pytest

from kgqa.linker import EntityLinker, normalize


@pytest.fixture
def linker():
    entities = {
        'Q1': {'label': 'Star Wars', 'aliases': ['星球大战', '星战', '共同片名']},
        'Q2': {'label': 'Film Beta', 'aliases': ['第二电影', '共同片名']},
        'Q3': {'label': 'Her', 'aliases': ['她的 Her 电影']},
    }
    return EntityLinker(entities, entities)


@pytest.mark.parametrize('title', ['星球大战', '星战', '《星球大战》'])
def test_unicode_exact_alias(linker, title):
    assert linker.link(f'Who directed {title}?') == ('Q1', 1.0, 'exact')


@pytest.mark.parametrize('title', [
    '未知电影', '未知 Her 电影', '未知 Her', 'Her 电影', '星球大战续集', '星球大站',
])
def test_unknown_unicode_never_matches_ascii_remnant_or_fuzzy(linker, title):
    assert linker.link(f'Who directed {title}?')[0] is None


def test_unknown_mixed_alias_does_not_link_to_its_ascii_remnant():
    entities = {'Q1': {'label': 'Alpha', 'aliases': ['甲 ABC 电影']}}
    linker = EntityLinker(entities, entities)
    assert linker.link('Who directed 乙 ABC 电影?')[0] is None
    assert linker.link('Who directed 甲 ABC 电影?')[0] == 'Q1'


@pytest.mark.parametrize('titles', [
    '共同片名', '星球大战 and 第二电影', 'Star Wars and 第二电影',
    '星球大战 and Film Beta',
])
def test_unicode_conflicts_and_multiple_movies_refused(linker, titles):
    assert linker.link(f'Compare {titles}')[2] == 'ambiguous'


def test_two_languages_for_same_movie_are_one_entity(linker):
    question = 'Who directed Star Wars / 星球大战?'
    assert linker.link(question) == ('Q1', 1.0, 'exact')
    assert linker.mask(question, 'Q1') == 'who directed movie_title movie_title'


@pytest.mark.parametrize('question,masked', [
    ('Who directed 星球大战?', 'who directed movie_title'),
    ('Which country produced 《星战》?', 'which country produced movie_title'),
    ('Who directed 她的 Her 电影?', 'who directed movie_title'),
])
def test_unicode_mask_keeps_classifier_ascii_contract(linker, question, masked):
    assert linker.mask(question, linker.link(question)[0]) == masked


def test_nfkc_and_casefold_preserve_unicode_full_title():
    entities = {'Q1': {'label': '电影 ＡＢＣ １２', 'aliases': ['École 电影']}}
    linker = EntityLinker(entities, entities)
    assert linker.link('Who directed 电影 abc 12?') == ('Q1', 1.0, 'exact')
    assert linker.link('Who directed E\u0301COLE 电影?') == ('Q1', 1.0, 'exact')
    assert linker.mask('Who directed 电影 ＡｂＣ １２?', 'Q1') == 'who directed movie_title'


def test_single_character_unicode_title_is_complete_name():
    entities = {'Q1': {'label': 'Her', 'aliases': ['她']}}
    linker = EntityLinker(entities, entities)
    assert linker.link('Who directed 她?') == ('Q1', 1.0, 'exact')
    assert linker.link('Who directed 她们?')[0] is None


def test_ascii_normalization_and_matching_stay_unchanged(linker):
    assert normalize('星球大战 ÉCOLE ＡＢＣ') == 'cole abc'
    assert linker.link('Who directed STAR WARS?') == ('Q1', 1.0, 'exact')
    assert linker.link('Who directed Str Wars?')[::2] == ('Q1', 'fuzzy')
    assert linker.mask('Who directed Str Wars?', 'Q1') == 'who directed movie_title'
    assert linker.link('Compare Star Wars and Film Beta')[2] == 'ambiguous'


@pytest.mark.parametrize('title,suffix', [('Film', ' 电影'), ('星球大战', '')])
def test_unknown_numeric_sequel_cannot_fall_back_in_unicode_query(title, suffix):
    entities = {'M1': {'label': title, 'aliases': []},
                'M2': {'label': title + ' 2', 'aliases': []}}
    linker = EntityLinker(entities, entities)
    assert linker.link(f'Who directed {title} 3{suffix}?')[0] is None
    assert linker.link(f'Who directed {title} 2?') == ('M2', 1.0, 'exact')


def test_unicode_extension_uses_existing_distinguishing_word_protection():
    entities = {'M1': {'label': '电影 Alpha', 'aliases': []},
                'M2': {'label': '电影 Alpha Resurrection Part Two', 'aliases': []}}
    linker = EntityLinker(entities, entities)
    assert linker.link('Who directed 电影 Alpha Resurrection Completely Different?')[0] is None
    assert linker.link('Who directed 电影 Alpha Resurrection Part Two?') == ('M2', 1.0, 'exact')


def test_unicode_suspected_sequel_is_not_fuzzy_linked():
    entities = {'M1': {'label': '星球大战', 'aliases': []},
                'M2': {'label': '星球大战 帝国反击战', 'aliases': []}}
    linker = EntityLinker(entities, entities)
    assert linker.link('Who directed 星球大战 帝国反击站?')[0] is None
    assert linker.link('Who directed 星球大战 帝国反击战?') == ('M2', 1.0, 'exact')
