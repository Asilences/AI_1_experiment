from kgqa.candidates import CandidateIndex


def index():
    return CandidateIndex([
        {'id': 'Q1', 'label': 'Inception', 'label_zh': '盗梦空间', 'aliases': ['盗梦'], 'release_year': 2010},
        {'id': 'Q2', 'label': 'The Avengers', 'aliases': ['复仇者联盟'], 'release_year': 2012},
        {'id': 'Q3', 'label': 'The Avengers', 'aliases': [], 'release_year': 1998},
        {'id': 'Q4', 'label': 'Up', 'aliases': [], 'release_year': 2009}])


def test_partial_and_typo_recall_do_not_silently_answer():
    movies = index()
    assert movies.search('incep')[0]['id'] == 'Q1'
    assert movies.search('Inceptoin')[0]['id'] == 'Q1'
    ids, choices, state = movies.mentions('Who directed Inceptoin?')
    assert not ids and choices[0]['id'] == 'Q1' and state == 'needs_confirmation'


def test_multiple_exact_titles_and_chinese_alias_are_supported():
    movies = index()
    assert movies.mentions('盗梦空间的演员和编剧是谁？')[0] == ['Q1']
    assert movies.mentions('Q1 and Q2 shared cast')[0] == ['Q1', 'Q2']


def test_homonyms_unknown_sequels_and_ordinary_short_words_do_not_auto_accept():
    movies = index()
    assert not movies.mentions('Who directed The Avengers?')[0]
    assert len(movies.mentions('Who directed The Avengers?')[1]) == 2
    assert not movies.mentions('Inception 99 的导演是谁')[0]
    assert not movies.mentions('What is up with this movie?')[0]
    assert movies.mentions('Who directed “Up”?')[0] == ['Q4']
    assert movies.mentions('Q999 director')[2] == 'not_found'


def test_movie_named_what_does_not_capture_question_word():
    movies = CandidateIndex([{'id': 'Q1', 'label': 'Inception', 'aliases': []},
                             {'id': 'Q2', 'label': 'What?', 'aliases': []}])
    assert movies.mentions('What is the cast of Inception?')[0] == ['Q1']
    assert movies.mentions('Who directed “What?”?')[0] == ['Q2']


def test_traditional_source_title_matches_simplified_question_without_inventing_alias():
    movies = CandidateIndex([{'id': 'Q1', 'label': 'Interstellar', 'label_zh': '星際穿越', 'aliases': []}])
    assert movies.mentions('星际穿越的演员是谁？')[0] == ['Q1']
    assert movies.movies['Q1']['label_zh'] == '星際穿越'
