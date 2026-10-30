from kgqa.common import write_json
from kgqa.v2_data import cached_entities, coverage, select_pilot, statements


def test_statement_preserves_amount_scope_unknown_and_rank():
    source = {'retrieved_at': '2026-10-29T00:00:00Z', 'cache_file': 'fixture'}
    qualifiers = {'P3005': [{'snaktype': 'value', 'datavalue': {'value': {'id': 'Q30'}}}]}
    claim = {'id': 'Q1$abc', 'rank': 'normal', 'qualifiers': qualifiers,
             'references': [{'snaks': {'P854': [{'datavalue': {'value': 'https://example.org'}}]}}],
             'mainsnak': {'snaktype': 'value', 'datatype': 'quantity',
                          'datavalue': {'value': {'amount': '+100.25', 'unit': 'http://www.wikidata.org/entity/Q4917'}}}}
    entity = {'id': 'Q1', 'lastrevid': 25, 'claims': {'P2142': [claim], 'P58': [
        {'rank': 'normal', 'mainsnak': {'snaktype': 'somevalue', 'datatype': 'wikibase-item'}},
        {'rank': 'deprecated', 'mainsnak': {'snaktype': 'value'}}]}}
    rows = statements(entity, source)
    amount = next(r for r in rows if r['property'] == 'P2142')
    assert amount['value']['amount'] == '+100.25'
    assert amount['qualifiers'] == qualifiers and amount['references'] == claim['references']
    assert amount['revision'] == 25 and amount['retrieved_at'] == source['retrieved_at']
    assert len(rows) == 2 and next(r for r in rows if r['property'] == 'P58')['value'] is None


def test_pilot_is_reproducible_has_unknown_stratum_and_does_not_mutate():
    movies = [{'id': f'Q{i}', 'facts': {'P495': [f'Q{100+i%3}'], 'P364': ['Q1860']},
               'release_year': None if i % 2 else 2000} for i in range(1, 31)]
    selected = select_pilot(movies, 12)
    assert selected == select_pilot(movies, 12)
    assert len(set(selected)) == 12
    assert any(int(q[1:]) % 2 for q in selected) and any(not int(q[1:]) % 2 for q in selected)
    assert len(movies) == 30


def test_coverage_counts_missing_and_unknown_without_calling_it_conflict():
    movies = [{'id': 'Q1'}, {'id': 'Q2'}, {'id': 'Q3'}]
    rows = [{'subject': 'Q1', 'property': 'P58', 'snaktype': 'value', 'value': 'Q9', 'references': [], 'qualifiers': {}},
            {'subject': 'Q1', 'property': 'P58', 'snaktype': 'value', 'value': 'Q8', 'references': [], 'qualifiers': {}},
            {'subject': 'Q2', 'property': 'P58', 'snaktype': 'somevalue', 'value': None, 'references': [], 'qualifiers': {}}]
    report = coverage(movies, rows, {'Q1': {}, 'Q2': {}})
    assert report['properties']['P58']['coverage'] == 1/3
    assert report['properties']['P58']['multi_value_movies'] == 1
    assert report['properties']['P58']['unknown_snaks'] == 1
    assert report['missing_movie_ids'] == ['Q3']


def test_cache_import_requires_claims_and_retains_revision_and_source(tmp_path):
    older = tmp_path / 'older.json'
    newer = tmp_path / 'newer.json'
    label_only = tmp_path / 'labels.json'
    for path, revision, props in [(older, 2, 'labels|claims|info'), (newer, 5, 'labels|claims|info'),
                                  (label_only, 9, 'labels|info')]:
        write_json(path, {'params': {'props': props}, 'retrieved_at': str(revision),
                         'response': {'entities': {'Q1': {'id': 'Q1', 'lastrevid': revision,
                           'claims': {'P161': [], 'P999': []}, 'labels': {'en': {'value': 'Film'}}}}}})
    entities, sources = cached_entities([older, newer, label_only], {'Q1'})
    assert entities['Q1']['lastrevid'] == 5
    assert sources['Q1']['retrieved_at'] == '5'
    assert 'P999' not in entities['Q1']['claims']
