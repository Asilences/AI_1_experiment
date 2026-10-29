import copy

import pytest

from kgqa import expand, prepare, pipeline, web
from kgqa.common import RELATIONS, read_json, read_jsonl, sha256, write_json
from kgqa.linker import EntityLinker
from kgqa.pipeline import PipelineQA, load_model


def item_claim(qid):
    return {'rank': 'normal', 'mainsnak': {'snaktype': 'value', 'datatype': 'wikibase-item',
                                        'datavalue': {'value': {'id': qid}}}}


def film(qid):
    return {'id': qid, 'lastrevid': 123, 'labels': {'en': {'value': 'Film ' + qid}, 'zh': {'value': '电影' + qid}},
            'aliases': {'en': [{'value': 'Alias ' + qid}], 'zh': [{'value': '别名' + qid}]},
            'claims': {'P31': [item_claim('Q11424')],
                       **{r: [item_claim('Q' + str(9000 + i))] for i, r in enumerate(RELATIONS)},
                       'P577': [{'rank': 'normal', 'mainsnak': {'datavalue': {'value': {'time': '+2012-05-01T00:00:00Z'}}}}]}}


class FakeApi:
    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def entities(self, ids, claims=True):
        self.calls.append((list(ids), claims))
        if claims and self.fail_on in ids:
            raise RuntimeError('Simulated interruption')
        if claims:
            return {qid: film(qid) for qid in ids}
        return {qid: {'id': qid, 'labels': {'en': {'value': 'Tail ' + qid}}, 'lastrevid': 456} for qid in ids}


def source_files(tmp_path):
    base = tmp_path / 'base.json'
    movie, entities, _ = expand.fetch_api_batch(FakeApi(), ['Q1'], {})
    write_json(base, {'movies': movie, 'entities': entities, 'created_at': '2026-09-12T00:00:00Z'})
    candidates = tmp_path / 'candidates.json'
    write_json(candidates, {'retrieved_at': '2026-10-29T00:00:00Z', 'params': {'query': 'SELECT ?film WHERE {}'},
                            'response': {'results': {'bindings': [
                                {'film': {'value': 'http://www.wikidata.org/entity/' + qid}} for qid in ['Q1', 'Q2', 'Q3']]}}})
    return base, candidates


def test_item_api_cleans_real_statement_structure_and_preserves_revision():
    movies, entities, rejected = expand.fetch_api_batch(FakeApi(), ['Q1'], {})
    movie = movies[0]
    assert movie['label'] == 'Film Q1' and movie['revision'] == 123
    assert movie['release_year'] == 2012 and movie['label_zh'] == '电影Q1'
    assert set(movie['aliases']) == {'Alias Q1', '别名Q1', '电影Q1'}
    assert len(movie['qa_relations']) == 4 and entities['Q9000']['label'] == 'Tail Q9000'
    assert not rejected


def test_nonfilms_and_incomplete_multianswers_are_not_accepted():
    class BrokenApi(FakeApi):
        def entities(self, ids, claims=True):
            records = super().entities(ids, claims)
            if claims:
                records['Q1']['claims']['P31'] = [item_claim('Q5')]
                records['Q2']['claims']['P57'] += [item_claim('Q99999')]
                records['Q2']['claims']['P364'] = []
            else:
                records.pop('Q99999', None)
            return records
    movies, _, rejected = expand.fetch_api_batch(BrokenApi(), ['Q1', 'Q2'], {})
    assert not movies
    assert rejected['no_longer_direct_instance_of_film'] == 1
    assert rejected['fewer_than_three_complete_qa_relations'] == 1


def test_resume_retains_completed_batch_and_never_changes_base_snapshot(tmp_path):
    base, candidates = source_files(tmp_path)
    before = base.read_bytes()
    work = tmp_path / 'work'
    with pytest.raises(RuntimeError, match='interruption'):
        expand.collect(base, candidates, work, target=3, batch_size=1, client=FakeApi(fail_on='Q3'))
    assert read_json(work / 'progress_3.json')['cursor'] == 1
    client = FakeApi()
    snapshot = expand.collect(base, candidates, work, target=3, batch_size=1, client=client)
    result = read_json(snapshot)
    assert len(result['movies']) == 3 and result['base']['movies'] == 1
    assert [ids for ids, claims in client.calls if claims] == [['Q3']]
    assert base.read_bytes() == before
    assert result['expansion_identity']['base_sha256'] == sha256(base)
    old = snapshot.read_bytes()
    expand.collect(base, candidates, work, target=3, client=FakeApi(fail_on='Q2'))
    assert snapshot.read_bytes() == old


def test_resume_rejects_changed_candidate_source(tmp_path):
    base, candidates = source_files(tmp_path)
    work = tmp_path / 'work'
    with pytest.raises(RuntimeError):
        expand.collect(base, candidates, work, target=3, batch_size=1, client=FakeApi(fail_on='Q3'))
    document = read_json(candidates)
    document['retrieved_at'] = 'changed'
    write_json(candidates, document)
    with pytest.raises(ValueError, match='hashes changed'):
        expand.collect(base, candidates, work, target=3, client=FakeApi())


def test_shared_titles_have_unambiguous_clickable_question_titles():
    movies = [{'id': 'Q1', 'label': 'The Avengers', 'aliases': [], 'release_year': 2012},
              {'id': 'Q2', 'label': 'The Avengers', 'aliases': [], 'release_year': 1998},
              {'id': 'Q3', 'label': 'The Avengers', 'aliases': []}]
    entities = {}
    expand.disambiguate_titles(movies, entities)
    linker = EntityLinker([m['id'] for m in movies], entities)
    assert linker.link('Who directed The Avengers?')[0] is None
    for movie in movies:
        assert linker.link('Who directed ' + movie['question_title'] + '?')[0] == movie['id']


def test_single_letter_titles_remain_linkable_without_relaxing_global_threshold():
    movies = [{'id': 'Q1', 'label': 'M', 'aliases': [], 'release_year': 1931},
              {'id': 'Q2', 'label': 'Q', 'aliases': []}]
    entities = {}
    expand.disambiguate_titles(movies, entities)
    linker = EntityLinker(['Q1', 'Q2'], entities)
    for movie in movies:
        assert movie['question_title'] != movie['label']
        assert linker.link('Who directed ' + movie['question_title'] + '?')[0] == movie['id']


def test_expanded_splits_preserve_old_assignments_and_70_15_15():
    movies = [{'id': f'Q{i}'} for i in range(20)]
    previous = {'train': ['Q0', 'Q1', 'Q2', 'Q3'], 'valid': ['Q4'], 'test': ['Q5']}
    original = copy.deepcopy(previous)
    result = prepare.extend_splits(movies, previous)
    assert {k: len(v) for k, v in result.items()} == {'train': 14, 'valid': 3, 'test': 3}
    for split in previous:
        assert set(previous[split]) <= set(result[split])
    assert len(set(sum(result.values(), []))) == 20
    assert result == prepare.extend_splits(movies, previous) and previous == original
    with pytest.raises(ValueError, match='disjoint'):
        prepare.extend_splits(movies, {'train': ['Q1'], 'valid': ['Q1'], 'test': []})


def test_versioned_prepare_and_model_training_do_not_overwrite_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    movies, entities, _ = expand.fetch_api_batch(FakeApi(), [f'Q{i}' for i in range(1, 21)], {})
    expand.disambiguate_titles(movies, entities)
    snapshot = tmp_path / 'snapshot.json'
    write_json(snapshot, {'movies': movies, 'entities': entities})
    data, model_dir = tmp_path / 'new_data', tmp_path / 'new_model'
    prepare.build(snapshot, data)
    assert not (tmp_path / 'data/processed').exists()
    before = (data / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        prepare.build(snapshot, data)
    assert (data / 'manifest.json').read_bytes() == before
    model_path = pipeline.train(data, model_dir)
    assert not (tmp_path / 'models/pipeline').exists()
    model, threshold = load_model(model_path)
    splits = read_json(data / 'splits.json')
    system = PipelineQA(entities, sum(splits.values(), []), model, threshold)
    row = next(r for r in read_jsonl(data / 'qa.jsonl') if r['split'] == 'test' and r['relation_id'] == 'P57')
    assert set(system.predict(row['question'])['answer_ids']) == set(row['answer_ids'])
    assert read_json(model_dir / 'config.json')['input_hashes']['entities.json'] == sha256(data / 'entities.json')
    with pytest.raises(FileExistsError):
        pipeline.train(data, model_dir)


def test_active_profile_validates_model_pairing_and_legacy_bypasses_it(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'ROOT', tmp_path)
    data, model = tmp_path / 'data', tmp_path / 'model.pkl'
    data.mkdir()
    model.write_bytes(b'test model')
    for name in ('entities.json', 'splits.json', 'qa.jsonl'):
        (data / name).write_text('{}', encoding='utf-8')
    profile = {'data_dir': 'data', 'model_path': 'model.pkl',
               'input_hashes': {name: sha256(data / name) for name in ('entities.json', 'splits.json', 'qa.jsonl')},
               'model_sha256': sha256(model)}
    write_json(tmp_path / '.work/ui-dataset.json', profile)
    assert web.startup_paths() == (data.resolve(), model.resolve())
    assert web.startup_paths(legacy=True) == (None, None)
    model.write_bytes(b'different model')
    with pytest.raises(ValueError, match='hashes do not match'):
        web.startup_paths()
    assert web.startup_paths(data_dir=data) == (data, None)


def test_full_version_build_evaluates_before_activating_matching_ui_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(expand, 'ROOT', tmp_path)
    monkeypatch.setattr(web, 'ROOT', tmp_path)
    movies, entities, _ = expand.fetch_api_batch(FakeApi(), [f'Q{i}' for i in range(1, 21)], {})
    expand.disambiguate_titles(movies, entities)
    snapshot = tmp_path / 'snapshot.json'
    write_json(snapshot, {'movies': movies, 'entities': entities})
    data, model = tmp_path / 'data/new', tmp_path / 'models/new'
    metrics = expand.prepare_and_train(snapshot, data, model, activate=True)
    assert metrics['count'] > 0 and metrics['prediction_coverage'] == 100
    assert metrics['entity_macro_f1'] == 100
    assert web.startup_paths() == (data.resolve(), (model / 'model.pkl').resolve())
    state = web.QAService(data, model / 'model.pkl').status()
    assert state['a0']['available'] and not state['neural']['available']
    assert state['dataset_name'] == '扩展电影库 · 20 部'
    before = (model / 'model.pkl').read_bytes()
    expand.prepare_and_train(snapshot, data, model)
    assert (model / 'model.pkl').read_bytes() == before
