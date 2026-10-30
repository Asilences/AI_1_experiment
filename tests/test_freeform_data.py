import pytest

from kgqa.acceptance import build_fixture
from kgqa.common import read_json, read_jsonl, sha256, write_jsonl
from kgqa.freeform_data import FAMILIES, build


@pytest.fixture
def source(tmp_path):
    data = tmp_path / 'original'
    build_fixture(data)
    entities = read_json(data / 'entities.json')
    triples = [f'{head}\t{relation}\t{tail}\n' for head, entity in entities.items()
               for relation, tails in entity.get('facts', {}).items() for tail in tails]
    (data / 'triples.tsv').write_text(''.join(triples), encoding='utf-8')
    return data


def test_export_preserves_frozen_graph_splits_answers_and_source_bytes(source, tmp_path):
    before = {path.name: sha256(path) for path in source.iterdir() if path.is_file()}
    output = tmp_path / 'freeform'
    manifest = build(source, output)
    assert before == {path.name: sha256(path) for path in source.iterdir() if path.is_file()}
    for name in ('entities.json', 'splits.json', 'triples.tsv'):
        assert (source / name).read_bytes() == (output / name).read_bytes()
    original = {row['question_id']: row for row in read_jsonl(source / 'qa.jsonl')}
    rows = read_jsonl(output / 'qa.jsonl')
    assert len({row['question_id'] for row in rows}) == len(rows) == manifest['qa']
    assert all(row['answer_ids'] == original[row['source_question_id']]['answer_ids'] for row in rows)
    assert all(row['reference_answers'] == original[row['source_question_id']]['reference_answers'] for row in rows)
    assert all(row['split'] == original[row['source_question_id']]['split'] for row in rows)
    assert {row['question_id'] for row in rows if row['question_source'] == 'original_train'} == {
        row['question_id'] for row in original.values() if row['split'] == 'train'}
    assert all(row['question_source'] == 'synthetic_paraphrase' for row in rows if row['split'] != 'train')
    assert read_json(output / 'manifest.json')['source_hashes']['qa.jsonl'] == before['qa.jsonl']


def test_sentence_families_and_exact_templates_are_held_out_between_splits():
    family_ids, templates = {}, {}
    for split, relations in FAMILIES.items():
        family_ids[split] = {family for values in relations.values() for family, _ in values}
        templates[split] = {template.casefold() for values in relations.values() for _, template in values}
    for left, right in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        assert not family_ids[left] & family_ids[right]
        assert not templates[left] & templates[right]


def test_existing_dataset_is_not_overwritten(source, tmp_path):
    output = tmp_path / 'freeform'
    build(source, output)
    before = sha256(output / 'qa.jsonl')
    with pytest.raises(FileExistsError):
        build(source, output)
    assert sha256(output / 'qa.jsonl') == before


def test_bad_source_split_is_rejected_before_output(source, tmp_path):
    rows = read_jsonl(source / 'qa.jsonl')
    rows[0]['split'] = 'test'
    write_jsonl(source / 'qa.jsonl', rows)
    output = tmp_path / 'freeform'
    with pytest.raises(ValueError, match='movie split'):
        build(source, output)
    assert not output.exists()


def test_export_cannot_reuse_source_directory_or_repeat_augmentation(source, tmp_path):
    with pytest.raises(ValueError, match='separate'):
        build(source, source)
    first = tmp_path / 'freeform'
    build(source, first)
    with pytest.raises(ValueError, match='already augmented'):
        build(first, tmp_path / 'second')
