"""CPU-only path checks when the optional full neural dependencies are present."""
import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('transformers')

from kgqa.common import read_json, sha256, write_json, write_jsonl
from kgqa import neural
from kgqa.neural import GraphData, QADataset


def test_graph_and_linking_use_explicit_dataset_and_embeddings(tmp_path):
    data, embeddings = tmp_path / 'dataset', tmp_path / 'embeddings'
    write_json(data / 'entities.json', {'Q1': {'label': 'Example Film', 'facts': {'P57': ['Q2']}}, 'Q2': {'label': 'Example Director'}})
    write_json(data / 'splits.json', {'train': ['Q1'], 'valid': [], 'test': []})
    write_json(embeddings / 'mapping.json', {'entity_to_id': {'Q1': 0, 'Q2': 1}, 'relation_to_id': {'P57': 0}})
    np.savez(embeddings / 'embeddings.npz', entity=np.zeros((2, 100)), relation=np.zeros((1, 100)))
    graph = GraphData(True, data, embeddings)
    assert graph.facts('Q1')[0][1].shape == (300,)
    rows = [{'question': 'Who directed Example Film?', 'reference_answers': ['Example Director.'], 'movie_id': 'Q1'}]
    dataset = QADataset(rows, graph, link_questions=True)
    assert dataset.linked_movie_ids == ['Q1'] and 'Example Director' in dataset[0]['text']


def test_wrong_embedding_mapping_is_rejected_instead_of_losing_new_facts(tmp_path):
    data, embeddings = tmp_path / 'dataset', tmp_path / 'embeddings'
    write_json(data / 'entities.json', {'Q1': {'label': 'Movie', 'facts': {'P57': ['Q2']}}, 'Q2': {'label': 'Director'}})
    write_json(embeddings / 'mapping.json', {'entity_to_id': {'Q1': 0}, 'relation_to_id': {'P57': 0}})
    np.savez(embeddings / 'embeddings.npz', entity=np.zeros((1, 100)), relation=np.zeros((1, 100)))
    with pytest.raises(ValueError, match='does not cover'):
        GraphData(True, data, embeddings)


@pytest.mark.parametrize('warm_start', [False, True])
def test_training_reads_selected_qa_and_records_checkpoint_before_model_loading(tmp_path, monkeypatch, warm_start):
    data = tmp_path / 'new_data'
    write_json(data / 'entities.json', {'Q1': {'label': 'Movie'}})
    write_json(data / 'splits.json', {'train': ['Q1'], 'valid': [], 'test': []})
    write_jsonl(data / 'qa.jsonl', [{'split': split, 'question': 'A new natural question'} for split in ('train', 'valid', 'test')])
    checkpoint = tmp_path / 'previous/checkpoint'
    checkpoint.mkdir(parents=True)
    monkeypatch.setattr(neural, 'ROOT', tmp_path)
    monkeypatch.setattr(neural, 'seed_all', lambda seed: None)
    monkeypatch.setattr(neural.torch.cuda, 'is_available', lambda: False)
    loaded = []
    def stop_at_tokenizer(source):
        loaded.append(source)
        raise RuntimeError('TEST_STOP_BEFORE_MODEL_LOAD')
    monkeypatch.setattr(neural.AutoTokenizer, 'from_pretrained', stop_at_tokenizer)
    with pytest.raises(RuntimeError, match='TEST_STOP'):
        neural.train('new_run', 1e-4, use_embeddings=False, data_dir=data,
                     initial_checkpoint=checkpoint if warm_start else None)
    config = read_json(tmp_path / 'runs/new_run/config.json')
    assert config['qa_by_split'] == {'train': 1, 'valid': 1, 'test': 1}
    assert config['input_hashes']['qa.jsonl'] == sha256(data / 'qa.jsonl')
    assert not config['optimizer_state_resumed']
    assert loaded == [str(checkpoint) if warm_start else 'google/flan-t5-small']
