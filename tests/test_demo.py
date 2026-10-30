import builtins
import importlib
import sys
from types import SimpleNamespace

import pytest

from kgqa.common import write_json


def test_a0_demo_needs_no_neural_dependencies_or_selection(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def import_without_neural(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'transformers'}:
            raise AssertionError(f'A0 unexpectedly imported {name}')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', import_without_neural)
    sys.modules.pop('kgqa.demo', None)
    demo_module = importlib.import_module('kgqa.demo')
    entities = {'Q1': {'label': 'Example Film', 'facts': {'P57': ['Q2']}},
                'Q2': {'label': 'Jane Director'}}
    write_json(tmp_path / 'data' / 'processed' / 'entities.json', entities)
    write_json(tmp_path / 'data' / 'processed' / 'splits.json', {'train': ['Q1'], 'valid': [], 'test': []})

    class Classifier:
        classes_ = ['P57']

        def predict_proba(self, rows):
            return [[1.0] for _ in rows]

    monkeypatch.setattr(demo_module, 'ROOT', tmp_path)
    monkeypatch.setattr(demo_module, 'load_model', lambda: (Classifier(), .5))
    system = demo_module.Demo(load_neural=False)
    assert system.neural is None and system.neural_run is None
    assert system.answer_a0('Who directed Example Film?')['answer'] == 'Example Film was directed by Jane Director.'
    assert not (tmp_path / 'output' / 'selected_model.json').exists()


def test_a0_demo_loads_the_selected_personal_model_and_data(tmp_path, monkeypatch):
    from kgqa import demo as demo_module

    data = tmp_path / 'isolated_data'
    model = tmp_path / 'personal_run' / 'model.pkl'
    write_json(data / 'entities.json', {'M': {'label': 'Other Film', 'facts': {'P57': ['D']}},
                                      'D': {'label': 'Other Director'}})
    write_json(data / 'splits.json', {'train': ['M'], 'valid': [], 'test': []})

    class Classifier:
        classes_ = ['P57']

        def predict_proba(self, rows):
            return [[1.0] for _ in rows]

    loaded = []
    monkeypatch.setattr(demo_module, 'load_model', lambda path: (loaded.append(path) or Classifier(), .3))
    system = demo_module.Demo(load_neural=False, data_dir=data, model_path=model)
    assert loaded == [model]
    assert system.answer_a0('Who directed Other Film?')['answer'] == 'Other Film was directed by Other Director.'
    assert not (tmp_path / 'models').exists()


@pytest.mark.parametrize('option', ['data_dir', 'model_path'])
def test_custom_a0_paths_cannot_be_mixed_with_shared_neural_data(tmp_path, option):
    from kgqa.demo import Demo

    with pytest.raises(ValueError, match='require --system a0'):
        Demo(load_neural=True, **{option: tmp_path / 'custom'})


def recovery_demo(enabled=True):
    from kgqa.demo import Demo
    demo = Demo.__new__(Demo)
    demo.recover_names, demo.name_recovery = enabled, None
    demo.entities = {'M': {'label': 'Example Film', 'facts': {'P57': [f'D{i}' for i in range(26)]}}}
    demo.entities.update({f'D{i}': {'label': f'Director {i}'} for i in range(26)})
    demo.tokenizer = object()
    demo.graph = SimpleNamespace(use_embeddings=True,
                                 entity_to_id={qid: i for i, qid in enumerate(demo.entities) if qid != 'D0'},
                                 relation_to_id={'P57': 0}, facts=lambda qid: [('observed fact', None)])
    return demo


def test_recovery_is_opt_in_and_candidates_follow_mapping_and_fact_limit(monkeypatch):
    captured = []

    class FakeRecovery:
        def __init__(self, entities, tokenizer):
            captured.append('created')

        def recover(self, raw_answer, candidate_ids):
            captured.append(candidate_ids)
            return {'raw_answer': raw_answer, 'answer': 'restored', 'replacements': [{'entity_id': 'D1'}]}

    monkeypatch.setitem(sys.modules, 'kgqa.name_recovery', SimpleNamespace(NameRecovery=FakeRecovery))
    baseline = recovery_demo(False)
    assert baseline._recover_generated('original', 'M') == {'answer': 'original'}
    assert captured == []
    demo = recovery_demo()
    result = demo._recover_generated('original', 'M')
    assert captured == ['created', [f'D{i}' for i in range(1, 25)]]
    assert result['raw_answer'] == 'original' and result['answer'] == 'restored'
    assert result['recovery_ms'] >= 0
    demo._recover_generated('another', 'M')
    assert captured.count('created') == 1


def test_unlinked_question_returns_refusal_without_recovery_or_generation():
    from kgqa.common import REFUSAL
    demo = recovery_demo()
    demo.neural, demo.neural_run = object(), 'B0'
    demo.linker = SimpleNamespace(link=lambda question: (None, 0, 'unlinked'))
    result = demo.answer_neural('Unknown movie')
    assert result['answer'] == result['raw_answer'] == REFUSAL
    assert result['replacements'] == [] and result['recovery_ms'] == 0
    assert demo.name_recovery is None
    assert result['system'] == 'B0 神经生成系统 + 名称恢复'


def test_live_neural_flow_preserves_generation_latency_and_raw_answer(monkeypatch):
    demo = recovery_demo()
    demo.neural, demo.neural_run, demo.device = object(), 'B0', object()
    demo.linker = SimpleNamespace(link=lambda question: ('M', 1, 'exact'))
    demo.name_recovery = SimpleNamespace(recover=lambda raw, ids: {
        'raw_answer': raw, 'answer': 'restored', 'replacements': [{'entity_id': ids[0]}]})
    inputs = []
    monkeypatch.setitem(sys.modules, 'torch.utils.data', SimpleNamespace(DataLoader=lambda dataset, **kw: dataset))
    monkeypatch.setitem(sys.modules, 'kgqa.neural', SimpleNamespace(
        QADataset=lambda rows, graph: inputs.extend(rows) or rows,
        Collator=lambda *args: None,
        predictions=lambda *args: [{'answer': 'generated', 'latency_ms': 12.5}],
    ))
    result = demo.answer_neural('Who directed Example Film?')
    assert result['answer'] == 'restored' and result['raw_answer'] == 'generated'
    assert result['latency_ms'] == 12.5 and result['recovery_ms'] >= 0
    assert result['replacements'][0]['entity_id'] == 'D1'
    assert inputs[0]['movie_id'] == 'M' and inputs[0]['answer_ids'] == []


def test_run_passes_recovery_option_and_rejects_a0_only(monkeypatch):
    from kgqa import demo as module
    received = []

    class FakeDemo:
        def __init__(self, **kwargs):
            received.append(kwargs)

        def answer_neural(self, question):
            return {}

    monkeypatch.setattr(module, 'Demo', FakeDemo)
    monkeypatch.setattr(module, 'print_result', lambda result: None)
    module.run('question', 'best', recover_names=True)
    assert received == [{'load_neural': True, 'data_dir': None, 'model_path': None, 'recover_names': True}]
    with pytest.raises(ValueError, match='requires --system'):
        module.run('question', 'a0', recover_names=True)
