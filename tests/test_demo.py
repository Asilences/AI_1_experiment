import builtins
import importlib
import sys

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
