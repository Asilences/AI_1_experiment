import builtins
import importlib
import sys

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
