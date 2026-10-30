"""Exercise collation limits without importing optional GPU dependencies."""
import ast
from pathlib import Path

import numpy as np
import pytest


def collator_class():
    tree = ast.parse((Path(__file__).parents[1] / 'kgqa/neural.py').read_text(encoding='utf-8'))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Collator')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<Collator>', 'exec'), namespace)
    return namespace['Collator']


class Tokenizer:
    pad_token_id = 0

    def __init__(self): self.limits = []

    def __call__(self, texts=None, **kwargs):
        self.limits.append(kwargs['max_length'])
        return {'input_ids': np.array([[1, 0]]), 'attention_mask': np.array([[1, 0]])}


@pytest.mark.parametrize('length', [384, 512])
def test_input_limit_is_explicit_and_target_remains_96(length):
    tokenizer = Tokenizer()
    result = collator_class()(tokenizer, False, length)([{'text': 'question', 'target': 'answer', 'row': {'id': 1}}])
    assert tokenizer.limits == [length, 96]
    assert result['labels'].tolist() == [[1, -100]]
    assert result['rows'] == [{'id': 1}]


@pytest.mark.parametrize('length', [0, -1, True, 1.5])
def test_invalid_input_limit_is_rejected(length):
    with pytest.raises(ValueError, match='positive integer'):
        collator_class()(Tokenizer(), False, length)
