import pytest

from kgqa.common import read_json, write_json, write_jsonl
from kgqa.recovery_experiment import recover_predictions, run


class Tokenizer:
    unk_token_id = 0

    def __call__(self, text_target, **kwargs):
        return {'input_ids': [0 if c == 'ã' else ord(c) for c in text_target]}

    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids if i)


def fixture(tmp_path):
    data = tmp_path / 'data'
    entities = {'M': {'label': 'Film', 'facts': {'P57': ['D']}, 'aliases': []}, 'D': {'label': 'João Director'}}
    write_json(data / 'entities.json', entities)
    write_json(data / 'splits.json', {'train': ['M'], 'valid': [], 'test': []})
    rows = [{'question_id': split, 'split': split, 'question': 'Who directed Film?', 'movie_id': 'M',
             'relation_id': 'P57', 'answer_ids': ['D'], 'reference_answers': ['João Director.']}
            for split in ('valid', 'test')]
    write_jsonl(data / 'qa.jsonl', rows)
    for split in ('valid', 'test'):
        write_jsonl(tmp_path / (split + '.jsonl'), [{'question_id': split, 'question': 'Who directed Film?',
                                                   'answer': 'Joo Director.', 'answer_ids': [], 'latency_ms': 5}])
    return data, entities


def test_inference_uses_question_link_not_injected_gold(tmp_path):
    _, entities = fixture(tmp_path)
    prediction = {'question_id': 'x', 'question': 'Unknown title?', 'answer': 'Joo Director.',
                  'movie_id': 'M', 'answer_ids': ['D'], 'relation_id': 'P57'}
    result = recover_predictions([prediction], entities, ['M'], Tokenizer())[0]
    assert result['answer'] == prediction['answer']
    assert result['linked_movie_id'] is None and not result['replacements']


def test_validation_freezes_protocol_then_test_uses_strict_scores(tmp_path):
    data, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='requires a successful validation'):
        run(data, tmp_path / 'test.jsonl', Tokenizer(), tmp_path / 'premature', 'test')
    valid_dir = tmp_path / 'valid-result'
    valid = run(data, tmp_path / 'valid.jsonl', Tokenizer(), valid_dir, 'valid')
    assert valid['selected_on_validation'] and valid['fixed_question_ids'] == ['valid']
    report = run(data, tmp_path / 'test.jsonl', Tokenizer(), tmp_path / 'test-result', 'test',
                 validation_report=valid_dir / 'report.json')
    assert report['baseline']['entity_exact'] == 0
    assert report['with_name_recovery']['entity_exact'] == 100
    assert not report['regressed_question_ids']
    with pytest.raises(FileExistsError):
        run(data, tmp_path / 'valid.jsonl', Tokenizer(), valid_dir, 'valid')
    selected = read_json(valid_dir / 'report.json')
    selected['protocol']['fact_limit'] = 999
    write_json(valid_dir / 'report.json', selected)
    with pytest.raises(ValueError, match='Protocol changed'):
        run(data, tmp_path / 'test.jsonl', Tokenizer(), tmp_path / 'drift', 'test',
            validation_report=valid_dir / 'report.json')


def test_question_text_must_match_even_when_ids_match(tmp_path):
    data, _ = fixture(tmp_path)
    write_jsonl(tmp_path / 'valid.jsonl', [{'question_id': 'valid', 'question': 'Different Film?',
                                         'answer': 'Joo Director.', 'answer_ids': []}])
    with pytest.raises(ValueError, match='question text'):
        run(data, tmp_path / 'valid.jsonl', Tokenizer(), tmp_path / 'result', 'valid')
