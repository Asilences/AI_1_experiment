from copy import deepcopy
import json

from kgqa import neural_preflight as preflight
from kgqa.common import sha256, write_json, write_jsonl


class CharacterTokenizer:
    """A tiny tokenizer that cannot encode ã, with one EOS token."""
    unk_token_id = 0

    def __init__(self):
        self.calls = []

    def __call__(self, text=None, text_target=None, **kwargs):
        self.calls.append((text, text_target, kwargs))
        value = text_target if text_target is not None else text
        return {'input_ids': [0 if char == 'ã' else ord(char) + 2 for char in value] + [1]}

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(item - 2) for item in ids if item > 1)

    def get_vocab(self):
        return {'<unk>': 0, '</s>': 1, 'a': 99}


def dataset(tmp_path, *, question='Who directed Film?', target='João', facts=1):
    entities = {'Q1': {'label': 'Film', 'aliases': [], 'facts': {'P57': [f'T{i}' for i in range(facts)]}}}
    entities.update({f'T{i}': {'label': 'João' if i == facts - 1 else f'Director{i}'} for i in range(facts)})
    rows = [{'question_id': f'q_{split}', 'split': split, 'movie_id': 'Q1',
             'question': question, 'answer_ids': [f'T{facts - 1}'], 'reference_answers': [target]}
            for split in ('train', 'valid', 'test')]
    write_json(tmp_path / 'entities.json', entities)
    write_json(tmp_path / 'splits.json', {'train': ['Q1'], 'valid': [], 'test': []})
    write_jsonl(tmp_path / 'qa.jsonl', rows)
    return entities, rows


def test_unicode_loss_is_diagnostic_and_data_are_hashed(tmp_path):
    dataset(tmp_path)
    tokenizer = CharacterTokenizer()
    report = preflight.build_report(tmp_path, tokenizer, example_limit=1)
    assert report['kind'] == 'neural_data_preflight'
    assert 'no model training' in report['scope']
    assert report['provenance']['input_sha256']['qa.jsonl'] == sha256(tmp_path / 'qa.jsonl')
    for summary in report['by_split'].values():
        assert summary['target_roundtrip_changed_rows'] == summary['target_unknown_token_rows'] == 1
        assert summary['examples']['target_roundtrip'][0]['decoded'] == 'Joo'
        assert summary['entity_labels']['roundtrip_changed_entities'] == 1
        assert summary['answer_missing_from_retained_facts_rows'] == 0
    assert all(call[2]['truncation'] is False for call in tokenizer.calls)
    assert all(call[2]['add_special_tokens'] is True for call in tokenizer.calls)


def test_exact_384_and_96_token_boundaries_include_eos(tmp_path):
    tokenizer = CharacterTokenizer()
    plain = preflight._input_text('Film', [('Film | director | João', 'T0')])
    padding = 383 - len(plain)
    _, rows = dataset(tmp_path, question='Film' + 'x' * padding, target='a' * 95)
    at_limit = preflight.build_report(tmp_path, tokenizer)['by_split']['train']
    assert at_limit['max_untruncated_input_tokens'] == 384
    assert at_limit['max_untruncated_target_tokens'] == 96
    assert at_limit['input_truncated_rows'] == at_limit['target_truncated_rows'] == 0
    rows[0]['question'] += 'x'
    rows[0]['reference_answers'][0] += 'a'
    write_jsonl(tmp_path / 'qa.jsonl', rows)
    above = preflight.build_report(tmp_path, tokenizer)['by_split']['train']
    assert above['input_truncated_rows'] == above['target_truncated_rows'] == 1
    assert above['examples']['input_truncation'][0]['untruncated_tokens'] == 385
    assert above['examples']['target_truncation'][0]['untruncated_tokens'] == 97


def test_fact_limit_separates_lost_answers_from_mapping_and_linker_absence(tmp_path):
    entities, rows = dataset(tmp_path, facts=25)
    report = preflight.build_report(tmp_path, CharacterTokenizer(), example_limit=1)
    summary = report['by_split']['train']
    assert summary['max_available_facts'] == 25
    assert summary['fact_truncated_rows'] == summary['fact_truncated_unique_movies'] == 1
    assert summary['answer_missing_from_retained_facts_rows'] == summary['answer_dropped_by_fact_limit_rows'] == 1
    assert summary['examples']['answers_absent_from_retained_facts'][0]['dropped_by_fact_limit_ids'] == ['T24']
    mapping = {'entity_to_id': {qid: i for i, qid in enumerate(entities) if qid != 'T24'},
               'relation_to_id': {'P57': 0}}
    write_json(tmp_path / 'mapping.json', mapping)
    mapped = preflight.build_report(tmp_path, CharacterTokenizer(), mapping_path=tmp_path / 'mapping.json')
    summary = mapped['by_split']['train']
    assert summary['fact_truncated_rows'] == summary['answer_dropped_by_fact_limit_rows'] == 0
    assert summary['answer_missing_from_retained_facts_rows'] == 1
    assert summary['examples']['answers_absent_from_retained_facts'][0]['absent_before_fact_limit_ids'] == ['T24']
    rows[1]['question'] = 'No recognizable movie title'
    write_jsonl(tmp_path / 'qa.jsonl', rows)
    unlinked = preflight.build_report(tmp_path, CharacterTokenizer())['by_split']['valid']
    assert unlinked['linked_movie_mismatch_rows'] == unlinked['answer_missing_from_retained_facts_rows'] == 1
    assert unlinked['answer_dropped_by_fact_limit_rows'] == 0
    assert unlinked['examples']['answers_absent_from_retained_facts'][0]['linked_movie_id'] is None


def test_text_order_empty_facts_and_example_limit(tmp_path):
    entities, rows = dataset(tmp_path)
    entities['Q1']['facts'] = {'P364': ['T0'], 'P57': ['T0'], 'P136': ['missing'], 'P495': ['empty']}
    entities['empty'] = {'label': ''}
    assert preflight._facts('Q1', entities) == [
        ('Film | director | João', 'T0'), ('Film | original language | João', 'T0')]
    assert preflight._input_text('Who?', []) == 'question: Who? facts: '
    rows.extend({**deepcopy(rows[0]), 'question_id': f'extra{i}'} for i in range(3))
    write_jsonl(tmp_path / 'qa.jsonl', rows)
    report = preflight.build_report(tmp_path, CharacterTokenizer(), example_limit=1)
    assert report['by_split']['train']['target_roundtrip_changed_rows'] == 4
    assert len(report['by_split']['train']['examples']['target_roundtrip']) == 1
    assert preflight.build_report(tmp_path, CharacterTokenizer(), example_limit=0)['entity_labels']['examples'] == []


def test_local_tokenizer_identity_omits_absolute_path_and_tracks_contents(tmp_path):
    dataset(tmp_path)
    tokenizer_dir = tmp_path / 'private-machine-name' / 'checkpoint'
    tokenizer_dir.mkdir(parents=True)
    write_json(tokenizer_dir / 'tokenizer.json', {'local': 'example'})
    identity = preflight.tokenizer_identity(CharacterTokenizer(), tokenizer_dir)
    assert identity['source_name'] == 'checkpoint'
    assert identity['files_sha256']['tokenizer.json'] == sha256(tokenizer_dir / 'tokenizer.json')
    assert 'private-machine-name' not in json.dumps(identity)
    report = preflight.build_report(tmp_path, CharacterTokenizer(), tokenizer_source=tokenizer_dir)
    assert str(tmp_path) not in json.dumps(report)
