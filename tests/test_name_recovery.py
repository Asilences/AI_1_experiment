from kgqa.common import REFUSAL
from kgqa.name_recovery import NameRecovery


class LossyTokenizer:
    unk_token_id = 0

    def __init__(self):
        self.calls = []

    def __call__(self, text_target, **kwargs):
        self.calls.append(text_target)
        return {'input_ids': [0 if char in 'ãíŌ' else ord(char) + 2 for char in text_target] + [1]}

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(item - 2) for item in ids if item > 1)


def make_recovery(labels):
    tokenizer = LossyTokenizer()
    return NameRecovery({qid: {'label': label} for qid, label in labels.items()}, tokenizer), tokenizer


def test_recovers_only_emitted_full_name_and_records_original_spans():
    recovery, tokenizer = make_recovery({'D1': 'Alfonso Ungría', 'D2': 'João César Monteiro'})
    raw = 'It was directed by Alfonso Ungra, with Joo César Monteiro.'
    result = recovery.recover(raw, ['D1', 'D2'])
    assert result['raw_answer'] == raw
    assert result['answer'] == 'It was directed by Alfonso Ungría, with João César Monteiro.'
    assert [item['entity_id'] for item in result['replacements']] == ['D1', 'D2']
    for item in result['replacements']:
        assert raw[item['start']:item['end']] == item['before']
    recovery.recover(raw, ['D1', 'D1', 'D2'])
    assert tokenizer.calls == ['Alfonso Ungría', 'João César Monteiro']


def test_empty_candidates_refusal_and_omitted_answer_are_unchanged():
    recovery, _ = make_recovery({'D': 'João'})
    for text, candidates in [('Joo', []), (REFUSAL, ['D']), ('The director is unknown.', ['D'])]:
        assert recovery.recover(text, candidates) == {
            'raw_answer': text, 'answer': text, 'replacements': []}


def test_colliding_damaged_names_and_real_names_are_not_guessed():
    for labels in ({'A': 'João', 'B': 'Joío'}, {'A': 'João', 'B': 'Joo'},
                   {'A': 'João', 'B': 'João'}):
        recovery, _ = make_recovery(labels)
        assert recovery.recover('Joo directed it.', labels)['answer'] == 'Joo directed it.'


def test_no_unknown_token_does_not_correct_generation_errors_or_normalization():
    recovery, _ = make_recovery({'D': 'Gérard Oury'})
    assert recovery.recover('Gerd Oury directed it.', ['D'])['replacements'] == []

    class NormalizingTokenizer(LossyTokenizer):
        def decode(self, ids, skip_special_tokens=True):
            return super().decode(ids, skip_special_tokens).lower()

    recovery = NameRecovery({'D': {'label': 'Jane Doe'}}, NormalizingTokenizer())
    assert recovery.recover('jane doe', ['D'])['replacements'] == []


def test_word_boundaries_correct_names_and_short_fragments_are_protected():
    recovery, _ = make_recovery({'D': 'João', 'P': 'Joo Smith', 'S': 'Ōta'})
    raw = 'João; Joo Smith; SuperJoo; Jooish; ta; (joo).'
    result = recovery.recover(raw, ['D', 'P', 'S'])
    assert result['answer'] == 'João; Joo Smith; SuperJoo; Jooish; ta; (João).'
    assert len(result['replacements']) == 1


def test_longest_match_wins_and_replacement_is_not_reprocessed():
    recovery, _ = make_recovery({'A': 'João', 'B': 'João César Monteiro'})
    result = recovery.recover('Joo César Monteiro; Joo.', ['A', 'B'])
    assert result['answer'] == 'João César Monteiro; João.'
    assert [item['entity_id'] for item in result['replacements']] == ['B', 'A']
    assert recovery.recover(result['answer'], ['A', 'B'])['replacements'] == []


def test_unknown_and_unlabelled_candidates_are_inert():
    recovery = NameRecovery({'empty': {}}, LossyTokenizer())
    assert recovery.recover('Joo', ['missing', 'empty'])['replacements'] == []
