"""Inspect neural data/tokenizer limits without loading a model or scoring it."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .common import RELATIONS, read_json, read_jsonl, sha256, write_json
from .linker import EntityLinker


INPUT_LIMIT, TARGET_LIMIT, FACT_LIMIT = 384, 96, 24
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                   'spiece.model', 'added_tokens.json')


def _token_ids(tokenizer, text, *, target=False):
    arguments = {'text_target': text} if target else {'text': text}
    return tokenizer(**arguments, truncation=False, add_special_tokens=True)['input_ids']


def _roundtrip(tokenizer, text):
    ids = _token_ids(tokenizer, text, target=True)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    unk = getattr(tokenizer, 'unk_token_id', None)
    return {'tokens': len(ids), 'decoded': decoded, 'changed': decoded != text,
            'unknown_tokens': ids.count(unk) if unk is not None else 0}


def _facts(movie_id, entities, mapping=None):
    # Follow GraphData.facts exactly, before its slice. Importing neural.py would
    # import torch; share RELATIONS/linker and record the source hash instead.
    if not movie_id:
        return []
    entity, facts = entities[movie_id], []
    for relation in RELATIONS:
        for tail in entity.get('facts', {}).get(relation, []):
            if tail not in entities or not entities[tail].get('label'):
                continue
            if mapping is not None and (
                movie_id not in mapping['entity_to_id'] or tail not in mapping['entity_to_id']
                or relation not in mapping['relation_to_id']
            ):
                continue
            facts.append((f"{entity['label']} | {RELATIONS[relation]} | {entities[tail]['label']}", tail))
    return facts


def _input_text(question, facts):
    return f'question: {question} facts: ' + '; '.join(text for text, _ in facts)


def tokenizer_identity(tokenizer, source=None):
    identity = {'class': type(tokenizer).__name__}
    if source is not None:
        path = Path(source)
        if path.is_dir():
            identity.update(source_kind='local_directory', source_name=path.name,
                            files_sha256={name: sha256(path / name) for name in TOKENIZER_FILES
                                          if (path / name).is_file()})
        else:
            identity.update(source_kind='cached_model', source_name=str(source))
    if hasattr(tokenizer, 'get_vocab'):
        vocab = tokenizer.get_vocab()
        payload = json.dumps(vocab, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
        identity.update(vocabulary_size=len(vocab), vocabulary_sha256=hashlib.sha256(payload).hexdigest())
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    if backend is not None:
        identity['backend_sha256'] = hashlib.sha256(backend.to_str().encode('utf-8')).hexdigest()
    return identity


def _label_summary(ids, entities, roundtrips, example_limit):
    known = sorted(qid for qid in ids if entities.get(qid, {}).get('label'))
    changed = [qid for qid in known if roundtrips[qid]['changed']]
    return {'unique_entities': len(known), 'roundtrip_changed_entities': len(changed),
            'unknown_token_entities': sum(roundtrips[qid]['unknown_tokens'] > 0 for qid in known),
            'examples': [{'entity_id': qid, 'label': entities[qid]['label'], **roundtrips[qid]}
                         for qid in changed[:example_limit]]}


def build_report(data_dir, tokenizer, *, tokenizer_source=None, mapping_path=None, example_limit=5):
    if not isinstance(example_limit, int) or isinstance(example_limit, bool) or example_limit < 0:
        raise ValueError('example_limit must be a nonnegative integer')
    data_dir = Path(data_dir)
    entities, rows, splits = (read_json(data_dir / 'entities.json'), read_jsonl(data_dir / 'qa.jsonl'),
                              read_json(data_dir / 'splits.json'))
    if not rows:
        raise ValueError('qa.jsonl must contain at least one question')
    mapping = read_json(mapping_path) if mapping_path is not None else None
    linker = EntityLinker(sum(splits.values(), []), entities)
    label_roundtrips = {qid: _roundtrip(tokenizer, entity['label']) for qid, entity in entities.items()
                        if entity.get('label')}
    fact_cache, target_cache, by_split = {}, {}, {}
    for split in sorted({row['split'] for row in rows}):
        selected = [row for row in rows if row['split'] == split]
        counts = Counter()
        split_labels, trimmed_movies = set(), set()
        max_input = max_target = max_facts = 0
        examples = {name: [] for name in ('input_truncation', 'target_truncation', 'target_roundtrip',
                                          'fact_truncation', 'answers_absent_from_retained_facts')}

        def example(kind, value):
            if len(examples[kind]) < example_limit:
                examples[kind].append(value)

        for row in selected:
            if not row.get('reference_answers'):
                raise ValueError(f"Question {row['question_id']!r} must have a reference answer")
            movie_id = row['movie_id'] if split == 'train' else linker.link(row['question'])[0]
            counts['linked_movie_mismatch_rows'] += movie_id != row['movie_id']
            if movie_id not in fact_cache:
                fact_cache[movie_id] = _facts(movie_id, entities, mapping)
            available = fact_cache[movie_id]
            retained = available[:FACT_LIMIT]
            input_count = len(_token_ids(tokenizer, _input_text(row['question'], retained)))
            target = row['reference_answers'][0]
            if target not in target_cache:
                target_cache[target] = _roundtrip(tokenizer, target)
            target_info = target_cache[target]
            max_input, max_target = max(max_input, input_count), max(max_target, target_info['tokens'])
            max_facts = max(max_facts, len(available))
            short = {'question_id': row['question_id']}
            if input_count > INPUT_LIMIT:
                counts['input_truncated_rows'] += 1
                example('input_truncation', {**short, 'untruncated_tokens': input_count})
            if target_info['tokens'] > TARGET_LIMIT:
                counts['target_truncated_rows'] += 1
                example('target_truncation', {**short, 'untruncated_tokens': target_info['tokens']})
            if target_info['changed']:
                counts['target_roundtrip_changed_rows'] += 1
                example('target_roundtrip', {**short, 'target': target, **target_info})
            counts['target_unknown_token_rows'] += target_info['unknown_tokens'] > 0
            if len(available) > FACT_LIMIT:
                counts['fact_truncated_rows'] += 1
                trimmed_movies.add(movie_id)
                example('fact_truncation', {**short, 'linked_movie_id': movie_id,
                                            'available_facts': len(available), 'retained_facts': len(retained)})
            available_ids = {tail for _, tail in available}
            missing = set(row['answer_ids']) - {tail for _, tail in retained}
            dropped = missing & available_ids
            if missing:
                counts['answer_missing_from_retained_facts_rows'] += 1
                counts['answer_missing_entity_occurrences'] += len(missing)
                counts['answer_dropped_by_fact_limit_rows'] += bool(dropped)
                counts['answer_dropped_by_fact_limit_entity_occurrences'] += len(dropped)
                example('answers_absent_from_retained_facts', {
                    **short, 'gold_movie_id': row['movie_id'], 'linked_movie_id': movie_id,
                    'missing_answer_ids': sorted(missing), 'dropped_by_fact_limit_ids': sorted(dropped),
                    'absent_before_fact_limit_ids': sorted(missing - available_ids),
                })
            split_labels.update([row['movie_id'], *row['answer_ids']])
            if movie_id:
                split_labels.add(movie_id)
            split_labels.update(tail for _, tail in available)
        count_keys = ('linked_movie_mismatch_rows', 'input_truncated_rows', 'target_truncated_rows',
                      'target_roundtrip_changed_rows', 'target_unknown_token_rows', 'fact_truncated_rows',
                      'answer_missing_from_retained_facts_rows', 'answer_missing_entity_occurrences',
                      'answer_dropped_by_fact_limit_rows', 'answer_dropped_by_fact_limit_entity_occurrences')
        by_split[split] = {
            'rows': len(selected), 'gold_movies': len({row['movie_id'] for row in selected}),
            **{key: counts[key] for key in count_keys},
            'max_untruncated_input_tokens': max_input, 'max_untruncated_target_tokens': max_target,
            'max_available_facts': max_facts, 'fact_truncated_unique_movies': len(trimmed_movies),
            'entity_labels': _label_summary(split_labels, entities, label_roundtrips, example_limit),
            'examples': examples,
        }
    inputs = {name: sha256(data_dir / name) for name in ('entities.json', 'qa.jsonl', 'splits.json')}
    if mapping_path is not None:
        inputs['embedding_mapping.json'] = sha256(mapping_path)
    source_dir = Path(__file__).parent
    return {
        'schema_version': 1, 'kind': 'neural_data_preflight',
        'scope': 'Dataset/tokenizer inspection only; no model training, generation, or evaluation scores.',
        'protocol': {
            'input_max_length': INPUT_LIMIT, 'target_max_length': TARGET_LIMIT, 'fact_limit': FACT_LIMIT,
            'token_counts_include_special_tokens': True, 'target_reference_index': 0,
            'roundtrip_comparison': 'Exact equality after decode(skip_special_tokens=True), without truncation.',
            'context_selection': 'Gold movie for train; EntityLinker over all split movies for other splits.',
            'fact_filter': 'TransE mapping filter' if mapping is not None else 'Text facts without embedding mapping filter (C0).',
            'label_scope_by_split': 'Gold movie/answers plus linked movie and available fact tails before the slice.',
            'missing_answers_note': 'Check tail entity IDs. Separate 24-fact loss from prior absence (including linking/mapping). Token truncation can remove further text.',
        },
        'provenance': {'input_sha256': inputs, 'tokenizer': tokenizer_identity(tokenizer, tokenizer_source),
                       'source_sha256': {name: sha256(source_dir / name) for name in
                                         ('neural.py', 'neural_preflight.py', 'common.py', 'linker.py')}},
        'rows': len(rows), 'entity_labels': _label_summary(set(entities), entities, label_roundtrips, example_limit),
        'by_split': by_split,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path, help='Processed directory with entities.json, qa.jsonl, splits.json.')
    parser.add_argument('--tokenizer', required=True, help='Local tokenizer directory or cached model ID. Never downloaded.')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mapping', type=Path, help='Optional TransE mapping.json to reproduce B0 fact filtering.')
    parser.add_argument('--example-limit', type=int, default=5)
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    report = build_report(args.data_dir, tokenizer, tokenizer_source=args.tokenizer,
                          mapping_path=args.mapping, example_limit=args.example_limit)
    write_json(args.output, report)
    print(json.dumps({'kind': report['kind'], 'rows': report['rows'],
                      'by_split': {split: {key: value for key, value in summary.items()
                                          if key not in ('examples', 'entity_labels')}
                                   for split, summary in report['by_split'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
