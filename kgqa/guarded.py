"""Frozen-model QA with explicit scope checks and auditable tool fallback.

This is a system extension, not a replacement score for the raw A0/B models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
import unicodedata

from .common import REFUSAL, RELATIONS, answer_text, write_json
from .freeform import FreeformParser
from .metrics import AnswerEntityIndex


def _surface(text):
    # Preserve punctuation and words: removing them could hide extra claims.
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


class RelationGraph:
    """Select one property from the SAME retained input facts, including its cap."""

    def __init__(self, graph, movie_id, relation):
        self.graph, self.movie_id, self.relation = graph, movie_id, relation
        self.use_embeddings = graph.use_embeddings
        pairs = graph._retained_facts(movie_id)
        facts = graph.facts(movie_id)
        if len(pairs) != len(facts) or any(pair[0] != fact[0] for pair, fact in zip(pairs, facts)):
            raise ValueError('Fact text/vector provenance does not match')
        entity = graph.entities[movie_id]
        expected_texts = {f'{entity["label"]} | {RELATIONS[relation]} | {graph.entities[tail]["label"]}'
                          for tail in entity.get('facts', {}).get(relation, [])
                          if graph.entities.get(tail, {}).get('label')}
        # _retained_facts returns (TEXT, tail_id), not (property_id, tail_id).
        selected = [(pair, fact) for pair, fact in zip(pairs, facts) if pair[0] in expected_texts]
        self.pairs, self.items = [x[0] for x in selected], [x[1] for x in selected]

    def facts(self, movie_id):
        return self.items if movie_id == self.movie_id else []

    def fact_tail_ids(self, movie_id):
        return [tail for _, tail in self.pairs] if movie_id == self.movie_id else []


class GuardedQA:
    """A bounded adapter + strict evidence checker, optionally using KB templates."""

    def __init__(self, demo):
        self.demo = demo
        self.parser = FreeformParser(demo.movie_ids, demo.entities, demo.linker)
        self.index = AnswerEntityIndex(demo.entities)

    def _neural(self, question, movie_id, relation):
        from torch.utils.data import DataLoader
        from .neural import Collator, QADataset, predictions

        demo = self.demo
        if demo.neural is None:
            demo._load_neural()
        # Never combine a custom A0 snapshot with an unrelated neural graph.
        if demo.graph.entities != demo.entities:
            raise ValueError('Neural graph and parser dataset do not match')
        graph = RelationGraph(demo.graph, movie_id, relation)
        row = {'question_id': 'guarded-interactive', 'question': question, 'movie_id': movie_id,
               'reference_answers': [REFUSAL], 'answer_ids': [], 'relation_id': 'UNKNOWN',
               'is_answerable': False}
        loader = DataLoader(QADataset([row], graph), batch_size=1,
                            collate_fn=Collator(demo.tokenizer, graph.use_embeddings,
                                                demo.neural_config.get('input_max_length', 384)))
        generated = predictions(demo.neural, loader, demo.tokenizer, demo.entities, demo.device)[0]
        return {'system': f'{demo.neural_run} + 无训练增强', 'answer': generated['answer'],
                'latency_ms': generated['latency_ms'],
                'model_evidence': [item[0] for item in graph.items],
                'visible_answer_ids': graph.fact_tail_ids(movie_id)}

    def answer(self, question, system='a0', fallback=True):
        if system not in {'a0', 'neural'}:
            raise ValueError('system must be a0 or neural')
        started = time.perf_counter()
        parsed = self.parser.parse(question)
        movie, relation = parsed['movie_id'], parsed['relation_id']
        result = {'system': f'{system} + 无训练增强', 'movie_id': movie, 'relation': relation,
                  'answer': REFUSAL, 'answer_ids': [], 'evidence': [], 'refused': True,
                  'raw_answer': None, 'fallback_used': False, 'model_question': parsed['canonical_question'],
                  'guard': {'method': 'evidence_guard_v1', 'action': 'blocked',
                            'reason': parsed.get('reason_code') or parsed['status'],
                            'expected_ids': [], 'observed_ids': []}}

        def finish():
            result['latency_ms'] = (time.perf_counter() - started) * 1000
            return {'question': question, 'mode': 'guarded', 'interpretation': parsed, 'result': result}

        if parsed['status'] != 'ready':
            result['guard']['action'] = 'clarify' if parsed['status'] == 'needs_clarification' else 'blocked'
            return finish()
        # A canonical question must still link to the same entity. No hidden
        # injection of the parser's chosen ID into an ambiguous baseline input.
        if self.demo.linker.link(parsed['canonical_question'])[0] != movie:
            parsed.update(status='needs_clarification', message='标准片名仍有歧义，请先确认电影版本。')
            result['guard'].update(action='clarify', reason='canonical_link_mismatch')
            return finish()
        ids = list(dict.fromkeys(tail for tail in self.demo.entities[movie].get('facts', {}).get(relation, [])
                                 if self.demo.entities.get(tail, {}).get('label')))
        if not ids:
            parsed.update(status='unanswerable', message='知识库缺少这项可显示的事实。')
            result['guard'].update(reason='missing_evidence')
            return finish()
        raw = (self.demo.answer_a0(parsed['canonical_question']) if system == 'a0'
               else self._neural(parsed['canonical_question'], movie, relation))
        result.update(raw_answer=raw['answer'], model_evidence=raw.get('model_evidence', raw.get('evidence', [])),
                      generation_ms=raw.get('latency_ms'))
        # Guarded generation may only use the retained model input. In hybrid
        # mode the full property query is a separately labelled tool operation.
        visible = raw.get('visible_answer_ids', ids)
        observed = self.index.match(raw['answer'], movie, preferred_ids=visible)
        names = [self.demo.entities[tail]['label'] for tail in ids]
        title = self.demo.entities[movie]['label']
        allowed = {_surface(answer_text(head, relation, names, variant))
                   for head in {title, self.demo.entities[movie].get('question_title', title)}
                   for variant in (0, 1)}
        reason = 'verified'
        if raw['answer'] == REFUSAL:
            reason = 'model_refusal'
        elif set(observed) - set(ids):
            reason = 'outside_relation_evidence'
        elif set(observed) != set(ids):
            reason = 'missing_or_unrecognized_entities'
        elif set(ids) - set(visible):
            reason = 'evidence_truncated'
        elif _surface(raw['answer']) not in allowed:
            # Entity overlap alone cannot establish correctness of arbitrary
            # prose, negation, wrong predicates or extra unsupported claims.
            reason = 'unverified_surface'
        result['guard'].update(reason=reason, expected_ids=ids, observed_ids=observed,
                               visible_ids=visible)
        if reason == 'verified':
            result.update(answer=raw['answer'], answer_ids=observed, refused=False)
            result['guard']['action'] = 'accepted'
        elif fallback:
            result.update(answer=answer_text(title, relation, names), answer_ids=ids,
                          refused=False, fallback_used=True)
            result['guard']['action'] = 'tool_fallback'
        else:
            result['guard']['action'] = 'rejected'
        if not result['refused']:
            result['evidence'] = [f'{title} | {RELATIONS[relation]} | {self.demo.entities[tail]["label"]}'
                                  for tail in ids]
        # Chinese rendering follows the FINAL answer IDs after explicit tool
        # intervention, while preserving the untouched model text above.
        outside_title = re.sub(r'《[^》]+》', '', question)
        if any('\u3400' <= char <= '\u9fff' for char in outside_title):
            result['language'] = 'zh'
            result['original_answer'] = result['answer']
            result['answer'] = (self.demo.chinese.format_answer(movie, relation, result['answer_ids'])
                                if result['answer_ids'] else self.demo.chinese.localize_generated(REFUSAL, movie))
        return finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question', required=True)
    parser.add_argument('--system', choices=['a0', 'neural'], default='a0')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--no-fallback', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.system == 'neural' and (args.data_dir is not None or args.model_path is not None):
        parser.error('神经模式使用默认配套数据；自定义数据只用于 A0。')
    from .demo import Demo
    demo = Demo(load_neural=False, data_dir=args.data_dir, model_path=args.model_path)
    value = GuardedQA(demo).answer(args.question, args.system, not args.no_fallback)
    if args.output:
        write_json(args.output, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
