from __future__ import annotations

import argparse
from pathlib import Path
import time

from .common import REFUSAL, ROOT, read_json
from .linker import EntityLinker
from .pipeline import PipelineQA, load_model


def entity_label(entities, qid):
    if not qid:
        return '未识别'
    return f"{entities.get(qid, {}).get('label', qid)} ({qid})"


class Demo:
    def __init__(self, load_neural=True, data_dir=None, model_path=None, recover_names=False):
        if (load_neural or recover_names) and (data_dir is not None or model_path is not None):
            raise ValueError('Custom data/model paths require --system a0')
        self.recover_names = recover_names
        self.name_recovery = None
        data = Path(data_dir) if data_dir is not None else ROOT / 'data' / 'processed'
        self.entities = read_json(data / 'entities.json')
        self.splits = read_json(data / 'splits.json')
        self.movie_ids = sum(self.splits.values(), [])
        model, threshold = load_model(Path(model_path)) if model_path is not None else load_model()
        self.pipeline = PipelineQA(self.entities, self.movie_ids, model, threshold)
        self.linker = EntityLinker(self.movie_ids, self.entities)
        self.neural_run = None
        self.neural = None
        if load_neural:
            self._load_neural()

    def _load_neural(self):
        import torch
        from transformers import AutoTokenizer
        from .neural import GraphData, GraphPrefixModel

        if self.neural_run is None:
            self.neural_run = read_json(ROOT / 'output' / 'selected_model.json')['selected_run']
        run = ROOT / 'runs' / self.neural_run
        config = read_json(run / 'config.json')
        self.neural_config = config
        self.graph = GraphData(config['use_embeddings'])
        self.tokenizer = AutoTokenizer.from_pretrained(run / 'checkpoint')
        self.neural = GraphPrefixModel(str(run / 'checkpoint'), config['use_embeddings'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.neural = self.neural.to(device)
        if config['use_embeddings']:
            state = torch.load(run / 'checkpoint' / 'graph_prefix.pt', map_location=device, weights_only=True)
            self.neural.projection.load_state_dict(state['projection'])
            self.neural.norm.load_state_dict(state['norm'])
        self.neural.eval()

    def answer_a0(self, question):
        result = self.pipeline.predict(question)
        evidence = [
            f"{entity_label(self.entities, head)} --{relation}--> {entity_label(self.entities, tail)}"
            for head, relation, tail in result.get('evidence', [])
        ]
        return {
            'system': 'A0 管道式系统',
            'movie': entity_label(self.entities, result.get('movie_id')),
            'link_status': result.get('link_status', ''),
            'relation': result.get('relation_id', 'UNKNOWN'),
            'confidence': result.get('relation_confidence'),
            'evidence': evidence,
            'answer': result['answer'],
            'latency_ms': result.get('latency_ms', 0),
        }

    def answer_neural(self, question):
        if self.neural is None:
            self._load_neural()
        system_name = f'{self.neural_run} 神经生成系统' + (' + 名称恢复' if self.recover_names else '')
        movie_id, link_score, link_status = self.linker.link(question)
        if not movie_id:
            return {'system': system_name, 'movie': '未识别', 'link_status': link_status,
                    'evidence': [], **self._recover_generated(REFUSAL, None), 'latency_ms': 0}
        from torch.utils.data import DataLoader
        from .neural import Collator, QADataset, predictions
        row = {'question_id': 'interactive', 'question': question, 'movie_id': movie_id,
               'reference_answers': [REFUSAL], 'answer_ids': [], 'relation_id': 'UNKNOWN', 'is_answerable': False}
        loader = DataLoader(QADataset([row], self.graph), batch_size=1,
                            collate_fn=Collator(self.tokenizer, self.graph.use_embeddings,
                                                getattr(self, 'neural_config', {}).get('input_max_length', 384)))
        result = predictions(self.neural, loader, self.tokenizer, self.entities, self.device)[0]
        evidence = [item[0] for item in self.graph.facts(movie_id)]
        return {'system': system_name, 'movie': entity_label(self.entities, movie_id),
                'link_status': f'{link_status}, score={link_score:.3f}', 'evidence': evidence,
                **self._recover_generated(result['answer'], movie_id), 'latency_ms': result['latency_ms']}

    def _recover_generated(self, raw_answer, movie_id):
        if not self.recover_names:
            return {'answer': raw_answer}
        if not movie_id:
            return {'answer': raw_answer, 'raw_answer': raw_answer, 'replacements': [], 'recovery_ms': 0}
        from .name_recovery import NameRecovery
        from .neural_preflight import _facts
        started = time.perf_counter()
        if self.name_recovery is None:
            self.name_recovery = NameRecovery(self.entities, self.tokenizer)
        mapping = ({'entity_to_id': self.graph.entity_to_id, 'relation_to_id': self.graph.relation_to_id}
                   if self.graph.use_embeddings else None)
        candidates = [tail for _, tail in _facts(movie_id, self.entities, mapping)[:24]]
        recovered = self.name_recovery.recover(raw_answer, candidates)
        return {**recovered, 'recovery_ms': (time.perf_counter() - started) * 1000}


def print_result(result):
    print(f"\n[{result['system']}]")
    print(f"链接电影：{result['movie']}（{result['link_status']}）")
    if result.get('relation'):
        confidence = result.get('confidence')
        suffix = f", confidence={confidence:.3f}" if confidence is not None else ''
        print(f"预测关系：{result['relation']}{suffix}")
    evidence = result.get('evidence', [])
    print('证据：')
    if evidence:
        for item in evidence[:12]: print(f'  - {item}')
        if len(evidence) > 12: print(f'  ... 另有 {len(evidence) - 12} 条')
    else:
        print('  - 无可用证据')
    print(f"答案：{result['answer']}")
    if 'raw_answer' in result:
        print(f"原始生成：{result['raw_answer']}")
        print(f"名称恢复：{len(result['replacements'])} 处，{result['recovery_ms']:.2f} ms")
    print(f"生成延迟：{result['latency_ms']:.2f} ms")


def run(question=None, system='both', data_dir=None, model_path=None, recover_names=False):
    if recover_names and system == 'a0':
        raise ValueError('--recover-names requires --system best or both')
    demo = Demo(load_neural=system in ('both', 'best'), data_dir=data_dir, model_path=model_path,
                recover_names=recover_names)
    def answer(text):
        print(f"\n问题：{text}")
        if system in ('a0', 'both'): print_result(demo.answer_a0(text))
        if system in ('best', 'both'): print_result(demo.answer_neural(text))
    if question:
        answer(question); return
    print('电影知识图谱问答演示。请输入英文问题；输入 exit 结束。')
    print('示例：Who directed Shutter Island? / What genre is Boss Level?')
    while True:
        try: text = input('\nQuestion> ').strip()
        except (EOFError, KeyboardInterrupt): break
        if text.casefold() in {'exit', 'quit', 'q'}: break
        if text: answer(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive comparison of A0 and the validation-selected neural system.')
    parser.add_argument('--question'); parser.add_argument('--system', choices=['a0', 'best', 'both'], default='both')
    parser.add_argument('--data-dir', type=Path, help='Processed data directory for A0 only.')
    parser.add_argument('--model-path', type=Path, help='A0 model.pkl from a personal or acceptance run.')
    parser.add_argument('--recover-names', action='store_true', help='Restore unambiguous tokenizer-damaged names from retrieved facts; retain raw output.')
    args = parser.parse_args()
    if args.system != 'a0' and (args.data_dir is not None or args.model_path is not None):
        parser.error('--data-dir and --model-path require --system a0')
    if args.recover_names and args.system == 'a0':
        parser.error('--recover-names requires --system best or both')
    try:
        run(args.question, args.system, args.data_dir, args.model_path, args.recover_names)
    except FileNotFoundError as error:
        parser.exit(2, f'Missing data or model: {error.filename or error}. '
                       'Run python -m kgqa.acceptance for an offline A0 example; '
                       'see docs/提交验收说明.md for personal-run paths.\n')

