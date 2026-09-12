from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .common import REFUSAL, ROOT, read_json
from .linker import EntityLinker
from .neural import Collator, GraphData, GraphPrefixModel, QADataset, predictions
from .pipeline import PipelineQA, load_model


def entity_label(entities, qid):
    if not qid:
        return '未识别'
    return f"{entities.get(qid, {}).get('label', qid)} ({qid})"


class Demo:
    def __init__(self, load_neural=True):
        data = ROOT / 'data' / 'processed'
        self.entities = read_json(data / 'entities.json')
        self.splits = read_json(data / 'splits.json')
        self.movie_ids = sum(self.splits.values(), [])
        model, threshold = load_model()
        self.pipeline = PipelineQA(self.entities, self.movie_ids, model, threshold)
        self.linker = EntityLinker(self.movie_ids, self.entities)
        self.neural_run = read_json(ROOT / 'output' / 'selected_model.json')['selected_run']
        self.neural = None
        if load_neural:
            self._load_neural()

    def _load_neural(self):
        run = ROOT / 'runs' / self.neural_run
        config = read_json(run / 'config.json')
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
        movie_id, link_score, link_status = self.linker.link(question)
        if not movie_id:
            return {'system': f'{self.neural_run} 神经生成系统', 'movie': '未识别', 'link_status': link_status,
                    'evidence': [], 'answer': REFUSAL, 'latency_ms': 0}
        row = {'question_id': 'interactive', 'question': question, 'movie_id': movie_id,
               'reference_answers': [REFUSAL], 'answer_ids': [], 'relation_id': 'UNKNOWN', 'is_answerable': False}
        loader = DataLoader(QADataset([row], self.graph, link_questions=True), batch_size=1,
                            collate_fn=Collator(self.tokenizer, self.graph.use_embeddings))
        result = predictions(self.neural, loader, self.tokenizer, self.entities, self.device)[0]
        evidence = [item[0] for item in self.graph.facts(movie_id)]
        return {'system': f'{self.neural_run} 神经生成系统', 'movie': entity_label(self.entities, movie_id),
                'link_status': f'{link_status}, score={link_score:.3f}', 'evidence': evidence,
                'answer': result['answer'], 'latency_ms': result['latency_ms']}


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
    print(f"生成延迟：{result['latency_ms']:.2f} ms")


def run(question=None, system='both'):
    demo = Demo(load_neural=system in ('both', 'best'))
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
    args = parser.parse_args(); run(args.question, args.system)

