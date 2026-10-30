from types import SimpleNamespace
import sys

from kgqa.chinese import ChineseAdapter
from kgqa.common import REFUSAL
from kgqa.pipeline import PipelineQA


def entities():
    return {'M': {'label': 'Example Film', 'label_zh': '示例电影', 'facts': {'P57': ['D']}},
            'D': {'label': 'Jane Director', 'label_zh': '简导演'},
            'OTHER': {'label': 'Another Director', 'label_zh': '另一位导演'}}


class Classifier:
    classes_ = ['P57']

    def __init__(self):
        self.inputs = []

    def predict_proba(self, rows):
        self.inputs.extend(rows)
        return [[1.0] for _ in rows]


def test_chinese_a0_uses_existing_classifier_and_returns_actual_predicted_ids():
    classifier = Classifier()
    system = PipelineQA(entities(), ['M'], classifier, .3)
    result = system.predict('《示例电影》的导演是谁？')
    assert result['answer_ids'] == ['D'] and result['movie_id'] == 'M'
    assert result['answer'] == '《示例电影》的导演是简导演。'
    assert result['original_answer'] == 'Example Film was directed by Jane Director.'
    assert result['model_question'] == 'Who directed Example Film?'
    assert classifier.inputs == ['who directed movie_title']
    assert result['evidence'] == [['M', 'P57', 'D']]


def test_chinese_refusal_does_not_classify_or_silently_answer_partial_request():
    classifier = Classifier()
    system = PipelineQA(entities(), ['M'], classifier, .3)
    for question in ('《未知电影》的导演是谁？', '《示例电影》的导演和主演是谁？'):
        result = system.predict(question)
        assert result['refused'] and not result['answer_ids']
        assert result['original_answer'] == REFUSAL and result['language'] == 'zh'
    assert classifier.inputs == []


def test_chinese_missing_facts_returns_refusal_instead_of_success_note():
    data = entities()
    data['M']['facts'] = {}
    result = PipelineQA(data, ['M'], Classifier(), .3).predict('《示例电影》的导演是谁？')
    assert result['refused'] and not result['answer_ids']
    assert '没有足够' in result['answer']


def test_english_a0_result_and_classifier_input_remain_unchanged():
    classifier = Classifier()
    result = PipelineQA(entities(), ['M'], classifier, .3).predict('Who directed Example Film?')
    assert result['answer'] == 'Example Film was directed by Jane Director.'
    assert 'language' not in result and classifier.inputs == ['who directed movie_title']


def neural_stub(monkeypatch, answer):
    from kgqa.demo import Demo
    data = entities()
    system = Demo.__new__(Demo)
    system.entities = data
    system.chinese = ChineseAdapter(data, ['M'])
    system.neural, system.neural_run, system.device = object(), 'B0', object()
    system.recover_names = False
    system.tokenizer = object()
    system.graph = SimpleNamespace(use_embeddings=False, facts=lambda qid: [('retrieved candidate', None)])
    system.linker = SimpleNamespace(link=lambda question: (_ for _ in ()).throw(AssertionError('Chinese grounding must not relink an ambiguous English title')))
    rows = []
    monkeypatch.setitem(sys.modules, 'torch.utils.data', SimpleNamespace(DataLoader=lambda dataset, **kwargs: dataset))
    monkeypatch.setitem(sys.modules, 'kgqa.neural', SimpleNamespace(
        QADataset=lambda items, graph: rows.extend(items) or items,
        Collator=lambda *args: None,
        predictions=lambda *args: [{'answer': answer, 'latency_ms': 12.5}]))
    return system, rows


def test_chinese_neural_reuses_model_and_preserves_a_wrong_prediction(monkeypatch):
    answer = 'Example Film was directed by Another Director.'
    system, rows = neural_stub(monkeypatch, answer)
    result = system.answer_neural('《示例电影》的导演是谁？')
    assert result['answer'] == '《示例电影》的导演是另一位导演。'
    assert result['original_answer'] == answer
    assert rows[0]['question'] == 'Who directed Example Film?'
    assert rows[0]['movie_id'] == 'M' and rows[0]['answer_ids'] == []
    assert result['latency_ms'] == 12.5 and result['evidence'] == ['retrieved candidate']


def test_chinese_unsupported_query_does_not_load_neural_model(monkeypatch):
    system, rows = neural_stub(monkeypatch, 'must not generate')
    system.neural = None
    system._load_neural = lambda: (_ for _ in ()).throw(AssertionError('No model load on unsupported question'))
    result = system.answer_neural('《示例电影》的主演是谁？')
    assert result['refused'] and result['original_answer'] == REFUSAL
    assert rows == [] and result['latency_ms'] == 0


def test_chinese_web_uses_adapter_movie_and_localized_refusal(tmp_path, monkeypatch):
    from kgqa.common import write_json
    from kgqa.web import QAService
    data = entities()
    write_json(tmp_path / 'entities.json', data)
    write_json(tmp_path / 'splits.json', {'train': ['M'], 'valid': [], 'test': []})
    model = tmp_path / 'model.pkl'
    model.write_bytes(b'fixture')
    service = QAService(tmp_path, model)
    pipeline = PipelineQA(data, ['M'], Classifier(), .3)
    demo = SimpleNamespace(entities=data, chinese=pipeline.chinese, linker=pipeline.linker,
                           neural=None, answer_a0=pipeline.predict)
    monkeypatch.setattr(service, '_load_demo', lambda: demo)
    positive = service.answer('《示例电影》的导演是谁？', 'a0')['results'][0]
    assert positive['ok'] and not positive['refused']
    assert positive['movie'] == {'id': 'M', 'label': '示例电影'}
    assert positive['answer_ids'] == ['D'] and positive['language'] == 'zh'
    refusal = service.answer('《示例电影》的主演是谁？', 'a0')['results'][0]
    assert refusal['ok'] and refusal['refused'] and refusal['movie'] is None
    status = service.status()
    assert len(status['examples_zh']) == 1
    assert status['chinese_coverage']['with_chinese_title'] == 1
