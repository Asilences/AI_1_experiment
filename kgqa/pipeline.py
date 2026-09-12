from __future__ import annotations

import argparse
import logging
import pickle
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from .common import ROOT, REFUSAL, answer_text, read_json, read_jsonl, setup_logging, write_json, write_jsonl
from .linker import EntityLinker, normalize


class PipelineQA:
    def __init__(self, entities, movies, classifier, threshold):
        self.entities, self.classifier, self.threshold = entities, classifier, threshold
        self.linker = EntityLinker(movies, entities)

    def predict(self, question):
        started = time.perf_counter()
        movie_id, link_score, link_status = self.linker.link(question)
        result = {'question': question, 'movie_id': movie_id, 'link_score': link_score, 'link_status': link_status,
                  'relation_id': 'UNKNOWN', 'answer_ids': [], 'answer': REFUSAL, 'evidence': []}
        if movie_id:
            title = self.entities[movie_id]['label']
            masked = normalize(question).replace(normalize(title), ' movie_title ')
            probabilities = self.classifier.predict_proba([masked])[0]
            index = int(np.argmax(probabilities))
            relation = self.classifier.classes_[index]
            result['relation_confidence'] = float(probabilities[index])
            result['relation_id'] = relation
            answer_ids = self.entities[movie_id].get('facts', {}).get(relation, [])
            if relation != 'UNKNOWN' and probabilities[index] >= self.threshold and answer_ids:
                names = [self.entities[qid]['label'] for qid in answer_ids]
                result.update(answer_ids=answer_ids, answer=answer_text(title, relation, names),
                              evidence=[[movie_id, relation, qid] for qid in answer_ids])
        result['latency_ms'] = (time.perf_counter() - started) * 1000
        return result


def mask_question(row, entities):
    title = entities[row['movie_id']]['label']
    return normalize(row['question']).replace(normalize(title), ' movie_title ')


def train(data_dir=ROOT / 'data' / 'processed'):
    qa, entities, splits = read_jsonl(data_dir / 'qa.jsonl'), read_json(data_dir / 'entities.json'), read_json(data_dir / 'splits.json')
    movie_ids = sum(splits.values(), [])
    train_rows = [row for row in qa if row['split'] == 'train']
    features = FeatureUnion([('word', TfidfVectorizer(ngram_range=(1, 2))),
                             ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5)))])
    matrix = features.fit_transform([mask_question(row, entities) for row in train_rows])
    estimator = LogisticRegression(max_iter=300, random_state=42, solver='liblinear').fit(
        matrix, [row['relation_id'] for row in train_rows])

    class Model:
        classes_ = estimator.classes_
        def predict_proba(self, rows):
            return estimator.predict_proba(features.transform(rows))

    best = (0, .5)
    valid = [row for row in qa if row['split'] == 'valid']
    linker = EntityLinker(movie_ids, entities)
    linked = [linker.link(row['question'])[0] for row in valid]
    masked = [mask_question(row, entities) for row in valid]
    probabilities = estimator.predict_proba(features.transform(masked))
    predicted_indexes = np.argmax(probabilities, axis=1)
    predicted_relations = estimator.classes_[predicted_indexes]
    confidences = probabilities[np.arange(len(valid)), predicted_indexes]
    for threshold in np.arange(.30, .86, .05):
        correct = 0
        for row, movie_id, relation, confidence in zip(valid, linked, predicted_relations, confidences):
            answer_ids = []
            if movie_id and relation != 'UNKNOWN' and confidence >= threshold:
                answer_ids = entities[movie_id].get('facts', {}).get(relation, [])
            correct += set(answer_ids) == set(row['answer_ids'])
        score = correct / max(1, len(valid))
        if score > best[0]:
            best = (score, float(threshold))
    output = ROOT / 'models' / 'pipeline'; output.mkdir(parents=True, exist_ok=True)
    with (output / 'model.pkl').open('wb') as stream:
        pickle.dump({'features': features, 'estimator': estimator, 'threshold': best[1]}, stream)
    logging.info('Validation exact set accuracy=%.4f threshold=%.2f', best[0], best[1])


def load_model(path=ROOT / 'models' / 'pipeline' / 'model.pkl'):
    with path.open('rb') as stream:
        saved = pickle.load(stream)
    class Model:
        classes_ = saved['estimator'].classes_
        def predict_proba(self, rows):
            return saved['estimator'].predict_proba(saved['features'].transform(rows))
    return Model(), saved['threshold']


def predict_rows(rows, output_path):
    data = ROOT / 'data' / 'processed'
    entities, splits = read_json(data / 'entities.json'), read_json(data / 'splits.json')
    model, threshold = load_model()
    system = PipelineQA(entities, sum(splits.values(), []), model, threshold)
    predictions = []
    for row in rows:
        prediction = system.predict(row['question']); prediction['question_id'] = row['question_id']; predictions.append(prediction)
    write_jsonl(output_path, predictions)


def predict_split(split='test'):
    qa = [row for row in read_jsonl(ROOT / 'data' / 'processed' / 'qa.jsonl') if row['split'] == split]
    run = ROOT / 'runs' / 'A0'; run.mkdir(parents=True, exist_ok=True)
    predict_rows(qa, run / f'predictions_{split}.jsonl')
    _, threshold = load_model(); write_json(run / 'config.json', {'system': 'pipeline', 'threshold': threshold})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['train', 'predict', 'robust']); parser.add_argument('--split', default='test')
    args = parser.parse_args(); setup_logging(f'pipeline_{args.action}')
    if args.action == 'train': train()
    elif args.action == 'predict': predict_split(args.split)
    else: predict_rows(read_jsonl(ROOT / 'data' / 'processed' / 'robustness.jsonl'), ROOT / 'runs' / 'A0' / 'predictions_robustness.jsonl')

