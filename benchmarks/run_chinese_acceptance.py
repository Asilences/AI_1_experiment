"""Controlled Chinese interface acceptance; separate from the frozen English experiment.

No training, network, or torch import is needed for A0. HTTP neural sampling is
explicitly optional and uses the real local /api/answer endpoint.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kgqa.common import read_json, read_jsonl, sha256, utc_now, write_json, write_jsonl

RELATIONS = ('P57', 'P136', 'P495', 'P364')
CJK = re.compile(r'[\u3400-\u9fff\uf900-\ufaff]')
TEMPLATES = {
    'P57': ('《{title}》的导演是谁？', '谁执导了《{title}》？', '请问《{title}》是哪位导演拍的？'),
    'P136': ('《{title}》是什么类型？', '《{title}》属于哪种流派？', '请告诉我《{title}》的电影类别。'),
    'P495': ('《{title}》的制片国家是哪里？', '《{title}》来自哪个国家？', '请问《{title}》的出品国是哪里？'),
    'P364': ('《{title}》的原始语言是什么？', '《{title}》使用什么语种？', '请问《{title}》的原声是什么语言？'),
}
SCOPE = ('Deterministic template-controlled Chinese interface acceptance on movie-held-out '
         'valid/test IDs in a visible frozen knowledge graph; not independently collected '
         'natural Chinese QA, multilingual retraining, or open-domain generalization.')
FIXTURE_PROTOCOL = ('v2 post-diagnostic revision: append the stored release year outside every positive '
                    'title when available, uniformly across all selected movies and variants. '
                    'v1 contained eight correctly refused ambiguous unqualified titles; no adapter change '
                    'or correctness-based movie selection is made to force an answer.')


def build_fixture(entities, splits, movies_per_split=10):
    """Select by ID/metadata only; never select cases by adapter or model correctness."""
    if movies_per_split < 1:
        raise ValueError('movies_per_split must be positive')
    movie_ids = sum((list(splits.get(s, [])) for s in ('train', 'valid', 'test')), [])
    if len(movie_ids) != len(set(movie_ids)):
        raise ValueError('Movie splits overlap or contain duplicates')
    selected, questions, specifications = {}, [], {}
    for split in ('valid', 'test'):
        eligible = [qid for qid in sorted(splits.get(split, []))
                    if CJK.search(entities[qid].get('label_zh') or '')
                    and all(entities[qid].get('facts', {}).get(r) for r in RELATIONS)]
        if len(eligible) < movies_per_split:
            raise ValueError(f'{split} has only {len(eligible)} complete Chinese-labeled movies')
        selected[split] = eligible[:movies_per_split]
        for movie in selected[split]:
            entity = entities[movie]
            for relation in RELATIONS:
                for i, variant in enumerate(('canonical', 'paraphrase', 'mixed_name')):
                    # Mixed input uses the actual question title, retaining its original script.
                    # No title uniqueness or model-correctness filter is applied.
                    title = (entity.get('question_title') or entity['label']) if i == 2 else entity['label_zh']
                    question_id = f'zh-controlled-v2-{split}-{movie}-{relation}-{variant}'
                    question = TEMPLATES[relation][i].format(title=title)
                    if entity.get('release_year'):
                        question = question.replace('》', f'》（{entity["release_year"]}年版）', 1)
                    questions.append({'question_id': question_id, 'split': split,
                                      'variant': variant, 'question': question})
                    specifications[question_id] = {'movie_id': movie, 'relation_id': relation,
                                                   'expected_refusal': False}
        first, second = selected[split][0], selected[split][-1]
        title, other = entities[first]['label_zh'], entities[second]['label_zh']
        negatives = [
            ('actor', f'《{title}》的主演是谁？'),
            ('release_time', f'《{title}》是什么时候上映的？'),
            ('multiple_relations', f'《{title}》的导演和类型是什么？'),
            ('two_movies', f'{first}和{second if second != first else "Q999999999999"}的导演是谁？'),
            ('unknown_title', '《中文受控验收不存在的电影_20261030》的导演是谁？'),
            ('negative_director', f'谁不是《{title}》的导演？'),
            ('runtime', f'《{title}》的片长是多少？'),
            ('plot', f'请介绍《{title}》的剧情。'),
            ('budget', f'《{title}》的预算是多少？'),
            ('two_quoted_titles', f'《{title}》和《{other}》的原始语言是什么？'
             if second != first else f'《{title}》的票房是多少？'),
        ]
        for category, question in negatives:
            question_id = f'zh-controlled-v2-{split}-negative-{category}'
            questions.append({'question_id': question_id, 'split': split, 'variant': 'negative',
                              'category': category, 'question': question})
            specifications[question_id] = {'movie_id': None, 'relation_id': None, 'expected_refusal': True}
    return questions, specifications, selected


def build_gold(specifications, entities):
    """Called only after the question fixture has been saved and hashed."""
    return [{'question_id': key, **case,
             'answer_ids': [] if case['expected_refusal'] else
             list(entities[case['movie_id']]['facts'][case['relation_id']])}
            for key, case in specifications.items()]


def predict_questions(questions, predictor):
    # The only argument reaching the predictor is the original question string.
    return [{'question_id': row['question_id'], **predictor(row['question'])} for row in questions]


def score_predictions(questions, gold, predictions):
    expected = {row['question_id']: row for row in questions}
    truth = {row['question_id']: row for row in gold}
    actual = {row['question_id']: row for row in predictions}
    if (len(expected) != len(questions) or len(truth) != len(gold) or len(actual) != len(predictions)
            or expected.keys() != truth.keys() or expected.keys() != actual.keys()):
        raise ValueError('Question, gold and prediction IDs must be unique and match exactly')
    scored = []
    for key, question in expected.items():
        target, pred = truth[key], actual[key]
        refused = pred.get('refused') is True and not pred.get('answer_ids')
        exact = set(pred.get('answer_ids', [])) == set(target['answer_ids'])
        negative = target['expected_refusal']
        scored.append({**question, 'expected_movie_id': target['movie_id'],
                       'expected_relation_id': target['relation_id'], 'expected_answer_ids': target['answer_ids'],
                       'expected_refusal': negative, 'prediction': pred,
                       'entity_exact': exact if not negative else refused,
                       'input_movie_correct': None if negative else pred.get('movie_id') == target['movie_id'],
                       'input_intent_correct': None if negative else pred.get('input_relation_id') == target['relation_id'],
                       'predicted_relation_correct': None if negative else pred.get('relation_id') == target['relation_id'],
                       'correct_refusal': refused if negative else None,
                       'chinese_answer': bool(CJK.search(pred.get('answer', '')))})

    def metrics(rows):
        positive = [r for r in rows if not r['expected_refusal']]
        negative = [r for r in rows if r['expected_refusal']]
        def fraction(group, key):
            return sum(bool(r[key]) for r in group) / len(group) if group else None
        return {'count': len(rows), 'positive_count': len(positive), 'negative_count': len(negative),
                'positive_entity_exact': fraction(positive, 'entity_exact'),
                'input_entity_link_accuracy': fraction(positive, 'input_movie_correct'),
                'input_intent_accuracy': fraction(positive, 'input_intent_correct'),
                'predicted_relation_accuracy': fraction(positive, 'predicted_relation_correct'),
                'negative_refusal_accuracy': fraction(negative, 'correct_refusal'),
                'chinese_output_coverage': fraction(rows, 'chinese_answer')}
    summary = metrics(scored)
    summary['by_split'] = {s: metrics([r for r in scored if r['split'] == s]) for s in sorted({r['split'] for r in scored})}
    summary['by_variant'] = {s: metrics([r for r in scored if r['variant'] == s]) for s in sorted({r['variant'] for r in scored})}
    summary['by_relation'] = {r: metrics([row for row in scored if row['expected_relation_id'] == r]) for r in RELATIONS}
    summary['status'] = 'PASS' if all(r['entity_exact'] and r['chinese_answer'] and
        (r['expected_refusal'] or (r['input_movie_correct'] and r['input_intent_correct']
                                  and r['predicted_relation_correct'])) for r in scored) else 'FAIL'
    return summary, scored


def _json_http(url, body=None, timeout=180):
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode('utf-8')
    request = Request(url, data=payload, headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def select_http_questions(questions, count=16):
    """Choose inputs independently of gold and service results, before any requests."""
    # One canonical question per relation/split before expanding to other cases.
    canonical = [r for r in questions if r['variant'] == 'canonical']
    seen, first, remaining = set(), [], []
    for row in canonical:
        key = (row['split'], row['question_id'].split('-')[-2])
        if key in seen:
            remaining.append(row)
        else:
            first.append(row)
            seen.add(key)
    negatives = [r for r in questions if r['variant'] == 'negative']
    negatives.sort(key=lambda r: (r.get('category', ''), r['split']))
    return (first + negatives[:8] + remaining + negatives[8:])[:count]


def http_neural_sample(base_url, questions, gold, count=16):
    """Real endpoint outputs are preserved; unavailable/failed generation is not a pass."""
    sample = select_http_questions(questions, count)
    base_url = base_url.rstrip('/')
    state = _json_http(base_url + '/api/status')
    if 'zh' not in state.get('question_languages', []):
        raise ValueError('HTTP server does not advertise the Chinese interface; neural entity metrics are unavailable for this server version')
    raw, predictions = [], []
    for row in sample:
        response = _json_http(base_url + '/api/answer', {'question': row['question'], 'system': 'best'})
        raw.append({'question_id': row['question_id'], 'response': response})
        result = next((p for p in response.get('results', []) if p.get('system') == 'best'), {})
        if not result.get('ok'):
            raise ValueError(f"Neural HTTP inference failed for {row['question_id']}: {result.get('error')}")
        if 'answer_ids' not in result or result.get('language') != 'zh':
            raise ValueError(f"HTTP response lacks observable Chinese prediction fields for {row['question_id']}; cannot score missing IDs as an empty prediction")
        predictions.append({'question_id': row['question_id'], **result,
                            'movie_id': (result.get('movie') or {}).get('id')})
    selected_ids = {r['question_id'] for r in sample}
    sample_gold = [r for r in gold if r['question_id'] in selected_ids]
    # Neural /api/answer exposes generated IDs and movie grounding, but not input
    # relation IDs. Report only directly observable fields rather than inventing them.
    truth = {r['question_id']: r for r in sample_gold}
    positive = [p for p in predictions if not truth[p['question_id']]['expected_refusal']]
    negative = [p for p in predictions if truth[p['question_id']]['expected_refusal']]
    exact = sum(set(p.get('answer_ids', [])) == set(truth[p['question_id']]['answer_ids']) for p in positive)
    refusals = sum(p.get('refused') is True and not p.get('answer_ids') for p in negative)
    f1 = []
    for pred in positive:
        predicted, target = set(pred['answer_ids']), set(truth[pred['question_id']]['answer_ids'])
        f1.append(2 * len(predicted & target) / (len(predicted) + len(target)) if predicted or target else 1.0)
    return {'status': 'COMPLETED', 'count': len(sample), 'entity_exact_or_correct_refusal': (exact + refusals) / len(sample),
            'positive_count': len(positive), 'negative_count': len(negative),
            'positive_entity_exact': exact / len(positive) if positive else None,
            'positive_entity_macro_f1': sum(f1) / len(f1) if f1 else None,
            'negative_refusal_accuracy': refusals / len(negative) if negative else None,
            'chinese_output_coverage': sum(bool(CJK.search(p.get('answer', ''))) for p in predictions) / len(sample),
            'scope': 'Small real neural HTTP smoke sample; no statistical generalization claim.',
            'server_status': state, 'raw_responses': raw, 'predictions': predictions}


def run(data_dir, model_path, output_dir, movies_per_split=10, http_url=None,
        http_sample_size=16, http_model_dir=None, prior_diagnostic=None):
    data_dir, model_path, output = Path(data_dir).resolve(), Path(model_path).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing existing output directory: {output}')
    if output.is_relative_to(data_dir) or output.is_relative_to(model_path.parent):
        raise ValueError('Use an output directory outside the source data and model directories')
    sources = {name: ROOT / name for name in ('kgqa/chinese.py', 'kgqa/pipeline.py', 'kgqa/linker.py',
                                             'kgqa/common.py', 'kgqa/demo.py', 'kgqa/web.py',
                                             'kgqa/metrics.py', 'kgqa/name_recovery.py',
                                             'kgqa/neural_preflight.py', 'benchmarks/run_chinese_acceptance.py')}
    catalog = ROOT / 'kgqa/locales/zh_entities.json'
    if catalog.exists():
        sources['kgqa/locales/zh_entities.json'] = catalog
    assets = {f'data/{name}': data_dir / name for name in ('entities.json', 'splits.json', 'qa.jsonl', 'manifest.json')}
    assets['model/model.pkl'] = model_path
    if (model_path.parent / 'config.json').exists():
        assets['model/config.json'] = model_path.parent / 'config.json'
    before = {name: sha256(path) for name, path in assets.items()}
    source_hashes = {name: sha256(path) for name, path in sources.items()}
    config_path = model_path.parent / 'config.json'
    if config_path.exists():
        config = read_json(config_path)
        if config.get('input_hashes') != {name: before[f'data/{name}'] for name in ('entities.json', 'splits.json', 'qa.jsonl')}:
            raise ValueError('Model config and input dataset hashes do not match')
        if config.get('model_sha256') != before['model/model.pkl']:
            raise ValueError('Model config and model SHA-256 do not match')
    entities, splits = read_json(data_dir / 'entities.json'), read_json(data_dir / 'splits.json')
    questions, specifications, selected = build_fixture(entities, splits, movies_per_split)
    output.mkdir(parents=True)
    fixture_path = output / 'questions.jsonl'
    write_jsonl(fixture_path, questions)
    fixture_hash = sha256(fixture_path)
    write_json(output / 'fixture_freeze.json', {'created_at': utc_now(), 'questions_sha256': fixture_hash,
               'selection': 'Lexicographic movie ID; Chinese title and four nonempty facts; no prediction-based selection.',
               'selected_movies': selected, 'templates': TEMPLATES, 'scope': SCOPE,
               'fixture_protocol': FIXTURE_PROTOCOL})
    gold = build_gold(specifications, entities)
    write_jsonl(output / 'gold.jsonl', gold)
    from kgqa.pipeline import PipelineQA, load_model
    classifier, threshold = load_model(model_path)
    system = PipelineQA(entities, sum(splits.values(), []), classifier, threshold)
    # Reload only the frozen input fixture. Gold/specifications never enter predict.
    predictions = predict_questions(read_jsonl(fixture_path), system.predict)
    write_jsonl(output / 'predictions_a0.jsonl', predictions)
    metrics, scored = score_predictions(questions, gold, predictions)
    write_jsonl(output / 'scored_a0.jsonl', scored)
    all_movies = sum(splits.values(), [])
    answer_ids = [qid for r in gold for qid in r['answer_ids']]
    coverage = {'movie_count': len(all_movies),
                'movies_with_chinese_label': sum(bool(CJK.search(entities[q].get('label_zh') or '')) for q in all_movies),
                'selected_movies': sum(map(len, selected.values())),
                'answer_entity_occurrences': len(answer_ids),
                'answer_entity_occurrences_with_chinese_label': sum(bool(CJK.search(entities[q].get('label_zh') or '')) for q in answer_ids),
                'answer_entity_occurrences_with_chinese_display_label': sum(bool(CJK.search(system.chinese.format_entity(q))) for q in answer_ids),
                'fallback': 'Original entity label, then QID; no fabricated Chinese translation.'}
    summary = {'created_at': utc_now(), 'benchmark': 'chinese_controlled_acceptance_v2',
               'fixture_protocol': FIXTURE_PROTOCOL,
               'scope': SCOPE, 'independent_natural_language_collection': False,
               'training_runs': 0, 'fixture_sha256': fixture_hash, 'gold_sha256': sha256(output / 'gold.jsonl'),
               'source_sha256': source_hashes, 'asset_sha256_before': before, 'a0': metrics,
               'chinese_metadata_coverage': coverage, 'status': metrics['status'],
               'model_dataset_pairing_config_checked': config_path.exists(),
               'environment': {'python': sys.version, 'executable': 'python',
                               'packages': {name: version(name) for name in ('numpy', 'scipy', 'scikit-learn')}},
               'limits': ['Shared hand-written templates; only 20 selected movies at default size.',
                          'Visible graph facts are the answer source; movie holdout is not graph holdout.',
                          'Incomplete movies and movies without Chinese labels are excluded by design.',
                          'mixed_name preserves stored question_title, which may itself be Chinese.',
                          'A Chinese sentence can contain original names when Chinese metadata is missing.',
                          'Does not replace, rescore, or alter the frozen English neural experiment.']}
    if prior_diagnostic:
        previous_path = Path(prior_diagnostic).resolve()
        previous = read_json(previous_path)
        summary['prior_diagnostic'] = {'summary_sha256': sha256(previous_path),
            'fixture_sha256': previous['fixture_sha256'], 'source_sha256': previous['source_sha256'],
            'a0': previous['a0'],
            'explanation': 'Eight unqualified Captain America title cases were correctly refused as ambiguous; fixture specification issue, not a product error.',
            'environment_note': 'v1 ran with qa-env (Anaconda base, include-system-site-packages=true), scikit-learn 1.5.1 versus pickle training 1.6.1; incompatible-version warnings occurred. v2 records and uses the clean runtime.',
            'interpretation': 'Post-diagnostic protocol revision, not independent confirmation on newly collected natural questions.'}
    if http_url:
        sample_path = output / 'neural_http_questions.jsonl'
        write_jsonl(sample_path, select_http_questions(questions, http_sample_size))
        summary['neural_http_fixture_sha256'] = sha256(sample_path)
        try:
            neural = http_neural_sample(http_url, questions, gold, http_sample_size)
            if http_model_dir:
                directory = Path(http_model_dir).resolve()
                neural['operator_supplied_model_sha256'] = {p.relative_to(directory).as_posix(): sha256(p)
                                                           for p in sorted(directory.rglob('*')) if p.is_file()}
                neural['model_identity_note'] = 'Operator-supplied files; HTTP does not attest server checkpoint identity.'
            else:
                neural['model_identity_note'] = 'Server run ID only; no checkpoint hash supplied.'
            write_json(output / 'neural_http_sample.json', neural)
            summary['neural_http'] = {k: v for k, v in neural.items() if k not in ('raw_responses', 'predictions')}
        except (OSError, ValueError, KeyError) as error:
            summary['neural_http'] = {'status': 'FAIL', 'error': str(error)}
            summary['status'] = 'FAIL'
    after = {name: sha256(path) for name, path in assets.items()}
    summary['asset_sha256_after'] = after
    summary['input_assets_unchanged'] = before == after
    summary['fixture_unchanged'] = sha256(fixture_path) == fixture_hash
    summary['source_unchanged'] = source_hashes == {name: sha256(path) for name, path in sources.items()}
    if not all(summary[k] for k in ('input_assets_unchanged', 'fixture_unchanged', 'source_unchanged')):
        summary['status'] = 'FAIL'
    write_json(output / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--movies-per-split', type=int, default=10)
    parser.add_argument('--http-url', help='Optional actual local neural web service base URL')
    parser.add_argument('--http-sample-size', type=int, default=16)
    parser.add_argument('--http-model-dir', type=Path, help='Optional operator-supplied neural checkpoint files to hash')
    parser.add_argument('--prior-diagnostic', type=Path, help='Preserved v1 summary.json to record diagnostic provenance')
    args = parser.parse_args(argv)
    if args.http_sample_size < 1:
        parser.error('--http-sample-size must be positive')
    try:
        summary = run(**vars(args))
    except (OSError, ValueError, KeyError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        return 1
    print(json.dumps({'status': summary['status'], 'a0': summary['a0'],
                      'output_dir': str(args.output_dir)}, ensure_ascii=False, indent=2))
    return 0 if summary['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
