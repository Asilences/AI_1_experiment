"""对保存的P0输出重放受控回答保护；不调用模型、不重新执行计划或测延迟。"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kgqa.common import read_json, read_jsonl, sha256, write_json, write_jsonl
from kgqa.explorer import Explorer
from kgqa.semantic_qa import controlled_generation_matches, render_tool


def replay(source, catalog, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new empty output directory')
    explorer = Explorer.__new__(Explorer)
    explorer.entities = read_json(catalog / 'entities.json')
    explorer.movies = {m['id']: m for m in read_json(catalog / 'movies.json')}
    outputs, counts = [], Counter()
    for old in read_jsonl(source):
        row = deepcopy(old)
        response = row['response']
        counts['count'] += 1
        if response.get('execution') and response.get('plan'):
            draft = render_tool(explorer, response['execution'], response['plan'])
            raw = response.get('raw_answer')
            generated = {'answer': raw, 'citations': response.get('guard', {}).get('citations')}
            response['answer'] = draft
            if raw is not None:
                counts['raw_generation_present'] += 1
                accepted = controlled_generation_matches(generated, draft, response.get('evidence', []))
                response['guard'] = {'method': 'controlled_evidence_text_v2',
                                     'action': 'accepted' if accepted else 'tool_template',
                                     'reason': 'controlled_text_match' if accepted else 'unverified_free_text_or_citations'}
                response['fallback_used'] = not accepted
                counts['newly_blocked_old_accepted'] += int(old['response']['guard']['action'] == 'accepted' and not accepted)
        counts['fallback_used'] += int(response.get('fallback_used', False))
        row['controller_replay'] = 'stored model output; no new model calls, plans or graph queries'
        row['old_answer'] = old['response']['answer']
        row['fallback_used'] = response.get('fallback_used', False)
        outputs.append(row)
    write_jsonl(output / 'predictions.jsonl', outputs)
    result = {'evaluation_type': 'offline_controller_replay', 'counts': dict(counts),
              'source_sha256': sha256(source), 'model_rerun': False,
              'scope': 'Original plan, tool and status metrics remain unchanged. Latency is not remeasured.'}
    write_json(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'verification/2026-10-31/semantic_p0/dev_v1/P0_predictions.jsonl')
    parser.add_argument('--catalog-dir', type=Path, default=ROOT / 'data/processed/movies_v2_20261030')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(replay(args.source, args.catalog_dir, args.output_dir)['counts'])


if __name__ == '__main__':
    main()
