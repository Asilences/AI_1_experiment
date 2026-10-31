"""Export controlled English paraphrases without changing a running experiment."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil

from .common import ROOT, read_json, read_jsonl, sha256, utc_now, write_json, write_jsonl

# Families are distinct sentence frames. They are authored synthetic questions,
# not independent human questions or proof of unrestricted language understanding.
FAMILIES = {
    'train': {
        'P57': [('director_identity', 'Could you tell me the director of {title}?'), ('helm_credit', 'Who helmed {title}?')],
        'P136': [('genre_explanation', 'Tell me the genre of {title}, please.'), ('genre_request', 'I would like to know the genres of {title}.')],
        'P495': [('production_country', 'Which countries produced {title}?'), ('country_request', 'Tell me the country of origin for {title}, please.')],
        'P364': [('spoken_languages', 'What languages are spoken in {title}?'), ('language_request', 'Please tell me the original language of {title}.')],
        'UNKNOWN': [('cast_request', 'Please list the cast of {title}.'), ('budget_request', 'Tell me the production budget of {title}.')],
    },
    'valid': {
        'P57': [('directing_credit', 'Who receives the directing credit for {title}?'), ('director_lookup', 'Can you look up who directed {title}?')],
        'P136': [('genre_classification', 'How would you classify {title} by genre?'), ('genre_catalog', 'Which genres are recorded for {title}?')],
        'P495': [('country_catalog', 'What production countries are recorded for {title}?'), ('nation_lookup', 'Can you name the nations that made {title}?')],
        'P364': [('language_catalog', 'What original languages are recorded for {title}?'), ('language_lookup', 'Can you identify the language spoken in {title}?')],
        'UNKNOWN': [('box_office_lookup', 'How much did {title} earn at the box office?'), ('composer_lookup', 'Who composed the music for {title}?')],
    },
    'test': {
        'P57': [('camera_credit', 'Who was behind the camera on {title}?'), ('filmmaker_lookup', 'Name the filmmaker responsible for {title}.')],
        'P136': [('genre_membership', 'In terms of genre, where does {title} belong?'), ('genre_summary', 'Give me the genre classification of {title}.')],
        'P495': [('origin_summary', 'Where is {title} from originally?'), ('nation_summary', 'List the nations behind the production of {title}.')],
        'P364': [('language_summary', 'Which languages did the makers originally use in {title}?'), ('language_fact', 'Give me the original-language information for {title}.')],
        'UNKNOWN': [('writer_lookup', 'Who wrote {title}?'), ('release_lookup', 'When was {title} released?')],
    },
}

# Keep v1 byte-reproducible. v2 fixes property semantics before any new scores
# are inspected; these questions still do not constitute a human benchmark.
FAMILIES_V2 = deepcopy(FAMILIES)
FAMILIES_V2['train']['P364'][0] = ('original_audio', 'What were the original audio languages of {title}?')
FAMILIES_V2['valid']['P364'][1] = ('original_language_lookup', 'Can you identify the original language of {title}?')
FAMILIES_V2['test']['P57'] = [
    ('directorial_credit', 'Which person has the directorial credit on {title}?'),
    ('director_name', 'Name the director responsible for directing {title}.')]
FAMILIES_V2['test']['P495'][0] = ('country_origin_summary', 'What is the country of origin of {title}?')
RECIPES = {'v1': FAMILIES, 'v2': FAMILIES_V2}


def build(source_dir, output_dir, recipe='v1'):
    if recipe not in RECIPES:
        raise ValueError(f'Unknown paraphrase recipe: {recipe}')
    families = RECIPES[recipe]
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('Source and output must be separate, non-nested directories')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite dataset: {output}')
    for name in ('manifest.json', 'entities.json', 'splits.json', 'triples.tsv', 'qa.jsonl'):
        if not (source / name).is_file():
            raise FileNotFoundError(f'Missing source dataset file: {name}')
    source_manifest = read_json(source / 'manifest.json')
    if source_manifest.get('dataset_role') == 'synthetic English paraphrase training and family-held-out regression':
        raise ValueError('Select the original frozen QA, not an already augmented dataset')
    entities, splits = read_json(source / 'entities.json'), read_json(source / 'splits.json')
    source_rows = read_jsonl(source / 'qa.jsonl')
    membership = {qid: split for split, ids in splits.items() for qid in ids}
    if len(membership) != sum(len(ids) for ids in splits.values()):
        raise ValueError('Source movie splits overlap')
    if len({row['question_id'] for row in source_rows}) != len(source_rows):
        raise ValueError('Duplicate source question IDs')
    for row in source_rows:
        if row['split'] != membership.get(row['movie_id']):
            raise ValueError('Source QA does not match its movie split')
    pairs = {}
    for row in source_rows:
        key = row['movie_id'], row['relation_id']
        if key in pairs and (set(row['answer_ids']) != set(pairs[key]['answer_ids'])
                             or row['is_answerable'] != pairs[key]['is_answerable']):
            raise ValueError('Inconsistent source answers for the same movie and relation')
        pairs.setdefault(key, row)
    rows = [{**row, 'question_source': 'original_train', 'source_question_id': row['question_id']}
            for row in source_rows if row['split'] == 'train']
    for (movie, relation), row in sorted(pairs.items()):
        title = entities[movie].get('question_title', entities[movie]['label'])
        for family, template in families[row['split']][relation]:
            rows.append({**row, 'question_id': f'freeform_{movie}_{relation}_{family}',
                         'question': template.format(title=title), 'template_family': family,
                         'question_source': 'synthetic_paraphrase', 'source_question_id': row['question_id']})
    output.mkdir(parents=True, exist_ok=True)
    for name in ('entities.json', 'splits.json', 'triples.tsv'):
        shutil.copyfile(source / name, output / name)
    write_jsonl(output / 'qa.jsonl', rows)
    manifest = {**source_manifest, 'created_at': utc_now(),
                'name': '5000电影英文自然问法受控改写' if len(membership) == 5000 else '英文自然问法受控改写',
                'dataset_role': 'synthetic English paraphrase training and family-held-out regression',
                'qa': len(rows), 'qa_by_split': dict(Counter(row['split'] for row in rows)),
                'qa_by_relation': dict(Counter(row['relation_id'] for row in rows)),
                'source_hashes': {name: sha256(source / name) for name in ('entities.json', 'splits.json', 'qa.jsonl', 'triples.tsv')},
                'graph_unchanged': True, 'movie_splits_unchanged': True,
                'original_qa_included_in': ['train'], 'question_language': 'en',
                'recipe_version': recipe,
                'recipe_revision_reason': ('Disambiguate original language, director and country of origin before scoring.'
                                           if recipe == 'v2' else 'Original authored paraphrase recipe.'),
                'unsupported_scope': 'UNKNOWN labels apply to the four-relation neural task, not the V2 catalog tools.',
                'families': {split: {relation: [{'id': family, 'template': template} for family, template in templates]
                                     for relation, templates in relations.items()} for split, relations in families.items()},
                'scope_note': 'Authored synthetic paraphrases; family-held-out results are not independent human free-question evaluation.'}
    write_json(output / 'manifest.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'data/processed/movies_5000_20261029')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--recipe', choices=sorted(RECIPES), default='v1', help='Use v2 for the clarified property semantics; v1 is retained for reproduction.')
    args = parser.parse_args(argv)
    try:
        manifest = build(args.source_dir, args.output_dir, args.recipe)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({key: manifest[key] for key in ('name', 'qa', 'qa_by_split', 'graph_unchanged')}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
