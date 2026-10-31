"""合成评分入口测试，分数不写入真实人工评价目录。"""
import csv

import pytest

from kgqa.human_eval import VISIBLE_FIELDS
from kgqa.human_review_web import ReviewStore


@pytest.fixture
def store(tmp_path):
    source = tmp_path / 'blank.csv'
    with source.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=VISIBLE_FIELDS)
        writer.writeheader()
        for pos in (1, 2):
            writer.writerow({'question_id': 'Q1_P57_1', 'answer_position': pos,
                             'question': 'Who directed Film?', 'reference': 'A director.', 'answer': 'A director.'})
    return ReviewStore(source, tmp_path / 'reviews')


def test_scores_persist_resume_and_preserve_blank_answers(store):
    values = {'question_id': 'Q1_P57_1', 'answer_position': 1, 'accuracy_0_2': 0,
              'completeness_0_2': 1, 'fluency_1_5': 4, 'unsupported_claim_0_1': 0, 'notes': 'Real zero in fixture'}
    state = store.save('fixture', values)
    assert state == store.load('fixture')
    rows = list(csv.DictReader(store.csv_text(state).splitlines()))
    assert rows[0]['accuracy_0_2'] == '0' and rows[1]['accuracy_0_2'] == ''
    assert 'system' not in rows[0]


@pytest.mark.parametrize('reviewer', ['../escape', '', 'x/y', 'a' * 41])
def test_reviewer_cannot_escape_output_directory(store, reviewer):
    with pytest.raises(ValueError):
        store.load(reviewer)


def test_blank_or_boolean_score_cannot_be_written_as_real_rating(store):
    for score in ('', None, False, 3):
        with pytest.raises(ValueError):
            store.save('fixture', {'question_id': 'Q1_P57_1', 'answer_position': 1, 'accuracy_0_2': score})
    assert store.load('fixture') == {}
