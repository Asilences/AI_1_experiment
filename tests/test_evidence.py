from kgqa import evidence
from kgqa.common import read_json, write_json


def test_evidence_exports_real_records_without_fake_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, 'ROOT', tmp_path)
    write_json(tmp_path / 'output' / 'environment.json', {'python_version': 'test'})
    result = evidence.build()
    assert result['record_type'] == 'saved_log_summary_not_screenshot'
    assert result['records']['environment']['data'] == {'python_version': 'test'}
    assert result['missing']
    output = tmp_path / 'output' / 'evidence'
    assert not list(output.glob('*.png'))
    assert 'NOT a terminal screenshot' in (output / 'run_summary.txt').read_text()
    assert read_json(output / 'run_summary.json') == result
