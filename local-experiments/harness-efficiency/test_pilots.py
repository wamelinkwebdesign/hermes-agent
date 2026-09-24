"""Offline harness checks. No model calls, credentials or installed-skill edits."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location('efficiency_pilots', Path(__file__).with_name('run_pilots.py'))
assert spec and spec.loader
pilots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilots)


def test_frozen_prepare_refuses_overwrite(tmp_path):
    (tmp_path / 'frozen.json').write_text('{}')
    with pytest.raises(FileExistsError):
        pilots.prepare(SimpleNamespace(output=str(tmp_path), skill='must-not-read'))
    assert (tmp_path / 'frozen.json').read_text() == '{}'


def test_call_refuses_previous_attempt(tmp_path):
    target = tmp_path / 'runs' / 'existing'
    target.mkdir(parents=True)
    (target / 'result.json').write_text('{"exit_code": 1}')
    with patch.object(pilots.subprocess, 'run') as run:
        with pytest.raises(FileExistsError):
            pilots.call(tmp_path, 'existing', 'new prompt')
        run.assert_not_called()
    assert json.loads((target / 'result.json').read_text())['exit_code'] == 1


def test_json_format_failure_is_not_repaired(tmp_path):
    path = tmp_path / 'answer.txt'
    path.write_text('The data is approximately {invalid}.')
    with pytest.raises(ValueError):
        pilots.json_answer(path)
    path.write_text('```json\n{"ok": true}\n```')
    assert pilots.json_answer(path) == {'ok': True}


def test_evaluation_preserves_failed_format_and_isolation(tmp_path):
    def write(relative, value, raw=False):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value if raw else json.dumps(value))

    write('skill/cases.json', pilots.CASES)
    answers = [{'id': c['id'], 'action': c['action'], 'needs_new_process': c['restart']} for c in pilots.CASES]
    routes = [{'id': c['id'], 'references': [f"references/{c['required_module']}.md"] if c['required_module'] else []} for c in pilots.CASES]
    write('runs/skill-route/answer.txt', routes)
    for variant in ('baseline', 'candidate'):
        write(f'runs/skill-{variant}/answer.txt', answers)
    write('runs/resume-baseline/answer.txt', 'Status only: not JSON.', raw=True)
    write('runs/resume-candidate/answer.txt', {
        'invoice_id': 'INV--0042', 'preview_path': '/work/sample/out/preview.csv',
        'tests_passed': 7, 'tests_failed': 1, 'deploy_authorized': False,
        'dirty_file': 'src/theme.ts', 'may_resume': False,
        'skill_to_reload': 'test-driven-development',
    })
    labels = ['summary-baseline', 'summary-candidate', 'skill-baseline', 'skill-route',
              'skill-candidate', 'resume-baseline', 'resume-candidate']
    for name in labels:
        write(f'runs/{name}/result.json', {'label': name, 'exit_code': 0, 'item_types': ['error', 'agent_message']})
    with patch.object(pilots.subprocess, 'run') as run:
        pilots.evaluate(SimpleNamespace(output=str(tmp_path)))
        run.assert_not_called()
    result = json.loads((tmp_path / 'results.json').read_text())
    assert len(result['runs']) == 7
    assert result['checks']['resume-baseline'] == {'valid_json_response': False}
    assert all(result['checks']['resume-candidate'].values())
    assert result['all_checks_passed'] is False
    assert result['isolation_clean'] is False
    assert result['rollout_eligible'] is False
