import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from test_collector import core

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def test_explicit_manifest_cli_end_to_end(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.test-runtime') as directory:
            home = Path(directory); manifest = home / 'manifest.json'
            def run(*args, text=None, expected=0):
                result = subprocess.run([sys.executable, str(ROOT / 'report.py'), *args],
                    input=text, capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, expected, result.stderr)
                return result.stdout
            task = run('init', '--manifest', str(manifest)).strip()
            session = run('identity', text='session\n').strip()
            value = json.loads(manifest.read_text())
            value['tasks'][0]['members'] = [{'session': session, 'turns': None, 'parent': None}]
            manifest.write_text(json.dumps(value))
            row = core.sanitize('post_api_request', {'session_id': 'session', 'turn_id': 't',
                'api_request_id': 'request', 'provider': 'openai-codex', 'model': 'gpt-6-astra',
                'usage': {'input_tokens': 10, 'cache_read_tokens': 0, 'cache_write_tokens': 0,
                          'output_tokens': 3, 'reasoning_tokens': 2}})
            core.store(home, row)
            inventory = json.loads(run('inventory', '--home', str(home)))
            self.assertEqual(len(inventory['attempts']), 1)
            args = ('report', '--home', str(home), '--manifest', str(manifest), '--task', task)
            self.assertFalse(json.loads(run(*args))['operator_asserted_completed'])
            run('complete', '--home', str(home), '--manifest', str(manifest), '--task', task, expected=2)
            run('complete', '--home', str(home), '--manifest', str(manifest), '--task', task,
                '--assert-completed')
            result = json.loads(run(*args))
            self.assertTrue(result['operator_asserted_completed'])
            self.assertIsNone(result['complete_task_cost'])
            self.assertEqual(result['parent_only']['tokens']['output_tokens'], 3)
            observed_task = core.opaque('run-task')
            row = core.sanitize('post_api_request', {'session_id': 'session', 'turn_id': 'other',
                'task_id': 'run-task', 'api_request_id': 'request-2', 'provider': 'openai-codex',
                'model': 'gpt-6-astra', 'usage': {'input_tokens': 1, 'cache_read_tokens': 0,
                'cache_write_tokens': 0, 'output_tokens': 1, 'reasoning_tokens': 0}})
            core.store(home, row)
            auto = json.loads(run('observed-task', '--home', str(home), observed_task))
            self.assertEqual(auto['observed_usage']['attempts'], 1)
            (ROOT / 'evidence/cli-smoke.json').write_text(json.dumps(result, indent=2) + '\n')

    def test_complete_rejects_empty_or_unobserved_group(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.test-runtime') as directory:
            home = Path(directory); (home / 'task-accounting').mkdir(mode=0o700)
            # Create the database through the collector, but observe a different session.
            core.store(home, core.sanitize('pre_api_request', {'session_id': 'other',
                'turn_id': 't', 'api_request_id': 'r'}))
            manifest = home / 'manifest.json'; task = 'a' * 32
            manifest.write_text(json.dumps({'version': 1, 'tasks': [{
                'id': task, 'operator_asserted_completed': False,
                'operator_asserted_lineage_complete': False, 'observed_tasks': [], 'members': []}]}))
            command = [sys.executable, str(ROOT / 'report.py'), 'complete', '--home', str(home),
                       '--manifest', str(manifest), '--task', task, '--assert-completed']
            self.assertEqual(subprocess.run(command, capture_output=True, text=True).returncode, 2)
            value = json.loads(manifest.read_text())
            value['tasks'][0]['members'] = [{'session': core.opaque('missing'),
                                              'turns': None, 'parent': None}]
            manifest.write_text(json.dumps(value))
            self.assertEqual(subprocess.run(command, capture_output=True, text=True).returncode, 2)

    def test_complete_rejects_mixed_pending_and_conflicted_rows(self):
        for unfinished in ('pending', 'conflicted'):
            with self.subTest(unfinished=unfinished), tempfile.TemporaryDirectory(dir=ROOT / '.test-runtime') as directory:
                home = Path(directory)
                payload = {'session_id': 'session', 'turn_id': 'turn', 'task_id': 'run',
                           'api_request_id': 'ok', 'usage': {'input_tokens': 10}}
                core.store(home, core.sanitize('post_api_request', payload))
                second = dict(payload, api_request_id='unfinished')
                if unfinished == 'pending':
                    core.store(home, core.sanitize('pre_api_request', second))
                else:
                    core.store(home, core.sanitize('post_api_request', second))
                    second['usage'] = {'input_tokens': 11}
                    core.store(home, core.sanitize('post_api_request', second))
                manifest = home / 'manifest.json'
                manifest.write_text(json.dumps({'version': 1, 'tasks': [{
                    'id': 'a' * 32, 'operator_asserted_completed': False,
                    'operator_asserted_lineage_complete': False,
                    'observed_tasks': [core.opaque('run')], 'members': []}]}))
                result = subprocess.run([sys.executable, str(ROOT / 'report.py'), 'complete',
                    '--home', str(home), '--manifest', str(manifest), '--task', 'a' * 32,
                    '--assert-completed'], capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(json.loads(manifest.read_text())['tasks'][0]['operator_asserted_completed'])


if __name__ == '__main__': unittest.main()
