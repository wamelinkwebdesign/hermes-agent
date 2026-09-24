import importlib.util
import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pilot_core', ROOT / 'collector.py')
assert spec and spec.loader
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / '.test-runtime')
        self.home = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def payload(self, **extra):
        return dict(session_id='session', turn_id='turn', api_request_id='request',
                    model='gpt-6-astra', provider='openai-codex', **extra)

    def rows(self):
        return core.read_snapshot(self.home)[0]

    def send(self, event, **kw):
        core.store(self.home, core.sanitize(event, self.payload(**kw)))

    def test_post_before_pre_and_duplicate(self):
        usage = dict(input_tokens=10, cache_read_tokens=20, cache_write_tokens=3,
                     output_tokens=7, reasoning_tokens=5)
        self.send('post_api_request', usage=usage)
        self.send('post_api_request', usage=usage)
        self.send('pre_api_request', retry_count=2)
        row, = self.rows()
        self.assertEqual(row['output_tokens'], 7)
        self.assertIsNone(row['retry_count'])
        self.assertEqual(row['attempt_precision'], 'terminal_retry_count_unavailable')
        self.assertEqual(row['status'], 'ok')
        self.assertEqual(row['input_tokens'], 10)

    def test_same_request_id_retry_preserves_error_and_success_once(self):
        usage = dict(input_tokens=10, cache_read_tokens=2, cache_write_tokens=0,
                     output_tokens=7, reasoning_tokens=3)
        self.send('pre_api_request', retry_count=0, started_at=100.0)
        self.send('api_request_error', retry_count=0, started_at=100.0,
                  ended_at=101.0, api_duration=1.0)
        self.send('pre_api_request', retry_count=1, started_at=100.0)
        self.send('post_api_request', usage=usage, started_at=100.0,
                  ended_at=103.0, api_duration=3.0, first_chunk_at=102.0)
        self.send('post_api_request', usage=usage, started_at=100.0,
                  ended_at=103.0, api_duration=3.0, first_chunk_at=102.0)
        rows = self.rows()
        self.assertEqual([r['status'] for r in rows], ['error', 'ok'])
        self.assertEqual(rows[0]['retry_count'], 0)
        self.assertIsNone(rows[1]['retry_count'])
        self.assertEqual(rows[1]['attempt_precision'], 'terminal_retry_count_unavailable')
        self.assertEqual(sum(r['output_tokens'] or 0 for r in rows), 7)
        self.assertEqual(rows[1]['api_duration'], 3.0)
        self.assertEqual(rows[1]['timing_scope'], 'logical_request_cumulative')

    def test_same_request_retry_materialization_is_order_independent(self):
        usage = {'input_tokens': 4, 'cache_read_tokens': 0, 'cache_write_tokens': 0,
                 'output_tokens': 2, 'reasoning_tokens': 1}
        self.send('post_api_request', usage=usage)
        self.send('pre_api_request', retry_count=1)
        self.send('api_request_error', retry_count=0)
        self.send('pre_api_request', retry_count=0)
        self.send('post_api_request', usage=usage)
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual({r['status'] for r in rows}, {'error', 'ok'})
        self.assertEqual(sum(r['output_tokens'] or 0 for r in rows), 2)

    def test_auxiliary_same_id_retries_keep_each_attempt(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse), tempfile.TemporaryDirectory(dir=ROOT / '.test-runtime') as directory:
                events = [
                    ('pre_auxiliary_call', {'retry_count': 0}),
                    ('post_auxiliary_call', {'retry_count': 0, 'error': 'synthetic failure'}),
                    ('pre_auxiliary_call', {'retry_count': 1}),
                    ('post_auxiliary_call', {'retry_count': 1, 'usage': {'input_tokens': 8, 'output_tokens': 5}}),
                ]
                if reverse:
                    events.reverse()
                for event, values in events + events:
                    core.store(Path(directory), core.sanitize(event, self.payload(**values)))
                rows, _ = core.read_snapshot(Path(directory))
                self.assertEqual([(row['retry_count'], row['status']) for row in rows], [(0, 'error'), (1, 'ok')])
                self.assertIsNone(rows[0]['output_tokens'])
                self.assertEqual(rows[1]['output_tokens'], 5)
                self.assertTrue(all(row['seen_pre'] and row['seen_post'] for row in rows))

    def test_absent_malformed_and_zero_usage(self):
        self.send('post_api_request', usage={'input_tokens': False, 'output_tokens': -1})
        row, = self.rows()
        self.assertIsNone(row['input_tokens'])
        self.assertIsNone(row['output_tokens'])
        self.assertIsNone(row['cache_read_tokens'])
        self.assertEqual(core.sanitize('post_api_request', self.payload(usage={'input_tokens': 0}))['input_tokens'], 0)

    def test_sensitive_fields_never_persist(self):
        sentinel = 'PRIVATE-秘密-é-🔒'
        self.send('post_api_request', usage={'input_tokens': 2, 'nested': {'secret': sentinel}},
                  request={'messages': sentinel}, response=sentinel, error=sentinel,
                  unknown={'deep': [sentinel]}, base_url=sentinel, user_label=sentinel)
        p = self.payload(); p.update(model=sentinel, provider=sentinel, session_id=sentinel,
                                     task_id=sentinel)
        core.store(self.home, core.sanitize('pre_api_request', p))
        for path in self.home.rglob('*'):
            if path.is_file():
                self.assertNotIn(sentinel.encode(), path.read_bytes())

    def test_task_identity_and_numeric_timings_are_fixed_scalars(self):
        row = core.sanitize('post_api_request', self.payload(
            task_id='run-a', started_at=10, ended_at=12.5, api_duration=2.5,
            first_chunk_at=11.25))
        self.assertEqual(row['task'], core.opaque('run-a'))
        self.assertEqual((row['started_at'], row['ended_at'], row['api_duration'],
                          row['first_chunk_at']), (10, 12.5, 2.5, 11.25))
        self.assertTrue(math.isfinite(row['observed_at']))
        for bad in (True, -1, float('nan'), float('inf'), '1'):
            invalid = core.sanitize('post_api_request', self.payload(
                started_at=bad, ended_at=bad, api_duration=bad, first_chunk_at=bad))
            self.assertEqual([invalid[k] for k in
                ('started_at', 'ended_at', 'api_duration', 'first_chunk_at')], [None] * 4)

    def test_subagent_start_stop_merge_bounded_outcome(self):
        base = dict(parent_session_id='parent', parent_turn_id='turn', child_session_id='child')
        core.store(self.home, core.sanitize('subagent_start', base))
        core.store(self.home, core.sanitize('subagent_stop', dict(
            base, child_status='completed', duration_ms=1234, child_summary='PRIVATE')))
        _, links = core.read_snapshot(self.home)
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]['seen_start'], 1)
        self.assertEqual(links[0]['seen_stop'], 1)
        self.assertEqual(links[0]['child_status'], 'completed')
        self.assertEqual(links[0]['duration_ms'], 1234)
        bad = core.sanitize('subagent_stop', dict(base, child_status='PRIVATE', duration_ms=True))
        self.assertIsNone(bad['child_status'])
        self.assertIsNone(bad['duration_ms'])

    def test_symlink_refusal(self):
        outside = self.home / 'outside'; outside.mkdir()
        (self.home / 'task-accounting').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(OSError):
            self.send('pre_api_request')
        self.assertEqual(list(outside.iterdir()), [])

    def test_db_symlink_and_bad_db(self):
        directory = self.home / 'task-accounting'; directory.mkdir(mode=0o700)
        victim = self.home / 'victim'; victim.write_text('unchanged')
        db = directory / 'attempts.sqlite3'; db.symlink_to(victim)
        with self.assertRaises(OSError): self.send('pre_api_request')
        self.assertEqual(victim.read_text(), 'unchanged')
        db.unlink(); db.write_text('not sqlite'); db.chmod(0o600)
        with self.assertRaises(sqlite3.DatabaseError): self.send('pre_api_request')

    def test_moa_outer_not_physical(self):
        p = self.payload(usage={'input_tokens': 999}); p['provider'] = 'moa'
        row = core.sanitize('post_api_request', p)
        self.assertEqual(row['status'], 'aggregate_excluded')
        self.assertIsNone(row['input_tokens'])

    def test_conflicting_post_not_silently_recounted(self):
        self.send('post_api_request', usage={'input_tokens': 2})
        self.send('post_api_request', usage={'input_tokens': 3})
        self.assertEqual(self.rows()[0]['conflict'], 1)

    def test_permissions(self):
        self.send('pre_api_request')
        self.assertEqual((self.home / 'task-accounting').stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.home / 'task-accounting/attempts.sqlite3').stat().st_mode & 0o777, 0o600)


if __name__ == '__main__':
    unittest.main()
