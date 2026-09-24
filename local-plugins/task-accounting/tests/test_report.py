import importlib.util
import unittest
from pathlib import Path
from test_collector import core

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pilot_report', ROOT / 'report.py')
assert spec and spec.loader
report = importlib.util.module_from_spec(spec); spec.loader.exec_module(report)
h = core.opaque


def member(session, parent=None, turns=None):
    return {'session': h(session), 'turns': None if turns is None else [h(t) for t in turns],
            'parent': h(parent) if parent else None}


def manifest(members, observed_tasks=None):
    return {'version': 1, 'tasks': [{'id': 'a' * 32, 'operator_asserted_completed': False,
            'operator_asserted_lineage_complete': False,
            'observed_tasks': observed_tasks or [], 'members': members}]}


def attempt(session, request, **extra):
    return core.sanitize('post_api_request', dict(session_id=session, turn_id='t', api_request_id=request,
        provider='openai-codex', model='gpt-6-astra', usage=dict(input_tokens=10, cache_read_tokens=20,
        cache_write_tokens=0, output_tokens=7, reasoning_tokens=5), **extra))


class ReportTests(unittest.TestCase):
    def test_parent_two_children_and_explicit_completion(self):
        rows = [attempt('p', 'r1'), attempt('c1', 'r2'), attempt('c2', 'r3'), attempt('other', 'r4')]
        m = manifest([member('p'), member('c1', 'p'), member('c2', 'p')])
        result = report.build_report(rows, [], m, 'a' * 32)
        self.assertEqual(result['parent_only']['tokens']['input_tokens'], 10)
        self.assertEqual(result['descendant_inclusive']['tokens']['input_tokens'], 30)
        self.assertEqual(result['descendant_inclusive']['tokens']['output_tokens'], 21)
        self.assertEqual(result['unattributed_attempts'], 1)
        self.assertFalse(result['operator_asserted_completed'])
        self.assertIsNone(result['complete_task_cost'])
        m['tasks'][0]['operator_asserted_completed'] = True
        self.assertTrue(report.build_report(rows, [], m, 'a' * 32)['operator_asserted_completed'])

    def test_overlap_and_cycle_rejected(self):
        m = manifest([member('p'), member('c', 'p')])
        m['tasks'].append(dict(m['tasks'][0], id='b' * 32))
        with self.assertRaises(ValueError): report.build_report([], [], m, 'a' * 32)
        with self.assertRaises(ValueError):
            report.build_report([], [], manifest([member('p', 'c'), member('c', 'p')]), 'a' * 32)

    def test_turn_selection_and_observed_child_link(self):
        rows = [attempt('p', 'one'), attempt('c', 'two')]
        links = [core.sanitize('subagent_start', dict(parent_session_id='p', parent_turn_id='t', child_session_id='c'))]
        r = report.build_report(rows, links, manifest([member('p', turns=['t'])]), 'a' * 32)
        self.assertEqual(r['descendant_inclusive']['attempts'], 2)
        self.assertIn('lineage_not_operator_asserted_complete', r['coverage_flags'])

    def test_missing_usage_and_rates_do_not_become_zero(self):
        row = attempt('p', 'r'); row['output_tokens'] = None
        r = report.build_report([row], [], manifest([member('p')]), 'a' * 32)
        self.assertIsNone(r['parent_only']['tokens']['output_tokens'])
        self.assertIsNone(r['parent_only']['observed_estimate'])

    def test_explicit_versioned_rates(self):
        rates = {'version': 1, 'revision': 'synthetic-test-only', 'currency': 'USD', 'rates': [{
            'provider': 'openai-codex', 'model': 'gpt-6-astra', 'input_tokens': '1',
            'cache_read_tokens': '0.5', 'cache_write_tokens': '2', 'output_tokens': '4'}]}
        m = manifest([member('p')]); m['tasks'][0].update(
            operator_asserted_completed=True, operator_asserted_lineage_complete=True)
        r = report.build_report([attempt('p', 'r')], [], m, 'a' * 32, rates)
        self.assertEqual(r['parent_only']['observed_estimate'], '0.000048')
        self.assertIsNone(r['complete_task_cost'])
        self.assertIn('provider_internal_retries_unknown', r['coverage_flags'])

    def test_observed_task_report_does_not_mix_concurrent_same_session_runs(self):
        a = attempt('shared', 'r1', task_id='run-a')
        b = attempt('shared', 'r2', task_id='run-b')
        result = report.build_observed_task_report([a, b], [], h('run-a'))
        self.assertEqual(result['observed_task'], h('run-a'))
        self.assertEqual(result['observed_usage']['attempts'], 1)
        self.assertEqual(result['observed_usage']['tokens']['output_tokens'], 7)
        self.assertIn('explicit_operator_grouping_required_for_multi_run_goal', result['coverage_flags'])

    def test_explicit_manifest_groups_run_ids_without_same_session_leakage(self):
        rows = [attempt('shared', 'r1', task_id='run-a'),
                attempt('shared', 'r2', task_id='run-b')]
        one = report.build_report(rows, [], manifest([], [h('run-a')]), 'a' * 32)
        self.assertEqual(one['descendant_inclusive']['attempts'], 1)
        both = report.build_report(rows, [], manifest([], [h('run-a'), h('run-b')]), 'a' * 32)
        self.assertEqual(both['descendant_inclusive']['attempts'], 2)

    def test_inventory_groups_observed_task_identity_and_time(self):
        rows = [attempt('shared', 'r1', task_id='run-a', started_at=10.0, ended_at=12.0),
                attempt('shared', 'r2', task_id='run-b', started_at=11.0, ended_at=13.0)]
        inventory = report.build_inventory(rows, [])
        self.assertEqual(len(inventory['observed_tasks']), 2)
        self.assertEqual({g['observed_task'] for g in inventory['observed_tasks']},
                         {h('run-a'), h('run-b')})
        self.assertTrue(all(g['first_observed_at'] <= g['last_observed_at']
                            for g in inventory['observed_tasks']))

    def test_retry_error_attempt_stays_unknown_without_losing_success_usage(self):
        rows = []
        for event, extra in (
            ('pre_api_request', {'retry_count': 0}),
            ('api_request_error', {'retry_count': 0}),
            ('pre_api_request', {'retry_count': 1}),
            ('post_api_request', {'usage': {'input_tokens': 10, 'cache_read_tokens': 0,
                'cache_write_tokens': 0, 'output_tokens': 7, 'reasoning_tokens': 2}})):
            core.store(self._home, core.sanitize(event, dict(session_id='p', turn_id='t',
                api_request_id='same', provider='openai-codex', model='gpt-6-astra', **extra)))
        rows, _ = core.read_snapshot(self._home)
        r = report.build_report(rows, [], manifest([member('p')]), 'a' * 32)
        self.assertEqual(r['parent_only']['attempts'], 2)
        self.assertEqual(r['parent_only']['observed_error_attempts'], 1)
        self.assertEqual(r['parent_only']['known_subtotals']['output_tokens'], 7)
        self.assertIsNone(r['parent_only']['tokens']['output_tokens'])
        self.assertIn('terminal_attempt_retry_count_unavailable', r['coverage_flags'])

    def setUp(self):
        import tempfile
        self._temp = tempfile.TemporaryDirectory(dir=ROOT / '.test-runtime')
        self._home = Path(self._temp.name)

    def tearDown(self):
        self._temp.cleanup()


if __name__ == '__main__': unittest.main()
