"""Real discovery/lifecycle tests. Only synthetic homes, no runtime monkeypatching."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT / '.test-runtime')
        cls.base = Path(cls.temp.name)
        # Before Hermes imports: no real profile, user configuration or secret files.
        os.environ['HERMES_HOME'] = str(cls.base)
        os.environ['HOME'] = str(cls.base)
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        from hermes_cli import plugins, lifecycle
        cls.set_home = staticmethod(set_hermes_home_override)
        cls.reset_home = staticmethod(reset_hermes_home_override)
        cls.plugins = plugins
        cls.lifecycle = lifecycle

    @classmethod
    def tearDownClass(cls):
        cls.plugins._reset_plugin_managers_for_tests()
        cls.temp.cleanup()

    def home(self, enabled=False, load=True):
        home = Path(tempfile.mkdtemp(dir=self.base)); home.chmod(0o700)
        package = home / 'plugins/task-accounting'; package.mkdir(parents=True)
        for name in ('__init__.py', 'collector.py', 'plugin.yaml'):
            shutil.copyfile(ROOT / name, package / name)
        cfg = {'plugins': {'enabled': ['task-accounting'] if load else [],
                           'entries': {'task-accounting': {'settings': {'enabled': enabled}}}},
               'observability': {'enabled': False}}
        (home / 'config.yaml').write_text(json.dumps(cfg))
        token = self.set_home(home)
        try:
            manager = self.plugins.get_plugin_manager()
            manager.discover_and_load()
        finally:
            self.reset_home(token)
        return home, manager

    def emit(self, home, event, **kwargs):
        token = self.set_home(home)
        try:
            return self.lifecycle.invoke_hook(event, **kwargs)
        finally:
            self.reset_home(token)

    def observer(self, manager):
        return manager._hooks['pre_api_request'][0].func.__self__

    def snapshot(self, home, manager):
        observer = self.observer(manager)
        self.assertTrue(observer.flush(2))
        from test_collector import core
        return core.read_snapshot(home)

    def test_disabled_no_storage_or_worker(self):
        home, manager = self.home()
        self.emit(home, 'pre_api_request', api_request_id='disabled')
        self.assertFalse((home / 'task-accounting').exists())
        self.assertIsNone(self.observer(manager).worker)
        home2, manager2 = self.home(load=False)
        self.assertFalse(manager2._plugins['task-accounting'].enabled)
        self.assertFalse((home2 / 'task-accounting').exists())

    def test_real_payloads_profiles_and_byte_invariance(self):
        from agent.api_request_hooks import ApiRequestHooksMixin
        from agent.auxiliary_hooks import _AuxCallHooks
        from agent.turn_response_intake import _fire_post_api_request_hook
        from test_collector import core
        a, ma = self.home(True); b, mb = self.home(True)
        sentinel = 'PRIVATE-秘密-é-🔒'
        wire = {'messages': [{'role': 'user', 'content': sentinel}], 'tools': [
            {'type': 'function', 'function': {'name': 'fixture', 'parameters': {'type': 'object'}}}]}
        before = json.dumps(wire, sort_keys=True, ensure_ascii=False)
        schemas = copy.deepcopy(wire['tools'])
        schema_bytes = json.dumps(schemas, sort_keys=True)
        registered_before = set(ma._plugin_tool_names)
        class Agent(ApiRequestHooksMixin):
            provider = 'openai-codex'; api_mode = 'codex_responses'; model = 'gpt-6-astra'
            session_id = 'parent'; platform = 'cli'; base_url = sentinel
        agent = Agent()
        response = SimpleNamespace(usage={'input_tokens': 30, 'output_tokens': 7,
            'input_tokens_details': {'cached_tokens': 20, 'cache_write_tokens': 0},
            'output_tokens_details': {'reasoning_tokens': 5}}, model=agent.model)
        message = SimpleNamespace(content=sentinel, tool_calls=[])
        for home, manager, request in ((a, ma, 'r1'), (b, mb, 'r2'), (a, ma, 'r3')):
            self.emit(home, 'pre_api_request', session_id='parent', turn_id='turn', api_request_id=request,
                      task_id='run-' + request, model=agent.model, provider=agent.provider,
                      retry_count=0, request=wire,
                      unknown={'nested': [sentinel]})
            token = self.set_home(home)
            try:
                _fire_post_api_request_hook(agent, response, message, 'stop', api_messages=wire['messages'],
                    api_call_count=1, api_duration=0.1, api_start_time=1.0, api_request_id=request,
                    effective_task_id='run-' + request, turn_id='turn')
            finally: self.reset_home(token)
        self.assertEqual(len(self.snapshot(a, ma)[0]), 2)
        self.assertEqual(len(self.snapshot(b, mb)[0]), 1)
        row = self.snapshot(a, ma)[0][0]
        self.assertEqual((row['input_tokens'], row['cache_read_tokens'], row['output_tokens'], row['reasoning_tokens']), (10, 20, 7, 5))
        self.assertIsNotNone(row['task'])
        token = self.set_home(a)
        try:
            aux = _AuxCallHooks(aux_task='compression', metadata={'api_request_id': 'aux1', 'retry_count': 1},
                client=SimpleNamespace(base_url=sentinel), kwargs=wire, provider=agent.provider,
                model=agent.model, api_mode=agent.api_mode, streaming=False)
            aux.base.update(session_id='parent', task_id='run-aux', turn_id='turn')
            aux.post(response); aux.pre(); aux.post(response)
        finally: self.reset_home(token)
        rows, _ = self.snapshot(a, ma)
        self.assertEqual(len(rows), 3)
        self.assertEqual([r for r in rows if r['source'] == 'aux'][0]['retry_count'], 1)
        self.assertEqual([r for r in rows if r['source'] == 'aux'][0]['task'], core.opaque('run-aux'))
        self.assertEqual(json.dumps(wire, sort_keys=True, ensure_ascii=False), before)
        self.assertEqual(json.dumps(schemas, sort_keys=True), schema_bytes)
        self.assertEqual(ma._plugin_tool_names, registered_before)
        self.assertFalse(ma._system_prompt_sections)
        self.assertEqual(json.dumps(wire['tools'], sort_keys=True), schema_bytes)
        for home in (a, b):
            for path in home.rglob('*'):
                if path.is_file(): self.assertNotIn(sentinel.encode(), path.read_bytes())

    def test_runtime_gate_a_disabled_b_a(self):
        a, ma = self.home(True); b, mb = self.home(False)
        observer = self.observer(ma)
        for home, request in ((a, 'one'), (b, 'two'), (a, 'three')):
            token = self.set_home(home)
            try:
                observer.observe('pre_api_request', api_request_id=request)
            finally: self.reset_home(token)
        self.assertEqual(len(self.snapshot(a, ma)[0]), 2)
        self.assertFalse((b / 'task-accounting').exists())

    def test_parent_children_errors_and_aux_stream(self):
        from test_collector import core
        from test_report import manifest, member
        import report
        home, manager = self.home(True)
        usage = dict(input_tokens=10, cache_read_tokens=0, cache_write_tokens=0, output_tokens=3, reasoning_tokens=1)
        for session in ('parent', 'child1', 'child2'):
            self.emit(home, 'post_api_request', session_id=session, turn_id='t', api_request_id=session,
                      provider='openai-codex', model='gpt-6-astra', usage=usage,
                      moa_references=[{'usage': {'input_tokens': 9000}}], children={'tokens': 9000})
        for child in ('child1', 'child2'):
            self.emit(home, 'subagent_start', parent_session_id='parent', parent_turn_id='t', child_session_id=child)
            self.emit(home, 'subagent_stop', parent_session_id='parent', parent_turn_id='t',
                      child_session_id=child, child_status='completed', duration_ms=50)
        self.emit(home, 'api_request_error', session_id='other', turn_id='t', api_request_id='retry0', retry_count=0,
                  error={'message': 'PRIVATE-秘密-é-🔒'})
        self.emit(home, 'post_auxiliary_call', session_id='other', turn_id='t', api_request_id='retry1', retry_count=1,
                  streaming=True, usage=None)
        rows, links = self.snapshot(home, manager)
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(links), 2)
        self.assertTrue(all(link['seen_stop'] for link in links))
        result = report.build_report(rows, links, manifest([member('parent')]), 'a' * 32)
        self.assertEqual(result['descendant_inclusive']['tokens']['input_tokens'], 30)
        self.assertEqual(result['parent_only']['attempts'], 1)
        self.assertEqual(result['unattributed_attempts'], 2)
        self.assertEqual([r for r in rows if r['request'] == core.opaque('retry0')][0]['status'], 'error')

    def test_actual_main_emitters_reuse_request_id_without_losing_success(self):
        from agent.api_request_hooks import ApiRequestHooksMixin
        from agent.turn_response_intake import _fire_post_api_request_hook
        from test_collector import core
        home, manager = self.home(True)
        class Agent(ApiRequestHooksMixin):
            provider = 'openai-codex'; api_mode = 'codex_responses'; model = 'gpt-6-astra'
            session_id = 'session'; platform = 'cli'; base_url = 'https://example.invalid'
        agent = Agent()
        response = SimpleNamespace(usage={'input_tokens': 5, 'output_tokens': 2}, model=agent.model)
        message = SimpleNamespace(content='ok', tool_calls=[])
        self.emit(home, 'pre_api_request', task_id='run', session_id='session', turn_id='turn',
                  api_request_id='same', provider=agent.provider, model=agent.model,
                  retry_count=0, started_at=1.0)
        token = self.set_home(home)
        try:
            agent._invoke_api_request_error_hook(task_id='run', turn_id='turn',
                api_request_id='same', api_call_count=1, api_start_time=1.0,
                api_kwargs={}, error_type='SyntheticError', error_message='PRIVATE', retry_count=0)
        finally:
            self.reset_home(token)
        self.emit(home, 'pre_api_request', task_id='run', session_id='session', turn_id='turn',
                  api_request_id='same', provider=agent.provider, model=agent.model,
                  retry_count=1, started_at=1.0)
        token = self.set_home(home)
        try:
            _fire_post_api_request_hook(agent, response, message, 'stop', api_messages=[],
                api_call_count=2, api_duration=3.0, api_start_time=1.0,
                api_request_id='same', effective_task_id='run', turn_id='turn')
        finally:
            self.reset_home(token)
        rows, _ = self.snapshot(home, manager)
        self.assertEqual([row['status'] for row in rows], ['error', 'ok'])
        self.assertEqual(sum(row['output_tokens'] or 0 for row in rows), 2)
        self.assertTrue(all(row['task'] == core.opaque('run') for row in rows))
        self.assertEqual(rows[1]['api_duration'], 3.0)

    def test_bad_database_and_lock_are_fail_open(self):
        import sqlite3
        home, manager = self.home(True)
        self.emit(home, 'pre_api_request', api_request_id='initial')
        self.snapshot(home, manager)
        dbpath = home / 'task-accounting/attempts.sqlite3'
        db = sqlite3.connect(dbpath); db.execute('BEGIN EXCLUSIVE')
        try:
            before = time.monotonic()
            self.emit(home, 'pre_api_request', api_request_id='locked')
            self.assertTrue(self.observer(manager).flush(1))
            self.assertLess(time.monotonic() - before, 1)
        finally: db.rollback(); db.close()
        dbpath.write_bytes(b'not sqlite')
        self.emit(home, 'pre_api_request', api_request_id='broken')
        self.assertTrue(self.observer(manager).flush(1))
        self.assertGreaterEqual(self.observer(manager).failed, 2)

    def test_failure_does_not_break_lifecycle(self):
        home, manager = self.home(True)
        outside = home / 'outside'; outside.mkdir()
        (home / 'task-accounting').symlink_to(outside, target_is_directory=True)
        start = time.monotonic()
        result = self.emit(home, 'pre_api_request', api_request_id='r')
        self.assertLess(time.monotonic() - start, 1)
        self.assertTrue(self.observer(manager).flush(2))
        self.assertGreater(self.observer(manager).failed, 0)
        self.assertTrue(all(r is None for r in result))
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == '__main__': unittest.main()
