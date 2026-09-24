"""Content-free local accounting. No network or provider imports."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import time
import uuid
from contextlib import contextmanager

BUCKETS = ('input_tokens', 'cache_read_tokens', 'cache_write_tokens', 'output_tokens', 'reasoning_tokens')
PROVIDERS = frozenset(('openai', 'openai-codex', 'anthropic', 'moa', 'openrouter', 'nous', 'google', 'xai'))
MODELS = frozenset(('gpt-6-astra', 'gpt-5', 'gpt-5-mini', 'claude-sonnet-4-5', 'claude-opus-4-1'))
CHILD_STATUSES = frozenset(('completed', 'interrupted', 'failed', 'error', 'timeout'))
EVENTS = {'pre_api_request': ('main', 'pre'), 'post_api_request': ('main', 'post'),
          'api_request_error': ('main', 'error'), 'pre_auxiliary_call': ('aux', 'pre'),
          'post_auxiliary_call': ('aux', 'post')}
TIMINGS = ('started_at', 'ended_at', 'api_duration', 'first_chunk_at')


def opaque(value):
    if type(value) is not str or not value or len(value) > 4096:
        return None
    return hashlib.sha256(value.encode('utf-8', errors='replace')).hexdigest()


def label(value, allowed):
    return value if type(value) is str and value in allowed else opaque(value)


def number(value):
    return value if type(value) is int and 0 <= value <= 2**53 else None


def finite_number(value):
    if type(value) not in (int, float) or value < 0 or not math.isfinite(value):
        return None
    return value


def _event_key(group, phase, retry_count):
    retry = str(retry_count) if retry_count is not None else 'unknown'
    return opaque(f'{group}:{phase}:{retry}')


def sanitize(event, data):
    """Read named scalar fields only; never traverse or retain arbitrary payloads."""
    observed_at = time.time()
    if event in ('subagent_start', 'subagent_stop'):
        phase = 'start' if event == 'subagent_start' else 'stop'
        return {'link': True, 'parent': opaque(data.get('parent_session_id')),
                'parent_turn': opaque(data.get('parent_turn_id')),
                'child': opaque(data.get('child_session_id')),
                'seen_start': int(phase == 'start'), 'seen_stop': int(phase == 'stop'),
                'child_status': (data.get('child_status') if phase == 'stop' and
                                 type(data.get('child_status')) is str and
                                 data.get('child_status') in CHILD_STATUSES else None),
                'duration_ms': number(data.get('duration_ms')) if phase == 'stop' else None,
                'observed_at': observed_at, 'conflict': 0}
    source, phase = EVENTS[event]
    request = opaque(data.get('api_request_id'))
    session = opaque(data.get('session_id'))
    # A missing request identity cannot be safely deduplicated, so give it a unique group.
    group = opaque(source + ':' + (session or '') + ':' + request) if request else uuid.uuid4().hex
    retry_count = number(data.get('retry_count'))
    if source == 'main' and phase == 'post':
        retry_count = None
    aggregate = source == 'main' and data.get('provider') == 'moa'
    usage = data.get('usage') if phase == 'post' and not aggregate else None
    usage = usage if type(usage) is dict else {}
    status = 'pending' if phase == 'pre' else 'ok'
    if phase == 'error' or (source == 'aux' and data.get('error') is not None):
        status = 'error'
    if aggregate:
        status = 'aggregate_excluded'
    precision = 'hook_retry_count' if retry_count is not None else 'retry_count_unavailable'
    if source == 'main' and phase == 'post':
        # Hermes omits retry_count from the terminal-success hook. We may associate it with an
        # observed unclosed pre event, but must not claim an exact terminal retry ordinal.
        precision = 'terminal_retry_count_unavailable'
    row = {'key': _event_key(group, phase, retry_count), 'request_group': group,
           'source': source, 'phase': phase, 'request': request, 'session': session,
           'task': opaque(data.get('task_id')), 'turn': opaque(data.get('turn_id')),
           'model': label(data.get('response_model') or data.get('model'), MODELS),
           'provider': label(data.get('provider'), PROVIDERS), 'status': status,
           'retry_count': retry_count, 'api_call_count': number(data.get('api_call_count')),
           'aux_task': opaque(data.get('aux_task')), 'streaming': data.get('streaming') is True,
           'seen_pre': int(phase == 'pre'), 'seen_post': int(phase != 'pre'),
           'attempt_precision': precision,
           'timing_scope': ('logical_request_cumulative' if source == 'main'
                            else 'physical_aux_attempt'),
           'observed_at': observed_at, 'conflict': 0}
    row.update({name: finite_number(data.get(name)) for name in TIMINGS})
    row.update({key: number(usage.get(key)) for key in BUCKETS})
    # Reasoning is a subset of output, never an additional billable bucket.
    if row['reasoning_tokens'] is not None and row['output_tokens'] is not None:
        if row['reasoning_tokens'] > row['output_tokens']:
            row['reasoning_tokens'] = None
            row['conflict'] = 1
    return row


def _checked(path, directory=False):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or (directory and not stat.S_ISDIR(info.st_mode)):
        raise OSError('unsafe accounting path')
    if not directory and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
        raise OSError('unsafe accounting file')
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise OSError('accounting path must be private and owned')


def db_path(home, create=False):
    home = Path(home).absolute()
    for ancestor in (home, *home.parents):
        if ancestor.is_symlink():
            raise OSError('symlink profile ancestry')
    directory = home / 'task-accounting'
    if create:
        directory.mkdir(mode=0o700, exist_ok=True)
    _checked(directory, directory=True)
    path = directory / 'attempts.sqlite3'
    if create and not path.exists() and not path.is_symlink():
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        os.close(fd)
    _checked(path)
    for suffix in ('-journal', '-wal', '-shm'):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            _checked(sidecar)
    return path


@contextmanager
def connection(home, create=False):
    path = db_path(home, create=create)
    db = sqlite3.connect(path.as_uri() + ('?mode=rw' if create else '?mode=ro'), uri=True, timeout=0.025)
    try:
        deadline = time.monotonic() + 0.05
        db.set_progress_handler(lambda: int(time.monotonic() > deadline), 100)
        if create:
            db.execute('PRAGMA journal_mode=DELETE')
            db.execute('PRAGMA max_page_count=4096')
            # Schema is private and this pilot has never been installed, so rows are projected
            # observations and can be materialized deterministically at read time.
            db.execute('CREATE TABLE IF NOT EXISTS attempts (key TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS links (key TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        yield db
        if create:
            db.commit()
    finally:
        db.close()


def _merge_duplicate(old, row):
    merged = dict(old)
    ignored = {'observed_at', 'seen_pre', 'seen_post', 'conflict'}
    if any(old.get(k) is not None and row.get(k) is not None and old.get(k) != row.get(k)
           for k in old.keys() | row.keys() if k not in ignored):
        merged['conflict'] = 1
    for key, value in row.items():
        if merged.get(key) is None and value is not None:
            merged[key] = value
    merged['observed_at'] = min(old['observed_at'], row['observed_at'])
    merged['seen_pre'] = max(old.get('seen_pre', 0), row.get('seen_pre', 0))
    merged['seen_post'] = max(old.get('seen_post', 0), row.get('seen_post', 0))
    merged['conflict'] = max(old.get('conflict', 0), row.get('conflict', 0), merged.get('conflict', 0))
    return merged


def _merge_link(old, row):
    merged = dict(old)
    if old.get('seen_stop') and row.get('seen_stop') and any(
            old.get(key) != row.get(key) for key in ('child_status', 'duration_ms')):
        merged['conflict'] = 1
    for key in ('child_status', 'duration_ms'):
        if merged.get(key) is None and row.get(key) is not None:
            merged[key] = row[key]
    merged['seen_start'] = max(old.get('seen_start', 0), row.get('seen_start', 0))
    merged['seen_stop'] = max(old.get('seen_stop', 0), row.get('seen_stop', 0))
    merged['observed_at'] = min(old['observed_at'], row['observed_at'])
    merged['conflict'] = max(old.get('conflict', 0), row.get('conflict', 0), merged.get('conflict', 0))
    return merged


def store(home, row):
    """Store only sanitize() output. Duplicate projected observations are idempotent."""
    with connection(home, create=True) as db:
        if row.get('link'):
            if row['parent'] and row['child']:
                key = opaque(row['parent'] + ':' + (row['parent_turn'] or '') + ':' + row['child'])
                found = db.execute('SELECT payload FROM links WHERE key=?', (key,)).fetchone()
                if found:
                    row = _merge_link(json.loads(found[0]), row)
                db.execute('INSERT OR REPLACE INTO links VALUES (?,?)',
                           (key, json.dumps(row, sort_keys=True)))
            return
        found = db.execute('SELECT payload FROM attempts WHERE key=?', (row['key'],)).fetchone()
        if found:
            row = _merge_duplicate(json.loads(found[0]), row)
        db.execute('INSERT OR REPLACE INTO attempts VALUES (?,?)',
                   (row['key'], json.dumps(row, sort_keys=True)))


def _merge_attempt(pre, terminal, *, terminal_retry_known=True):
    if pre is None:
        result = dict(terminal)
    else:
        result = dict(terminal)
        for key, value in pre.items():
            if result.get(key) is None and value is not None:
                result[key] = value
        result['observed_at'] = min(pre['observed_at'], terminal['observed_at'])
        result['conflict'] = max(pre.get('conflict', 0), terminal.get('conflict', 0))
        result['seen_pre'] = 1
    result['seen_post'] = 1
    if not terminal_retry_known:
        result['retry_count'] = None
        result['attempt_precision'] = 'terminal_retry_count_unavailable'
    return result


def _materialize_group(events):
    source = events[0]['source']
    pres = [row for row in events if row['phase'] == 'pre']
    terminals = [row for row in events if row['phase'] != 'pre']
    if source == 'aux':
        # Auxiliary retries also reuse the logical request ID, but unlike main
        # success hooks both pre and post carry the physical retry ordinal.
        result = []
        ordinals = sorted({row['retry_count'] for row in events},
                          key=lambda ordinal: (ordinal is None, ordinal or 0))
        for ordinal in ordinals:
            terminal = next((row for row in terminals if row['retry_count'] == ordinal), None)
            pre = next((row for row in pres if row['retry_count'] == ordinal), None)
            result.append(_merge_attempt(pre, terminal) if terminal else dict(pre))
        return result

    errors = sorted((row for row in terminals if row['phase'] == 'error'),
                    key=lambda r: (r['retry_count'] is None,
                                   r['retry_count'] if r['retry_count'] is not None else 0,
                                   r['observed_at']))
    success = next((row for row in terminals if row['phase'] == 'post'), None)
    used_pre_keys = set()
    result = []
    for error in errors:
        pre = next((row for row in pres if row['retry_count'] == error['retry_count']), None)
        if pre:
            used_pre_keys.add(pre['key'])
        item = _merge_attempt(pre, error)
        item['key'] = opaque(item['request_group'] + ':error:' +
                             (str(item['retry_count']) if item['retry_count'] is not None else item['key']))
        result.append(item)
    unmatched = [row for row in pres if row['key'] not in used_pre_keys]
    if success is not None:
        # The terminal hook omits retry_count. Associate it with the latest unclosed pre only
        # to avoid a duplicate pending row, while keeping retry_count explicitly unavailable.
        pre = max(unmatched, key=lambda r: (r['retry_count'] is not None,
                                            r['retry_count'] if r['retry_count'] is not None else -1,
                                            r['observed_at'])) if unmatched else None
        if pre:
            used_pre_keys.add(pre['key'])
        item = _merge_attempt(pre, success, terminal_retry_known=False)
        item['key'] = opaque(item['request_group'] + ':terminal-success')
        result.append(item)
    for pre in pres:
        if pre['key'] not in used_pre_keys:
            item = dict(pre)
            item['key'] = opaque(item['request_group'] + ':pending:' +
                                 (str(item['retry_count']) if item['retry_count'] is not None else item['key']))
            result.append(item)
    return result


def _materialize(observations):
    groups = {}
    for row in observations:
        groups.setdefault(row['request_group'], []).append(row)
    rows = []
    for group in sorted(groups):
        rows.extend(_materialize_group(groups[group]))
    return rows


def read_snapshot(home):
    with connection(home) as db:
        observations = [json.loads(r[0]) for r in db.execute(
            'SELECT payload FROM attempts ORDER BY key')]
        links = [json.loads(r[0]) for r in db.execute('SELECT payload FROM links ORDER BY key')]
    return _materialize(observations), links
