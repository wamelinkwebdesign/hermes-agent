"""Explicit opaque task manifests and honest observed-usage reports."""
from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
import sys
import sqlite3
from typing import Any
import uuid

from collector import BUCKETS, MODELS, PROVIDERS, label, opaque, read_snapshot

HEX = re.compile(r'^[0-9a-f]{64}$')
TASK = re.compile(r'^[0-9a-f]{32}$')
BILLED = BUCKETS[:4]
TASK_FIELDS = {'id', 'operator_asserted_completed', 'operator_asserted_lineage_complete',
               'observed_tasks', 'members'}


def require(condition):
    if not condition:
        raise ValueError('invalid or ambiguous accounting manifest/rates')


def validate(manifest: Any) -> dict[str, Any]:
    require(type(manifest) is dict and set(manifest) == {'version', 'tasks'} and manifest['version'] == 1)
    tasks = manifest['tasks']
    require(type(tasks) is list and 0 < len(tasks) <= 64)
    ids = set(); assignments = {}; observed_assignments = set()
    for task in tasks:
        require(type(task) is dict and set(task) == TASK_FIELDS)
        require(type(task['id']) is str and TASK.fullmatch(task['id']) and task['id'] not in ids)
        ids.add(task['id'])
        require(type(task['operator_asserted_completed']) is bool and
                type(task['operator_asserted_lineage_complete']) is bool)
        observed_tasks = task['observed_tasks']
        require(type(observed_tasks) is list and len(observed_tasks) <= 256 and
                all(type(item) is str and HEX.fullmatch(item) for item in observed_tasks) and
                len(set(observed_tasks)) == len(observed_tasks) and
                not observed_assignments.intersection(observed_tasks))
        observed_assignments.update(observed_tasks)
        members = task['members']
        require(type(members) is list and len(members) <= 256)
        sessions = {}
        for member in members:
            require(type(member) is dict and set(member) == {'session', 'turns', 'parent'})
            session = member['session']; turns = member['turns']; parent = member['parent']
            require(type(session) is str and HEX.fullmatch(session) and session not in sessions)
            require(parent is None or (type(parent) is str and HEX.fullmatch(parent)))
            require(turns is None or (type(turns) is list and 0 < len(turns) <= 256 and
                    all(type(t) is str and HEX.fullmatch(t) for t in turns) and len(set(turns)) == len(turns)))
            sessions[session] = parent
            for previous in assignments.get(session, []):
                require(previous is not None and turns is not None and not set(previous).intersection(turns))
            assignments.setdefault(session, []).append(turns)
        for session in sessions:
            seen = set(); current = session
            while current is not None:
                require(current in sessions and current not in seen)
                seen.add(current); current = sessions[current]
    return manifest


def matches(row, member):
    return row['session'] == member['session'] and (member['turns'] is None or row['turn'] in member['turns'])


def task_matches(row, task):
    return row.get('task') in task['observed_tasks'] or any(matches(row, member) for member in task['members'])


def expand_links(manifest, links):
    expanded = deepcopy(manifest)
    for task in expanded['tasks']:
        for _ in range(257):
            members = {m['session']: m for m in task['members']}
            added = False
            for link in links:
                parent = members.get(link['parent'])
                if parent is None or (parent['turns'] is not None and link['parent_turn'] not in parent['turns']):
                    continue
                child = link['child']
                if child in members:
                    require(members[child]['parent'] == link['parent'])
                    continue
                task['members'].append({'session': child, 'turns': None, 'parent': link['parent']})
                members[child] = task['members'][-1]; added = True
            if not added:
                break
        require(len(task['members']) <= 256)
    return validate(expanded)


def rate_table(rates: Any):
    if rates is None:
        return {}, None
    require(type(rates) is dict and set(rates) == {'version', 'revision', 'currency', 'rates'})
    require(rates['version'] == 1 and rates['currency'] == 'USD' and opaque(rates['revision']))
    require(type(rates['rates']) is list and len(rates['rates']) <= 256)
    table = {}
    for row in rates['rates']:
        require(type(row) is dict and set(row) == {'provider', 'model', *BILLED})
        key = (label(row['provider'], PROVIDERS), label(row['model'], MODELS))
        require(all(key) and key not in table)
        amounts = {}
        for bucket in BILLED:
            require(type(row[bucket]) is str and len(row[bucket]) <= 32)
            value = Decimal(row[bucket])
            require(value.is_finite() and 0 <= value <= 1000000)
            amounts[bucket] = value
        table[key] = amounts
    return table, opaque(rates['revision'])


def totals(rows, table):
    physical = [r for r in rows if r['status'] != 'aggregate_excluded']
    known = [r for r in physical if not r['conflict']]
    tokens = {k: (sum(r[k] for r in known) if physical and len(known) == len(physical) and
                   all(r[k] is not None for r in known) else None) for k in BUCKETS}
    observed = Decimal(0); priced = 0
    for row in physical:
        prices = table.get((row['provider'], row['model']))
        if prices is None or row['conflict'] or any(row[k] is None for k in BILLED):
            continue
        observed += sum((Decimal(row[k]) * prices[k] for k in BILLED), Decimal(0)) / Decimal(1000000)
        priced += 1
    return {'attempts': len(physical), 'observed_success_attempts': sum(r['status'] == 'ok' for r in physical),
            'observed_error_attempts': sum(r['status'] == 'error' for r in physical),
            'unclosed_attempts': sum(r['status'] == 'pending' for r in physical),
            'aggregate_rows_excluded': len(rows) - len(physical), 'tokens': tokens,
            'known_subtotals': {k: sum(r[k] for r in known if r[k] is not None) for k in BUCKETS},
            'missing_usage_attempts': sum(any(r[k] is None for k in BILLED) for r in physical),
            'conflicted_attempts': len(physical) - len(known), 'priced_attempts': priced,
            'observed_estimate': str(observed) if physical and priced == len(physical) else None,
            'known_priced_subtotal': str(observed) if priced else None}


def _base_flags(rows):
    flags = ['provider_internal_retries_unknown', 'observed_retry_counts_cover_hermes_hooks_only',
             'normalized_optional_field_provenance_unknown', 'process_exit_queue_loss_unknown',
             'unhooked_paths_unknown']
    if any(r.get('attempt_precision') == 'terminal_retry_count_unavailable' for r in rows):
        flags.append('terminal_attempt_retry_count_unavailable')
    if any(not r.get('request') or not r.get('turn') or not r.get('session') for r in rows):
        flags.append('missing_identity')
    if any(r.get('streaming') and r.get('source') == 'aux' for r in rows):
        flags.append('streamed_aux_usage_unavailable')
    if any(not r.get('seen_post') for r in rows):
        flags.append('unclosed_attempts')
    if any(r.get('conflict') for r in rows):
        flags.append('conflicting_duplicate_payloads')
    if any(r.get('status') == 'aggregate_excluded' for r in rows):
        flags.append('moa_outer_excluded_verify_aux_coverage')
    return flags


def _assert_unambiguous_assignments(rows, tasks):
    for row in rows:
        require(sum(task_matches(row, task) for task in tasks) <= 1)


def build_report(rows, links, manifest, task_id, rates=None):
    validate(manifest)
    expanded = expand_links(manifest, links)
    selected = [t for t in expanded['tasks'] if t['id'] == task_id]
    require(len(selected) == 1)
    task = selected[0]; table, revision = rate_table(rates)
    _assert_unambiguous_assignments(rows, expanded['tasks'])
    chosen = [row for row in rows if task_matches(row, task)]
    parents = [row for row in chosen if row.get('task') in task['observed_tasks'] or
               any(member['parent'] is None and matches(row, member) for member in task['members'])]
    attributed = {row['key'] for row in rows if any(task_matches(row, item) for item in expanded['tasks'])}
    flags = _base_flags(chosen)
    if not task['operator_asserted_lineage_complete']:
        flags.append('lineage_not_operator_asserted_complete')
    if not task['operator_asserted_completed']:
        flags.append('task_not_operator_asserted_completed')
    if not chosen:
        flags.append('no_observed_attempts')
    return {'version': 1, 'task': task_id,
            'operator_asserted_completed': task['operator_asserted_completed'],
            'operator_asserted_lineage_complete': task['operator_asserted_lineage_complete'],
            'parent_only': totals(parents, table), 'descendant_inclusive': totals(chosen, table),
            'unattributed_attempts': sum(row['key'] not in attributed and
                                         row['status'] != 'aggregate_excluded' for row in rows),
            'coverage_flags': flags, 'complete_task_cost': None, 'rates_revision_hash': revision,
            'cost_basis': 'user_supplied_estimate_not_invoice',
            'unsupported_paths': ['SDK-internal retries', 'unhooked Codex app-server/provider paths',
                                  'auxiliary stream consumption usage', 'cross-profile automatic joins']}


def _time_summary(rows):
    def values(name):
        return [row[name] for row in rows if row.get(name) is not None]
    observed = values('observed_at'); started = values('started_at'); ended = values('ended_at')
    return {'first_observed_at': min(observed) if observed else None,
            'last_observed_at': max(observed) if observed else None,
            'earliest_hook_started_at': min(started) if started else None,
            'latest_hook_ended_at': max(ended) if ended else None,
            'timing_note': ('main timings are cumulative logical-request values; auxiliary timings '
                            'describe one physical auxiliary attempt; neither is summed here')}


def build_inventory(rows, links):
    groups = []
    for task_hash in sorted({row.get('task') for row in rows if row.get('task')}):
        chosen = [row for row in rows if row.get('task') == task_hash]
        groups.append({'observed_task': task_hash, 'attempts': len(chosen),
                       'sessions': sorted({row['session'] for row in chosen if row.get('session')}),
                       'turns': sorted({row['turn'] for row in chosen if row.get('turn')}),
                       'request_groups': len({row['request_group'] for row in chosen}),
                       **_time_summary(chosen)})
    return {'attempts': rows, 'links': links, 'observed_tasks': groups}


def build_observed_task_report(rows, links, task_hash, rates=None):
    require(type(task_hash) is str and HEX.fullmatch(task_hash))
    chosen = [row for row in rows if row.get('task') == task_hash]
    require(bool(chosen))
    table, revision = rate_table(rates)
    sessions = {row['session'] for row in chosen if row.get('session')}
    parent_turns = {(row['session'], row['turn']) for row in chosen
                    if row.get('session') and row.get('turn')}
    related_links = [link for link in links
                     if (link.get('parent'), link.get('parent_turn')) in parent_turns]
    flags = _base_flags(chosen)
    flags.extend(['explicit_operator_grouping_required_for_multi_run_goal',
                  'cross_profile_automatic_join_unsupported'])
    return {'version': 1, 'observed_task': task_hash, 'observed_usage': totals(chosen, table),
            'sessions': sorted(sessions), 'turns': sorted({row['turn'] for row in chosen if row.get('turn')}),
            'observed_child_links': len(related_links), 'time': _time_summary(chosen),
            'coverage_flags': flags, 'complete_task_cost': None,
            'rates_revision_hash': revision, 'cost_basis': 'user_supplied_estimate_not_invoice'}


def load_json(path):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'r', encoding='utf-8') as stream:
        value = stream.read(1000001)
    require(len(value) <= 1000000)
    return json.loads(value)


def save_manifest(path, value):
    validate(value)
    path = Path(path)
    require(not path.is_symlink() and not any(p.is_symlink() for p in path.parents))
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, indent=2); stream.write('\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('identity', help='Hash one raw identifier from stdin, never print it')
    init = sub.add_parser('init'); init.add_argument('--manifest', required=True)
    complete = sub.add_parser('complete'); complete.add_argument('--home', required=True)
    complete.add_argument('--manifest', required=True); complete.add_argument('--task', required=True)
    complete.add_argument('--assert-completed', required=True, action='store_true')
    inventory = sub.add_parser('inventory'); inventory.add_argument('--home', required=True)
    observed = sub.add_parser('observed-task'); observed.add_argument('observed_task', nargs='?')
    observed.add_argument('--home', required=True); observed.add_argument('--task', dest='observed_task_option')
    observed.add_argument('--rates')
    view = sub.add_parser('report'); view.add_argument('--home', required=True)
    view.add_argument('--manifest', required=True); view.add_argument('--task', required=True); view.add_argument('--rates')
    args = parser.parse_args(argv)
    try:
        if args.command == 'identity':
            value = opaque(sys.stdin.read(4097).rstrip('\n')); require(value is not None); print(value)
        elif args.command == 'init':
            require(not Path(args.manifest).exists())
            task = uuid.uuid4().hex
            save_manifest(args.manifest, {'version': 1, 'tasks': [{'id': task, 'members': [],
                          'observed_tasks': [], 'operator_asserted_completed': False,
                          'operator_asserted_lineage_complete': False}]}); print(task)
        elif args.command == 'complete':
            manifest = validate(load_json(args.manifest))
            tasks = [task for task in manifest['tasks'] if task['id'] == args.task]; require(len(tasks) == 1)
            rows, links = read_snapshot(args.home)
            expanded = expand_links(manifest, links)
            _assert_unambiguous_assignments(rows, expanded['tasks'])
            task = next(item for item in expanded['tasks'] if item['id'] == args.task)
            chosen = [row for row in rows if task_matches(row, task)]
            require(chosen and all(row.get('seen_post') and not row.get('conflict') for row in chosen))
            tasks[0]['operator_asserted_completed'] = True; save_manifest(args.manifest, manifest)
        elif args.command == 'inventory':
            rows, links = read_snapshot(args.home)
            print(json.dumps(build_inventory(rows, links), sort_keys=True, indent=2))
        elif args.command == 'observed-task':
            rows, links = read_snapshot(args.home)
            task_hash = args.observed_task or args.observed_task_option
            require(not (args.observed_task and args.observed_task_option))
            print(json.dumps(build_observed_task_report(rows, links, task_hash,
                  load_json(args.rates) if args.rates else None), sort_keys=True, indent=2))
        else:
            rows, links = read_snapshot(args.home)
            print(json.dumps(build_report(rows, links, load_json(args.manifest), args.task,
                  load_json(args.rates) if args.rates else None), sort_keys=True, indent=2))
        return 0
    except (ValueError, OSError, sqlite3.Error, InvalidOperation, KeyError, TypeError, StopIteration):
        print('Accounting input/storage unavailable or invalid; no complete result.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
