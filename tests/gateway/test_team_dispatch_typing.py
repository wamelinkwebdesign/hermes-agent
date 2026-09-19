"""Owner-only typing lifecycle through real dispatch/builder and native adapter.

Only the agent conversation result and Telegram network transport are synthetic.
The SQLite ledger, authorization, cancellation monitor and adapter are real.
"""
import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from telegram.error import TimedOut

from tests.gateway.test_team_dispatch_runtime import (
    PROMPTS, fleet, model, ingress, message, rows, scope, tick,
)


def probe_generation(fleet, monkeypatch, outcome):
    from run_agent import AIAgent
    loop = asyncio.get_running_loop()
    probe = SimpleNamespace(entered=asyncio.Event(), release=threading.Event(),
                            first=asyncio.Event(), action_cancelled=asyncio.Event(),
                            actions=[], tasks=set(), closed=asyncio.Event())
    close = AIAgent.close

    def conversation(agent, *args, **kwargs):
        loop.call_soon_threadsafe(probe.entered.set)
        assert probe.release.wait(20), 'test must release synthetic conversation'
        if outcome == 'generation_error':
            raise RuntimeError('synthetic generation failure')
        return {'final_response': '**Ready**', 'completed': True,
                'interrupted': agent._interrupt_requested}

    def closed(agent):
        try:
            return close(agent)
        finally:
            loop.call_soon_threadsafe(probe.closed.set)

    monkeypatch.setattr(AIAgent, 'run_conversation', conversation)
    monkeypatch.setattr(AIAgent, 'close', closed)
    for profile, (_, adapter, _) in fleet.items():
        async def action(*, chat_id, action, message_thread_id=None, owner=profile):
            probe.actions.append((owner, chat_id, action, message_thread_id))
            probe.tasks.add(asyncio.current_task())
            probe.first.set()
            if outcome == 'typing_error':
                raise TimedOut('synthetic typing failure')
            if outcome == 'typing_timeout':
                try:
                    await asyncio.Event().wait()
                finally:
                    probe.action_cancelled.set()
        monkeypatch.setattr(adapter._bot, 'send_chat_action', action)
    return probe


async def start_owner(fleet):
    await tick(fleet)
    await ingress(fleet, message(PROMPTS['engineering']))
    runner, adapter, home = fleet['default']
    with scope(home):
        await asyncio.gather(*tuple(adapter._pending_text_batch_tasks.values()))
        await runner._team_dispatch.tick()
        await asyncio.gather(*tuple(runner._team_dispatch.tasks))
    runner, adapter, home = fleet['engineering']
    with scope(home):
        await runner._team_dispatch.tick()
    assert len(runner._team_dispatch.tasks) == 1
    return next(iter(runner._team_dispatch.tasks))


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', [
    'success', 'generation_error', 'durable_cancel', 'task_cancel', 'timeout',
    'ownership_loss', 'authorization_loss', 'replacement', 'typing_error', 'typing_timeout',
])
async def test_typing_cleanup_and_final_safety(fleet, model, monkeypatch, outcome):
    from gateway import team_dispatch_agent
    probe = probe_generation(fleet, monkeypatch, outcome)
    if outcome == 'timeout':
        original = team_dispatch_agent.generate

        async def shorter(dispatch, row, *args):
            return await original(dispatch, {**row, 'expires_at': time.time() + 20}, *args)

        monkeypatch.setattr(team_dispatch_agent, 'generate', shorter)
    work = await start_owner(fleet)
    dispatch = fleet['engineering'][0]._team_dispatch
    try:
        await asyncio.wait_for(probe.first.wait(), 10)
        row = rows(fleet)[0]
        if outcome == 'durable_cancel':
            dispatch.store.cancel(row['root_request_id'], '111')
        elif outcome == 'task_cancel':
            work.cancel()
        elif outcome == 'ownership_loss':
            assert dispatch.store.route(row, 'design', 'single_domain')
            assert dispatch.store.claim(row['request_id'], expected_owner='design', expected_reason='single_domain')
        elif outcome == 'authorization_loss':
            dispatch.adapter.config.extra['allowed_topics'] = [8]
        elif outcome == 'replacement':
            dispatch.runner.adapters.clear()
        if outcome in {'durable_cancel', 'task_cancel', 'ownership_loss', 'authorization_loss', 'replacement', 'timeout'}:
            async with asyncio.timeout(10):
                while not all(task.done() for task in probe.tasks):
                    await asyncio.sleep(0.02)
        elif outcome == 'typing_timeout':
            await asyncio.wait_for(probe.action_cancelled.wait(), 5)
    finally:
        probe.release.set()
        await asyncio.gather(work, return_exceptions=True)
        await asyncio.wait_for(probe.closed.wait(), 10)
    assert set(probe.actions) == {('engineering', -100, 'typing', 7)}
    assert all(task.done() for task in probe.tasks)
    success = outcome in {'success', 'typing_error', 'typing_timeout'}
    assert len(dispatch.adapter._bot.sent) == (1 if success else 0)
    assert all(not ad._bot.sent for profile, (_, ad, _) in fleet.items() if profile != 'engineering')
    assert (rows(fleet)[0]['send_started_at'] is not None) is success
    if success:
        assert rows(fleet)[0]['state'] == 'delivered'


@pytest.mark.asyncio
@pytest.mark.parametrize('gate', ['disabled', 'paused', 'cancelled_constructor'])
async def test_no_typing_before_generation_or_when_native_indicator_disabled(fleet, model, monkeypatch, gate):
    from run_agent import AIAgent
    probe = probe_generation(fleet, monkeypatch, 'success')
    adapter = fleet['engineering'][1]
    if gate == 'disabled':
        adapter.config.typing_indicator = False
    elif gate == 'paused':
        adapter._typing_paused.add('-100')
    else:
        original = AIAgent.__init__

        def construct(agent, *args, **kwargs):
            original(agent, *args, **kwargs)
            row = rows(fleet)[0]
            fleet['engineering'][0]._team_dispatch.store.cancel(row['root_request_id'], '111')

        monkeypatch.setattr(AIAgent, '__init__', construct)
    work = await start_owner(fleet)
    try:
        if gate != 'cancelled_constructor':
            await asyncio.wait_for(probe.entered.wait(), 10)
    finally:
        probe.release.set()
        await work
    assert probe.actions == []
    assert len(adapter._bot.sent) == (0 if gate == 'cancelled_constructor' else 1)
