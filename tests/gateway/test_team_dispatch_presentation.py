"""Presentation on the real buffered dispatch path, synthetic external services only."""
import asyncio
import json
import threading

import httpx
import pytest
import yaml

from tests.gateway.test_team_dispatch_runtime import (
    PROMPTS, fleet, model, ingress, message, rows, settle, tick,
)


@pytest.mark.asyncio
async def test_owner_final_uses_safe_markdown_with_one_bound_send(fleet, model):
    model.final = '**Bold** *italic* [link](https://example.invalid/a) `a_b` <tag> & !'
    await tick(fleet)
    await ingress(fleet, message(PROMPTS['engineering']))
    await settle(fleet)
    sends = [(p, ad._bot.sent) for p, (_, ad, _) in fleet.items() if ad._bot.sent]
    assert len(sends) == 1 and sends[0][0] == 'engineering'
    assert len(sends[0][1]) == 1
    sent = sends[0][1][0]
    assert sent['text'] == '*Bold* _italic_ [link](https://example.invalid/a) `a_b` <tag\\> & \\!'
    assert sent['parse_mode'] == 'MarkdownV2'
    assert (sent['chat_id'], sent['message_thread_id'], sent['reply_to_message_id']) == (-100, 7, 10)
    row = rows(fleet)[0]
    assert row['state'] == 'delivered'
    assert row['answer'] == model.final
    assert row['receipt']['bot_id'] == str(fleet['engineering'][1]._bot.id)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [None, 'timeout', 'receipt'])
async def test_long_link_plain_mode_is_chosen_before_fence_without_resend(fleet, model, monkeypatch, failure):
    model.final = '[link](https://example.invalid/' + 'a' * 2100 + ')'
    runner, adapter, _ = fleet['engineering']
    store = runner._team_dispatch.store
    original_prepare = adapter.prepare_public_handoff
    original_begin, original_send = store.begin_send, adapter._bot.send_message
    events = []

    def prepare(content):
        assert rows(fleet)[0]['send_started_at'] is None
        prepared = original_prepare(content)
        events.append('prepare')
        assert prepared == (model.final, None)
        return prepared

    def begin(request_id, answer):
        assert events == ['prepare']
        events.append('fence')
        return original_begin(request_id, answer)

    async def send(**kwargs):
        assert rows(fleet)[0]['send_started_at'] is not None
        assert events == ['prepare', 'fence']
        events.append('send')
        assert kwargs == {'chat_id': -100, 'message_thread_id': 7, 'reply_to_message_id': 10,
                          'text': model.final, 'parse_mode': None}
        receipt = await original_send(**kwargs)
        if failure == 'timeout':
            raise TimeoutError('synthetic uncertain send')
        if failure == 'receipt':
            receipt.from_user.id = 999
        return receipt

    monkeypatch.setattr(adapter, 'prepare_public_handoff', prepare)
    monkeypatch.setattr(store, 'begin_send', begin)
    monkeypatch.setattr(adapter._bot, 'send_message', send)
    await tick(fleet)
    msg = message(PROMPTS['engineering'])
    await ingress(fleet, msg)
    await settle(fleet)
    await ingress(fleet, msg)
    await settle(fleet)
    row = rows(fleet)[0]
    assert row['state'] == ('delivery_uncertain' if failure else 'delivered')
    assert row['answer'] == model.final
    assert events == ['prepare', 'fence', 'send']
    assert [(p, len(ad._bot.sent)) for p, (_, ad, _) in fleet.items() if ad._bot.sent] == [('engineering', 1)]
    if not failure:
        assert row['receipt']['bot_id'] == str(adapter._bot.id)
    assert not store.pending()


@pytest.mark.asyncio
@pytest.mark.parametrize(('selected', 'thread'), [('engineering', 7), ('revenue', None), ('default', 7)])
async def test_only_generating_owner_refreshes_typing_and_cleans_up(fleet, model, monkeypatch, selected, thread):
    from gateway import team_dispatch_agent

    entered, release = threading.Event(), threading.Event()
    actions, typing_tasks, outside_generation, during_request = [], set(), [], []
    refreshed = asyncio.Event()
    original = httpx.HTTPTransport.handle_request

    def blocked(transport, request):
        if json.loads(request.content).get('tools'):
            entered.set()
            assert release.wait(15), 'test must release synthetic model response'
        return original(transport, request)

    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', blocked)
    # Deterministic generation gate. The dispatcher brackets run_conversation with
    # a threading.Event and hands the resulting live predicate to _keep_owner_typing.
    # Record the value production itself just gated the send on: the httpx-side
    # `entered` event trails the worker thread's generating.set() by an arbitrary
    # preemption window, so reading it from the loop thread raced the first refresh.
    original_keep = team_dispatch_agent._keep_owner_typing
    gate = {}

    async def keep(dispatch, row, active):
        def gated():
            gate['generating'] = active()
            return gate['generating']
        return await original_keep(dispatch, row, gated)

    monkeypatch.setattr(team_dispatch_agent, '_keep_owner_typing', keep)
    for profile, (_, adapter, home) in fleet.items():
        if thread is None:
            adapter.config.extra['team_dispatch']['destinations'] = [{'chat_id': '-100', 'thread_id': None}]
            adapter.config.extra['allowed_topics'] = []
            config = yaml.safe_load((home / 'config.yaml').read_text())
            config['platforms']['telegram'].update(adapter.config.extra)
            (home / 'config.yaml').write_text(yaml.safe_dump(config))
        async def action(*, chat_id, action, message_thread_id=None, owner=profile):
            outside_generation.append(not gate.get('generating', False))
            during_request.append(entered.is_set() and not release.is_set())
            actions.append((owner, chat_id, action, message_thread_id))
            typing_tasks.add(asyncio.current_task())
            if len(actions) >= 2 and entered.is_set():
                refreshed.set()
        monkeypatch.setattr(adapter._bot, 'send_chat_action', action)
    await tick(fleet)
    await ingress(fleet, message(PROMPTS[selected], thread=thread))
    work = asyncio.create_task(settle(fleet))
    try:
        await asyncio.wait_for(refreshed.wait(), 10)
    finally:
        release.set()
        await work
    assert len(actions) >= 2
    assert set(actions) == {(selected, -100, 'typing', thread)}
    assert not any(outside_generation)
    assert any(during_request), 'a typing refresh must overlap the in-flight model request'
    assert all(task.done() for task in typing_tasks)
    assert rows(fleet)[0]['state'] == 'delivered'
    assert sum(len(ad._bot.sent) for _, ad, _ in fleet.values()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['format', 'length', 'timeout', 'receipt'])
async def test_formatting_precedes_fence_and_uncertain_send_is_not_retried(fleet, model, monkeypatch, failure):
    model.final = '**One answer**'
    adapter = fleet['engineering'][1]
    dispatch = fleet['engineering'][0]._team_dispatch
    original_format, original_send = adapter.format_message, adapter._bot.send_message
    formatted = []

    def prepare(content):
        assert rows(fleet)[0]['send_started_at'] is None
        formatted.append(content)
        if failure == 'format':
            raise ValueError('synthetic preparation failure')
        return original_format(content)

    async def send(**kwargs):
        assert rows(fleet)[0]['send_started_at'] is not None
        assert formatted == [model.final]
        result = await original_send(**kwargs)
        if failure == 'timeout':
            raise TimeoutError('synthetic uncertain send')
        if failure == 'receipt':
            result.from_user.id = 999
        return result

    monkeypatch.setattr(adapter, 'format_message', prepare)
    monkeypatch.setattr(adapter._bot, 'send_message', send)
    if failure == 'length':
        model.final = '🙂' * 2001
    await tick(fleet)
    msg = message(PROMPTS['engineering'])
    await ingress(fleet, msg)
    await settle(fleet)
    await ingress(fleet, msg)
    await settle(fleet)
    row = rows(fleet)[0]
    uncertain = failure in {'timeout', 'receipt'}
    assert row['state'] == ('delivery_uncertain' if uncertain else 'failed_definitive')
    assert len(adapter._bot.sent) == (1 if uncertain else 0)
    assert len(fleet['default'][1]._bot.sent) == (0 if uncertain else 1)
    assert (row['send_started_at'] is not None) is uncertain
    assert len(dispatch.store.pending()) == 0
