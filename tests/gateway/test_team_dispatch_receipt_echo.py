"""Delivery receipt acceptance for Telegram reply-thread echo in non-forum groups.

Regression for the live canary incident: a non-forum supergroup send with no
requested topic thread and a reply anchor succeeded on the transport, but
Telegram echoed the anchor's reply-thread id on the returned Message
(telegram-bot-api #798). The strict receipt validator treated the echo as an
identity mismatch, quarantining a visibly delivered message as
``delivery_uncertain``. The echo of the exact requested anchor is accepted and
audited; every other identity check is unchanged.
"""
from types import SimpleNamespace

import pytest
import yaml

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.test_team_dispatch_runtime import (
    PROMPTS, fleet, model, ingress, message, rows, settle, tick,
)


def echo_adapter(echoed_thread, *, message_id=42, chat=-100, bot=901):
    """Real adapter, fake transport returning a scripted receipt Message."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token='synthetic-only'))
    calls = []

    async def send(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(message_id=message_id, chat=SimpleNamespace(id=chat),
                               message_thread_id=echoed_thread,
                               from_user=SimpleNamespace(id=bot))

    adapter._bot = SimpleNamespace(id=901, send_message=send)
    return adapter, calls


@pytest.mark.asyncio
async def test_canary_shape_group_reply_thread_echo_is_a_valid_receipt():
    """Live canary: non-forum group, thread_id None, reply anchor echoed back."""
    adapter, calls = echo_adapter(806)
    content, mode = adapter.prepare_public_handoff('**Antwoord**')
    receipt = await adapter.send_public_handoff('-100', None, '806', content, '901', parse_mode=mode)
    assert calls == [{'chat_id': -100, 'text': '*Antwoord*', 'reply_to_message_id': 806,
                      'parse_mode': 'MarkdownV2'}]
    assert receipt == {'bot_id': '901', 'chat_id': '-100', 'thread_id': None,
                       'message_id': '42', 'reply_thread_echo': '806'}


@pytest.mark.asyncio
async def test_thread_echo_of_a_different_anchor_is_still_a_mismatch():
    adapter, _ = echo_adapter(999)
    content, mode = adapter.prepare_public_handoff('**Antwoord**')
    with pytest.raises(ValueError, match='delivery_identity_mismatch'):
        await adapter.send_public_handoff('-100', None, '806', content, '901', parse_mode=mode)


@pytest.mark.asyncio
@pytest.mark.parametrize(('requested', 'echoed'), [('7', 7), ('7', '7')])
async def test_configured_topic_thread_keeps_strict_equality_without_echo_key(requested, echoed):
    adapter, calls = echo_adapter(echoed)
    content, mode = adapter.prepare_public_handoff('**Antwoord**')
    receipt = await adapter.send_public_handoff('-100', requested, '10', content, '901', parse_mode=mode)
    assert calls == [{'chat_id': -100, 'message_thread_id': 7, 'text': '*Antwoord*',
                      'reply_to_message_id': 10, 'parse_mode': 'MarkdownV2'}]
    assert receipt == {'bot_id': '901', 'chat_id': '-100', 'thread_id': '7', 'message_id': '42'}


@pytest.mark.asyncio
async def test_forum_topic_mismatch_is_still_rejected():
    adapter, _ = echo_adapter(8)
    content, mode = adapter.prepare_public_handoff('**Antwoord**')
    with pytest.raises(ValueError, match='delivery_identity_mismatch'):
        await adapter.send_public_handoff('-100', '7', '10', content, '901', parse_mode=mode)


@pytest.mark.asyncio
@pytest.mark.parametrize(('chat', 'bot'), [(-200, 901), (-100, 902)])
async def test_chat_and_bot_identity_are_still_rejected_on_the_echo_path(chat, bot):
    adapter, _ = echo_adapter(806, chat=chat, bot=bot)
    content, mode = adapter.prepare_public_handoff('**Antwoord**')
    with pytest.raises(ValueError, match='delivery_identity_mismatch'):
        await adapter.send_public_handoff('-100', None, '806', content, '901', parse_mode=mode)


@pytest.mark.asyncio
@pytest.mark.parametrize('message_id', [None, 0, -1, '42'])
async def test_missing_or_invalid_delivery_id_is_still_rejected_on_the_echo_path(message_id):
    adapter, _ = echo_adapter(806, message_id=message_id)
    content, mode = adapter.prepare_public_handoff('**Antwoord**')
    with pytest.raises(ValueError, match='missing_delivery_id'):
        await adapter.send_public_handoff('-100', None, '806', content, '901', parse_mode=mode)


@pytest.mark.asyncio
async def test_plain_group_send_without_echo_is_unchanged():
    adapter, calls = echo_adapter(None)
    content, mode = adapter.prepare_public_handoff('**Antwoord**')
    receipt = await adapter.send_public_handoff('-100', None, '806', content, '901', parse_mode=mode)
    assert 'message_thread_id' not in calls[0]
    assert receipt == {'bot_id': '901', 'chat_id': '-100', 'thread_id': None, 'message_id': '42'}


def reconfigure_non_forum_group(fleet):
    """Mirror the live canary destination: non-forum supergroup, no topic thread."""
    for _, (runner, adapter, home) in fleet.items():
        adapter.config.extra['team_dispatch']['destinations'] = [{'chat_id': '-100', 'thread_id': None}]
        adapter.config.extra['allowed_topics'] = []
        config = yaml.safe_load((home / 'config.yaml').read_text())
        config['platforms']['telegram'].update(adapter.config.extra)
        (home / 'config.yaml').write_text(yaml.safe_dump(config))


def install_reply_thread_echo(monkeypatch, adapter):
    """Telegram echoes the reply anchor's thread id when no topic was requested."""
    original = adapter._bot.send_message

    async def send(**kwargs):
        result = await original(**kwargs)
        if 'message_thread_id' not in kwargs:
            result.message_thread_id = kwargs['reply_to_message_id']
        return result

    monkeypatch.setattr(adapter._bot, 'send_message', send)


@pytest.mark.asyncio
async def test_specialist_send_with_group_reply_thread_echo_is_delivered_not_quarantined(
        fleet, model, monkeypatch):
    reconfigure_non_forum_group(fleet)
    install_reply_thread_echo(monkeypatch, fleet['engineering'][1])
    model.final = '**Antwoord** van engineering'
    await tick(fleet)
    first = message(PROMPTS['engineering'], thread=None)
    await ingress(fleet, first)
    await settle(fleet)
    row = rows(fleet)[0]
    assert row['state'] == 'delivered'
    assert row['answer'] == model.final
    assert row['receipt']['thread_id'] is None
    assert row['receipt']['reply_thread_echo'] == str(first.message_id)
    sent = fleet['engineering'][1]._bot.sent
    assert len(sent) == 1 and 'message_thread_id' not in sent[0]
    assert sum(len(ad._bot.sent) for _, ad, _ in fleet.values()) == 1
    # A delivered echo receipt must not quarantine the public conversation.
    await ingress(fleet, message('Werk dat verder uit.', mid=12, reply=first, thread=None))
    await settle(fleet)
    assert len(rows(fleet)) == 2
    assert rows(fleet)[-1]['root_request_id'] == row['request_id']


@pytest.mark.asyncio
async def test_fallback_send_with_group_reply_thread_echo_is_delivered_not_uncertain(
        fleet, model, monkeypatch):
    import json
    reconfigure_non_forum_group(fleet)
    install_reply_thread_echo(monkeypatch, fleet['default'][1])
    model.final = 'NO_REPLY'
    await tick(fleet)
    first = message(PROMPTS['engineering'], thread=None)
    await ingress(fleet, first)
    await settle(fleet)
    await ingress(fleet, first)
    await settle(fleet)
    row = rows(fleet)[0]
    assert row['state'] == 'failed_definitive'
    fallback_receipt = json.loads(row['fallback_receipt'])
    assert fallback_receipt['state'] == 'delivered'
    assert fallback_receipt['thread_id'] is None
    assert fallback_receipt['reply_thread_echo'] == str(first.message_id)
    sent = fleet['default'][1]._bot.sent
    assert len(sent) == 1 and 'message_thread_id' not in sent[0]
    assert not fleet['engineering'][1]._bot.sent
