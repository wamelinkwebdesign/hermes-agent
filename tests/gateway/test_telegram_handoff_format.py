"""Real handoff preparation and send path, no Telegram connection or rich retries."""
from types import SimpleNamespace

import pytest
from telegram.error import BadRequest, Forbidden, TimedOut

from gateway.config import PlatformConfig
from gateway.platforms.base import utf16_len
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture
def transport():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token='synthetic-only'))
    calls = []

    async def send(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(message_id=42, chat=SimpleNamespace(id=kwargs['chat_id']),
                               message_thread_id=kwargs.get('message_thread_id'),
                               from_user=SimpleNamespace(id=901))

    adapter._bot = SimpleNamespace(id=901, send_message=send)
    return adapter, calls


@pytest.mark.asyncio
@pytest.mark.parametrize(('raw', 'expected'), [
    ('ordinary text', 'ordinary text'),
    ('**bold** *italic*', '*bold* _italic_'),
    ('[a_b](https://example.invalid/a_(b)?x=1&y=2)',
     '[a\\_b](https://example.invalid/a_\\(b\\)?x=1&y=2)'),
    ('`a_b`\n```python\nx = "a\\b"\n```', '`a_b`\n```python\nx = "a\\\\b"\n```'),
    ('🙂 **café** e\u0301', '🙂 *café* e\u0301'),
    ('unclosed **bold and `code [oops](broken',
     'unclosed \\*\\*bold and \\`code \\[oops\\]\\(broken'),
    ('# + - = | { } . ( ) > < ! _ \\',
     '*\\+ \\- \\= \\| \\{ \\} \\. \\( \\) \\> < \\! \\_ \\\\*'),
    ('**bold *nested***', '*bold \\*nested*\\*'),
])
@pytest.mark.parametrize('thread', [None, '7'])
async def test_formatting_and_exact_destination_survive_single_transport(transport, raw, expected, thread):
    adapter, calls = transport
    content, mode = adapter.prepare_public_handoff(raw)
    receipt = await adapter.send_public_handoff('-100', thread, '10', content, '901', parse_mode=mode)
    assert calls == [{'chat_id': -100, 'text': expected, 'reply_to_message_id': 10,
                      'parse_mode': 'MarkdownV2', **({'message_thread_id': 7} if thread else {})}]
    assert receipt == {'bot_id': '901', 'chat_id': '-100', 'thread_id': thread, 'message_id': '42'}


@pytest.mark.asyncio
@pytest.mark.parametrize(('raw', 'plain'), [('🙂' * 2000, False), ('!' * 4000, True)])
async def test_utf16_boundary_and_prepared_plain_fallback(transport, raw, plain):
    adapter, calls = transport
    content, mode = adapter.prepare_public_handoff(raw)
    assert utf16_len(content) <= adapter.MAX_MESSAGE_LENGTH
    assert (mode is None) is plain
    assert content == raw
    await adapter.send_public_handoff('-100', None, '10', content, '901', parse_mode=mode)
    assert len(calls) == 1
    assert calls[0].get('parse_mode') == (None if plain else 'MarkdownV2')
    # Explicit None overrides any transport-level formatting default.
    assert 'parse_mode' in calls[0]


@pytest.mark.parametrize('raw', ['', '   ', '🙂' * 2001, 'a' * 4001])
def test_invalid_lengths_fail_before_transport(transport, raw):
    adapter, calls = transport
    with pytest.raises(ValueError, match='invalid_handoff_length'):
        adapter.prepare_public_handoff(raw)
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(('url_length', 'suffix', 'plain'), [
    (40, '', False),
    (1998, '', False),
    (1999, '', True),
    (2000, '', True),
    (2123, '', True),
    (1996, '(x)', True),
    (1998, '🙂', True),
    (1997, '\\', True),
])
async def test_long_link_preparation_preserves_text_and_one_bound_send(transport, url_length, suffix, plain):
    adapter, calls = transport
    prefix = 'https://example.invalid/'
    url = prefix + 'a' * (url_length - len(prefix)) + suffix
    raw = '**Read** [link](' + url + ') and [short](https://example.invalid/)'
    assert utf16_len(raw) <= 4000
    content, mode = adapter.prepare_public_handoff(raw)
    assert (mode is None) is plain
    if plain:
        assert content == raw
    else:
        assert content == '*Read* [link](' + url + ') and [short](https://example.invalid/)'
    receipt = await adapter.send_public_handoff('-100', '7', '10', content, '901', parse_mode=mode)
    assert calls == [{'chat_id': -100, 'message_thread_id': 7, 'reply_to_message_id': 10,
                      'text': content, 'parse_mode': None if plain else 'MarkdownV2'}]
    assert receipt == {'bot_id': '901', 'chat_id': '-100', 'thread_id': '7', 'message_id': '42'}


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['timeout', 'rejection', 'forbidden', 'chat', 'thread', 'bot', 'id'])
async def test_prepared_send_never_retries_or_changes_binding(transport, monkeypatch, failure):
    adapter, calls = transport
    original = adapter._bot.send_message

    async def send(**kwargs):
        receipt = await original(**kwargs)
        if failure == 'timeout':
            raise TimedOut('synthetic uncertainty')
        if failure == 'rejection':
            raise BadRequest('synthetic rejection')
        if failure == 'forbidden':
            raise Forbidden('synthetic rejection')
        if failure == 'chat':
            receipt.chat.id = -200
        if failure == 'thread':
            receipt.message_thread_id = 8
        if failure == 'bot':
            receipt.from_user.id = 902
        if failure == 'id':
            receipt.message_id = None
        return receipt

    monkeypatch.setattr(adapter._bot, 'send_message', send)
    content, mode = adapter.prepare_public_handoff('**bold**')
    with pytest.raises((ValueError, TimedOut, BadRequest, Forbidden)):
        await adapter.send_public_handoff('-100', '7', '10', content, '901', parse_mode=mode)
    assert len(calls) == 1
    assert calls[0] == {'chat_id': -100, 'message_thread_id': 7, 'reply_to_message_id': 10,
                        'text': '*bold*', 'parse_mode': 'MarkdownV2'}


@pytest.mark.asyncio
async def test_changed_transport_identity_never_sends(transport):
    adapter, calls = transport
    content, mode = adapter.prepare_public_handoff('**bold**')
    with pytest.raises(ValueError, match='transport_identity_changed'):
        await adapter.send_public_handoff('-100', '7', '10', content, '902', parse_mode=mode)
    assert calls == []
