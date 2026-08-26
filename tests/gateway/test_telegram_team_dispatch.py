"""Central Telegram team ingress and fresh target-event construction tests."""

from __future__ import annotations

import asyncio
import dataclasses
import time
import weakref
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from gateway.telegram_team_routing import RootOwnership
from plugins.platforms.telegram.adapter import TelegramAdapter


_ALLOWED_CHAT = "-1001234567890"
_USERNAMES = {
    "default": "Ace_Bot",
    "engineering": "Woz_Bot",
    "design": "Virgil_Bot",
}


def _adapter(profile: str) -> TelegramAdapter:
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(
        enabled=True,
        token=f"token-{profile}",
        extra={},
    )
    adapter._team_route_gate = None
    adapter._team_ingress_handler = None
    adapter._bot_username_observed = _USERNAMES[profile].lower()
    adapter._bot = SimpleNamespace(
        id={"default": 10, "engineering": 20, "design": 30}[profile],
        username=_USERNAMES[profile],
    )
    adapter._bot_identity_checked_at = time.monotonic()
    adapter.set_owner_profile(profile)
    return adapter


def _runner() -> tuple[GatewayRunner, dict[str, TelegramAdapter]]:
    config = GatewayConfig(
        multiplex_profiles=True,
        multiplex_profile_allowlist=["engineering", "design"],
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="token-default",
                extra={
                    "team_routing": {
                        "coordinator_profile": "default",
                        "coordinator_username": "Ace_Bot",
                        "members": dict(_USERNAMES),
                        "allowed_chats": [_ALLOWED_CHAT],
                    }
                },
            )
        },
    )
    roster = {profile: _adapter(profile) for profile in _USERNAMES}
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {Platform.TELEGRAM: roster["default"]}
    runner._profile_adapters = {
        profile: {Platform.TELEGRAM: roster[profile]}
        for profile in ("engineering", "design")
    }
    runner._active_profile_name = lambda: "default"
    for adapter in roster.values():
        adapter.gateway_runner = runner
    assert runner._validate_and_install_telegram_team_runtime() is True
    return runner, roster


def _prepare_batching(adapter: TelegramAdapter) -> None:
    adapter._drop_delayed_deliveries = False
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._pending_text_batch_id_index = {}
    adapter._pending_photo_batches = {}
    adapter._pending_photo_batch_tasks = {}
    adapter._pending_photo_batch_id_index = {}
    adapter._media_group_events = {}
    adapter._media_group_tasks = {}
    adapter._pending_media_group_batch_id_index = {}
    adapter._held_inbound_events = []
    adapter._held_inbound_redispatch_task = None
    adapter.HELD_INBOUND_MAX = 64
    adapter._text_batch_delay_seconds = 3600
    adapter._text_batch_split_delay_seconds = 3600
    adapter._TEXT_BATCH_FAST_DELAY_S = 3600
    adapter._TEXT_BATCH_SHORT_DELAY_S = 3600
    adapter._media_batch_delay_seconds = 3600
    adapter.MEDIA_GROUP_WAIT_SECONDS = 3600


async def _cancel_batch_tasks(*tasks) -> None:
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _enqueue_batch_event(
    adapter: TelegramAdapter,
    batch_kind: str,
    event: MessageEvent,
) -> None:
    if batch_kind in {"text", "command"}:
        adapter._enqueue_text_event(event)
    elif batch_kind == "photo":
        adapter._enqueue_photo_event("overflow-photo", event)
    else:
        await adapter._queue_media_group_event("overflow-album", event)


def _pending_batch_state(adapter: TelegramAdapter, batch_kind: str):
    if batch_kind in {"text", "command"}:
        return adapter._pending_text_batches, adapter._pending_text_batch_tasks
    if batch_kind == "photo":
        return adapter._pending_photo_batches, adapter._pending_photo_batch_tasks
    return adapter._media_group_events, adapter._media_group_tasks


async def _flush_pending_batch(
    adapter: TelegramAdapter,
    batch_kind: str,
    batch_key: str,
) -> None:
    if batch_kind in {"text", "command"}:
        adapter._text_batch_delay_seconds = 0
        adapter._TEXT_BATCH_FAST_DELAY_S = 0
        await adapter._flush_text_batch(batch_key)
    elif batch_kind == "photo":
        adapter._media_batch_delay_seconds = 0
        await adapter._flush_photo_batch(batch_key)
    else:
        adapter.MEDIA_GROUP_WAIT_SECONDS = 0
        await adapter._flush_media_group_event(batch_key)


def _raw_message(
    message_id: int,
    *,
    text: str = "hello",
    chat_id: int = int(_ALLOWED_CHAT),
    chat_type: str = "supergroup",
    reply_to_message_id: int | None = None,
    reply_author_username: str = "Human_User",
) -> SimpleNamespace:
    reply = None
    if reply_to_message_id is not None:
        reply = SimpleNamespace(
            message_id=reply_to_message_id,
            from_user=SimpleNamespace(
                id=999,
                username=reply_author_username,
                full_name="Reply Author",
            ),
            text="quoted historical text with @Ace_Bot",
            caption=None,
        )
    return SimpleNamespace(
        message_id=message_id,
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=77,
        is_topic_message=True,
        chat=SimpleNamespace(
            id=chat_id,
            type=chat_type,
            title="Team Room",
            full_name=None,
            is_forum=True,
        ),
        from_user=SimpleNamespace(
            id=123,
            username="Alice_User",
            full_name="Alice Example",
            is_bot=False,
        ),
        reply_to_message=reply,
        date=datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc),
    )


def _event(
    adapter: TelegramAdapter,
    message_id: int,
    *,
    text: str = "hello",
    chat_id: int = int(_ALLOWED_CHAT),
    chat_type: str = "group",
    raw_chat_type: str = "supergroup",
    reply_to_message_id: int | None = None,
    reply_author_username: str = "Human_User",
    message_type: MessageType = MessageType.TEXT,
    metadata: dict | None = None,
) -> MessageEvent:
    profile = adapter._owner_profile
    raw = _raw_message(
        message_id,
        text=text,
        chat_id=chat_id,
        chat_type=raw_chat_type,
        reply_to_message_id=reply_to_message_id,
        reply_author_username=reply_author_username,
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=str(chat_id),
        chat_name="Team Room",
        chat_type=chat_type,
        user_id="123",
        user_name="Alice Example",
        thread_id="77",
        chat_topic="Planning",
        user_id_alt="alice-alt",
        message_id=str(message_id),
        profile=profile,
    )
    source._transport_adapter_ref = weakref.ref(adapter)
    return MessageEvent(
        text=text,
        message_type=message_type,
        user_id="123",
        user_name="Alice Example",
        source=source,
        raw_message=raw,
        message_id=str(message_id),
        platform_update_id=message_id + 1000,
        media_urls=["/tmp/example.png"],
        media_types=["image/png"],
        reply_to_message_id=(
            str(reply_to_message_id) if reply_to_message_id is not None else None
        ),
        reply_to_text=("quoted historical text" if reply_to_message_id else None),
        reply_to_author_id=("999" if reply_to_message_id else None),
        reply_to_author_name=("Reply Author" if reply_to_message_id else None),
        reply_to_is_own_message=False,
        auto_skill="team-skill",
        channel_prompt="Team channel prompt",
        channel_context="Earlier safe channel context",
        metadata=dict(metadata or {"existing": "value"}),
        timestamp=raw.date,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", list(MessageType))
async def test_every_normalized_telegram_event_type_crosses_team_ingress_seam(
    monkeypatch, message_type
):
    adapter = _adapter("default")
    handler = AsyncMock(return_value=True)
    adapter.set_team_ingress_handler(handler)
    base_handle = AsyncMock()
    monkeypatch.setattr(BasePlatformAdapter, "handle_message", base_handle)
    event = _event(adapter, 40, message_type=message_type)

    await adapter.handle_message(event)

    handler.assert_awaited_once_with(adapter, event)
    base_handle.assert_not_awaited()


@pytest.mark.asyncio
async def test_absent_team_ingress_handler_preserves_legacy_base_handling(monkeypatch):
    adapter = _adapter("default")
    base_handle = AsyncMock()
    monkeypatch.setattr(BasePlatformAdapter, "handle_message", base_handle)
    event = _event(adapter, 41)

    await adapter.handle_message(event)

    base_handle.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_team_ingress_handler_exception_fails_closed(monkeypatch):
    adapter = _adapter("default")
    handler = AsyncMock(side_effect=RuntimeError("boom"))
    adapter.set_team_ingress_handler(handler)
    base_handle = AsyncMock()
    monkeypatch.setattr(BasePlatformAdapter, "handle_message", base_handle)

    await adapter.handle_message(_event(adapter, 42))

    base_handle.assert_not_awaited()


@pytest.mark.asyncio
async def test_pending_reservation_consumes_duplicate_then_commit_consumes_replay(
    monkeypatch,
):
    runner, roster = _runner()
    adapter = roster["engineering"]
    dispatcher = runner._telegram_team_dispatcher
    assert dispatcher is not None
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _blocking_base(_self, _event):
        entered.set()
        await release.wait()

    monkeypatch.setattr(BasePlatformAdapter, "handle_message", _blocking_base)
    commit = Mock(wraps=dispatcher.commit_ingress)
    dispatcher.commit_ingress = commit
    first = _event(adapter, 90, text="@Woz_Bot investigate")

    first_task = asyncio.create_task(adapter.handle_message(first))
    await entered.wait()
    await adapter.handle_message(
        _event(adapter, 90, text="@Woz_Bot duplicate while pending")
    )
    commit.assert_not_called()

    release.set()
    await first_task
    commit.assert_called_once()
    await adapter.handle_message(_event(adapter, 90, text="@Woz_Bot replay"))
    commit.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "cancellation"])
async def test_base_submission_failure_releases_reservation_and_retry_dispatches(
    monkeypatch,
    failure,
):
    runner, roster = _runner()
    adapter = roster["engineering"]
    dispatcher = runner._telegram_team_dispatcher
    assert dispatcher is not None
    calls = 0

    async def _base(_self, _event):
        nonlocal calls
        calls += 1
        if calls == 1:
            if failure == "cancellation":
                raise asyncio.CancelledError
            raise RuntimeError("base rejected submission")

    monkeypatch.setattr(BasePlatformAdapter, "handle_message", _base)
    release = Mock(wraps=dispatcher.release_ingress)
    commit = Mock(wraps=dispatcher.commit_ingress)
    dispatcher.release_ingress = release
    dispatcher.commit_ingress = commit

    first = _event(adapter, 91, text="@Woz_Bot investigate")
    expected_error = asyncio.CancelledError if failure == "cancellation" else RuntimeError
    with pytest.raises(expected_error):
        await adapter.handle_message(first)

    release.assert_called_once()
    commit.assert_not_called()
    # Root/alias preparation is intentionally idempotent and may remain after
    # Base rejects submission; the released reservation must not suppress retry.
    assert dispatcher.resolve_reply_owner(_ALLOWED_CHAT, 91) == RootOwnership(
        "91",
        "engineering",
    )

    retry = _event(adapter, 91, text="@Woz_Bot investigate again")
    await adapter.handle_message(retry)
    assert calls == 2
    commit.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("callback_failure", "expected_error"),
    [
        ("exception", None),
        ("cancellation", asyncio.CancelledError),
        ("invalid-decision", None),
    ],
)
async def test_ingress_callback_failure_after_attachment_releases_before_consuming(
    monkeypatch,
    callback_failure,
    expected_error,
):
    runner, roster = _runner()
    adapter = roster["engineering"]
    dispatcher = runner._telegram_team_dispatcher
    assert dispatcher is not None
    original_handler = adapter._team_ingress_handler
    assert original_handler is not None
    base_handle = AsyncMock()
    monkeypatch.setattr(BasePlatformAdapter, "handle_message", base_handle)
    release = Mock(wraps=dispatcher.release_ingress)
    dispatcher.release_ingress = release

    async def _failing_handler(current_adapter, event):
        consumed = await original_handler(current_adapter, event)
        assert consumed is False
        assert getattr(event, "_telegram_team_ingress_reservation", None) is not None
        if callback_failure == "cancellation":
            raise asyncio.CancelledError
        if callback_failure == "exception":
            raise RuntimeError("callback failed after reservation")
        return "invalid"

    adapter.set_team_ingress_handler(_failing_handler)  # type: ignore[arg-type]
    first = _event(adapter, 92, text="@Woz_Bot investigate")
    if expected_error is None:
        await adapter.handle_message(first)
    else:
        with pytest.raises(expected_error):
            await adapter.handle_message(first)

    release.assert_called_once()
    base_handle.assert_not_awaited()
    adapter.set_team_ingress_handler(original_handler)
    await adapter.handle_message(_event(adapter, 92, text="@Woz_Bot retry"))
    base_handle.assert_awaited_once()


@pytest.mark.asyncio
async def test_duplicate_replay_is_consumed_before_base_session_handling(monkeypatch):
    runner, roster = _runner()
    adapter = roster["engineering"]
    base_handle = AsyncMock()
    monkeypatch.setattr(BasePlatformAdapter, "handle_message", base_handle)
    first = _event(adapter, 100, text="@Woz_Bot investigate")
    replay = _event(adapter, 100, text="@Woz_Bot investigate")

    await adapter.handle_message(first)
    await adapter.handle_message(replay)

    base_handle.assert_awaited_once_with(first)
    assert first.metadata["telegram_team_ingress_claimed"] is True
    assert first.metadata["telegram_team_root_message_id"] == "100"
    assert runner._telegram_team_dispatcher.resolve_reply_owner(
        _ALLOWED_CHAT, "100"
    ) == RootOwnership("100", "engineering")


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_kind", ["text", "photo", "media-group"])
async def test_real_telegram_batch_paths_reserve_and_bind_every_constituent_id(
    monkeypatch,
    batch_kind,
):
    runner, roster = _runner()
    adapter = roster["engineering"]
    _prepare_batching(adapter)
    base_handle = AsyncMock()
    monkeypatch.setattr(BasePlatformAdapter, "handle_message", base_handle)
    message_type = MessageType.TEXT if batch_kind == "text" else MessageType.PHOTO
    first = _event(
        adapter,
        900,
        text="@Woz_Bot first",
        message_type=message_type,
    )
    second = _event(
        adapter,
        901,
        text="@Woz_Bot second",
        message_type=message_type,
    )

    scheduled = []
    if batch_kind == "text":
        adapter._enqueue_text_event(first)
        scheduled.append(next(iter(adapter._pending_text_batch_tasks.values())))
        adapter._enqueue_text_event(second)
        scheduled.append(next(iter(adapter._pending_text_batch_tasks.values())))
        await _cancel_batch_tasks(*scheduled)
        batch_key = next(iter(adapter._pending_text_batches))
        adapter._pending_text_batch_tasks.clear()
        adapter._text_batch_delay_seconds = 0
        adapter._TEXT_BATCH_FAST_DELAY_S = 0
        await adapter._flush_text_batch(batch_key)
    elif batch_kind == "photo":
        adapter._enqueue_photo_event("photo-burst", first)
        scheduled.append(next(iter(adapter._pending_photo_batch_tasks.values())))
        adapter._enqueue_photo_event("photo-burst", second)
        scheduled.append(next(iter(adapter._pending_photo_batch_tasks.values())))
        await _cancel_batch_tasks(*scheduled)
        batch_key = next(iter(adapter._pending_photo_batches))
        adapter._pending_photo_batch_tasks.clear()
        adapter._media_batch_delay_seconds = 0
        await adapter._flush_photo_batch(batch_key)
    else:
        await adapter._queue_media_group_event("album-1", first)
        scheduled.append(next(iter(adapter._media_group_tasks.values())))
        await adapter._queue_media_group_event("album-1", second)
        scheduled.append(next(iter(adapter._media_group_tasks.values())))
        await _cancel_batch_tasks(*scheduled)
        batch_key = next(iter(adapter._media_group_events))
        adapter._media_group_tasks.clear()
        adapter.MEDIA_GROUP_WAIT_SECONDS = 0
        await adapter._flush_media_group_event(batch_key)

    base_handle.assert_awaited_once()
    dispatched = base_handle.await_args_list[0].args[0]
    assert dispatched is first
    assert dispatched.raw_message.message_id == 900
    assert dispatched.message_id == "900"
    assert dispatched.source.message_id == "900"
    assert getattr(dispatched, "_telegram_batch_message_ids") == ("900", "901")
    assert "telegram_batch_message_ids" not in dispatched.metadata
    assert "telegram_batch_message_ids" not in dispatched.source.to_dict()
    expected = RootOwnership("900", "engineering")
    dispatcher = runner._telegram_team_dispatcher
    assert dispatcher is not None
    assert dispatcher.resolve_reply_owner(
        _ALLOWED_CHAT, 900
    ) == expected
    assert dispatcher.resolve_reply_owner(
        _ALLOWED_CHAT, 901
    ) == expected

    await adapter.handle_message(
        _event(
            adapter,
            901,
            text="@Woz_Bot duplicate constituent",
            message_type=message_type,
        )
    )
    base_handle.assert_awaited_once()


@pytest.mark.asyncio
async def test_known_parent_batch_maps_every_constituent_to_parent_root(monkeypatch):
    runner, roster = _runner()
    adapter = roster["engineering"]
    dispatcher = runner._telegram_team_dispatcher
    assert dispatcher is not None
    assert dispatcher.record_root(_ALLOWED_CHAT, 850, "engineering") is True
    _prepare_batching(adapter)
    base_handle = AsyncMock()
    monkeypatch.setattr(BasePlatformAdapter, "handle_message", base_handle)
    first = _event(adapter, 900, reply_to_message_id=850)
    second = _event(adapter, 901, reply_to_message_id=850)

    adapter._enqueue_text_event(first)
    task_1 = next(iter(adapter._pending_text_batch_tasks.values()))
    adapter._enqueue_text_event(second)
    task_2 = next(iter(adapter._pending_text_batch_tasks.values()))
    await _cancel_batch_tasks(task_1, task_2)
    batch_key = next(iter(adapter._pending_text_batches))
    adapter._pending_text_batch_tasks.clear()
    adapter._text_batch_delay_seconds = 0
    adapter._TEXT_BATCH_FAST_DELAY_S = 0
    await adapter._flush_text_batch(batch_key)

    expected = RootOwnership("850", "engineering")
    assert dispatcher.resolve_reply_owner(_ALLOWED_CHAT, 900) == expected
    assert dispatcher.resolve_reply_owner(_ALLOWED_CHAT, 901) == expected
    base_handle.assert_awaited_once_with(first)


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_kind", ["text", "photo", "media-group"])
async def test_team_batch_paths_keep_different_immediate_parents_in_separate_roots(
    monkeypatch,
    batch_kind,
):
    runner, roster = _runner()
    adapter = roster["engineering"]
    dispatcher = runner._telegram_team_dispatcher
    assert dispatcher is not None
    assert dispatcher.record_root(_ALLOWED_CHAT, 850, "engineering") is True
    assert dispatcher.record_root(_ALLOWED_CHAT, 999, "engineering") is True
    _prepare_batching(adapter)
    base_handle = AsyncMock()
    monkeypatch.setattr(BasePlatformAdapter, "handle_message", base_handle)
    message_type = MessageType.TEXT if batch_kind == "text" else MessageType.PHOTO
    events = [
        _event(
            adapter,
            900,
            text="first for 850",
            reply_to_message_id=850,
            message_type=message_type,
        ),
        _event(
            adapter,
            901,
            text="only for 999",
            reply_to_message_id=999,
            message_type=message_type,
        ),
        _event(
            adapter,
            902,
            text="second for 850",
            reply_to_message_id=850,
            message_type=message_type,
        ),
    ]
    events[0].media_urls = ["/tmp/900.png"]
    events[1].media_urls = ["/tmp/901.png"]
    events[2].media_urls = ["/tmp/902.png"]

    if batch_kind == "text":
        for event in events:
            adapter._enqueue_text_event(event)
        pending = adapter._pending_text_batches
        tasks = adapter._pending_text_batch_tasks
    elif batch_kind == "photo":
        for event in events:
            adapter._enqueue_photo_event("photo-burst", event)
        pending = adapter._pending_photo_batches
        tasks = adapter._pending_photo_batch_tasks
    else:
        for event in events:
            await adapter._queue_media_group_event("same-album", event)
        pending = adapter._media_group_events
        tasks = adapter._media_group_tasks

    await _cancel_batch_tasks(*list(tasks.values()))
    tasks.clear()
    assert len(pending) == 2
    batches_by_first_id = {
        getattr(event, "_telegram_batch_message_ids")[0]: event
        for event in pending.values()
    }
    assert getattr(
        batches_by_first_id["900"], "_telegram_batch_message_ids"
    ) == ("900", "902")
    assert getattr(
        batches_by_first_id["901"], "_telegram_batch_message_ids"
    ) == ("901",)

    if batch_kind == "text":
        adapter._text_batch_delay_seconds = 0
        adapter._TEXT_BATCH_FAST_DELAY_S = 0
        for key in list(pending):
            await adapter._flush_text_batch(key)
    elif batch_kind == "photo":
        adapter._media_batch_delay_seconds = 0
        for key in list(pending):
            await adapter._flush_photo_batch(key)
    else:
        adapter.MEDIA_GROUP_WAIT_SECONDS = 0
        for key in list(pending):
            await adapter._flush_media_group_event(key)

    assert base_handle.await_count == 2
    assert dispatcher.resolve_reply_owner(_ALLOWED_CHAT, 900) == RootOwnership(
        "850", "engineering"
    )
    assert dispatcher.resolve_reply_owner(_ALLOWED_CHAT, 902) == RootOwnership(
        "850", "engineering"
    )
    assert dispatcher.resolve_reply_owner(_ALLOWED_CHAT, 901) == RootOwnership(
        "999", "engineering"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_kind", ["text", "photo", "media-group"])
async def test_buffered_duplicate_replay_does_not_duplicate_payload_or_dispatch(
    monkeypatch,
    batch_kind,
):
    _, roster = _runner()
    adapter = roster["engineering"]
    _prepare_batching(adapter)
    base_handle = AsyncMock()
    monkeypatch.setattr(BasePlatformAdapter, "handle_message", base_handle)
    message_type = MessageType.TEXT if batch_kind == "text" else MessageType.PHOTO
    first = _event(adapter, 950, text="@Woz_Bot first payload", message_type=message_type)
    replay = _event(adapter, 950, text="@Woz_Bot replay payload", message_type=message_type)
    second = _event(adapter, 951, text="@Woz_Bot second payload", message_type=message_type)
    first.media_urls = ["/tmp/first.png"]
    replay.media_urls = ["/tmp/replay.png"]
    second.media_urls = ["/tmp/second.png"]

    if batch_kind == "text":
        adapter._enqueue_text_event(first)
        first_task = next(iter(adapter._pending_text_batch_tasks.values()))
        adapter._enqueue_text_event(replay)
        replay_task = next(iter(adapter._pending_text_batch_tasks.values()))
        adapter._enqueue_text_event(second)
        pending = adapter._pending_text_batches
        tasks = adapter._pending_text_batch_tasks
    elif batch_kind == "photo":
        adapter._enqueue_photo_event("photo-burst", first)
        first_task = next(iter(adapter._pending_photo_batch_tasks.values()))
        adapter._enqueue_photo_event("photo-burst", replay)
        replay_task = next(iter(adapter._pending_photo_batch_tasks.values()))
        adapter._enqueue_photo_event("photo-burst", second)
        pending = adapter._pending_photo_batches
        tasks = adapter._pending_photo_batch_tasks
    else:
        await adapter._queue_media_group_event("album-replay", first)
        first_task = next(iter(adapter._media_group_tasks.values()))
        await adapter._queue_media_group_event("album-replay", replay)
        replay_task = next(iter(adapter._media_group_tasks.values()))
        await adapter._queue_media_group_event("album-replay", second)
        pending = adapter._media_group_events
        tasks = adapter._media_group_tasks

    duplicate_kept_original_timer = replay_task is first_task
    await _cancel_batch_tasks(*list(tasks.values()))
    tasks.clear()
    assert duplicate_kept_original_timer is True
    assert len(pending) == 1
    buffered = next(iter(pending.values()))
    assert getattr(buffered, "_telegram_batch_message_ids") == ("950", "951")
    assert "replay payload" not in (buffered.text or "")
    assert "first payload" in (buffered.text or "")
    assert "second payload" in (buffered.text or "")
    assert buffered.media_urls == ["/tmp/first.png", "/tmp/second.png"]

    key = next(iter(pending))
    if batch_kind == "text":
        adapter._text_batch_delay_seconds = 0
        adapter._TEXT_BATCH_FAST_DELAY_S = 0
        await adapter._flush_text_batch(key)
    elif batch_kind == "photo":
        adapter._media_batch_delay_seconds = 0
        await adapter._flush_photo_batch(key)
    else:
        adapter.MEDIA_GROUP_WAIT_SECONDS = 0
        await adapter._flush_media_group_event(key)

    base_handle.assert_awaited_once_with(first)


@pytest.mark.asyncio
async def test_partial_overlap_batch_is_rejected_without_corrupting_pending_payload():
    _, roster = _runner()
    adapter = roster["engineering"]
    _prepare_batching(adapter)
    first = _event(adapter, 960, text="original payload")
    first.media_urls = ["/tmp/original.png"]
    adapter._enqueue_text_event(first)
    first_task = next(iter(adapter._pending_text_batch_tasks.values()))
    capability = adapter._telegram_batch_capability()
    overlapping = _event(adapter, 960, text="overlapping payload")
    overlapping.media_urls = ["/tmp/overlap.png"]
    setattr(overlapping, "_telegram_batch_message_ids", ("960", "961"))
    setattr(overlapping, "_telegram_batch_reply_to_message_ids", (None, None))
    setattr(overlapping, "_telegram_batch_identity_capability", capability)

    adapter._enqueue_text_event(overlapping)

    assert len(adapter._pending_text_batch_tasks) == 1
    assert next(iter(adapter._pending_text_batch_tasks.values())) is first_task
    await _cancel_batch_tasks(*list(adapter._pending_text_batch_tasks.values()))
    assert len(adapter._pending_text_batches) == 1
    assert getattr(first, "_telegram_batch_message_ids") == ("960",)
    assert first.text == "original payload"
    assert first.media_urls == ["/tmp/original.png"]


@pytest.mark.asyncio
async def test_incompatible_internal_constituent_parents_fail_closed_before_reservation():
    runner, roster = _runner()
    adapter = roster["engineering"]
    dispatcher = runner._telegram_team_dispatcher
    assert dispatcher is not None
    assert dispatcher.record_root(_ALLOWED_CHAT, 700, "engineering") is True
    reserve = Mock(side_effect=AssertionError("invalid provenance must not reserve"))
    dispatcher.reserve_ingress_batch = reserve
    event = _event(adapter, 740, reply_to_message_id=700)
    capability = adapter._telegram_batch_capability()
    setattr(event, "_telegram_batch_message_ids", ("740", "741"))
    setattr(event, "_telegram_batch_reply_to_message_ids", ("700", "999"))
    setattr(event, "_telegram_batch_identity_capability", capability)

    handler = adapter._team_ingress_handler
    assert handler is not None
    assert await handler(adapter, event) is True
    reserve.assert_not_called()


@pytest.mark.asyncio
async def test_unaddressed_batch_carries_validated_ids_only_in_runner_capability(
    monkeypatch,
):
    runner, roster = _runner()
    adapter = roster["default"]
    _prepare_batching(adapter)
    base_handle = AsyncMock()
    monkeypatch.setattr(BasePlatformAdapter, "handle_message", base_handle)
    first = _event(adapter, 900, text="please investigate")
    second = _event(adapter, 901, text="and include this")

    adapter._enqueue_text_event(first)
    task_1 = next(iter(adapter._pending_text_batch_tasks.values()))
    adapter._enqueue_text_event(second)
    task_2 = next(iter(adapter._pending_text_batch_tasks.values()))
    await _cancel_batch_tasks(task_1, task_2)
    batch_key = next(iter(adapter._pending_text_batches))
    adapter._pending_text_batch_tasks.clear()
    adapter._text_batch_delay_seconds = 0
    adapter._TEXT_BATCH_FAST_DELAY_S = 0
    await adapter._flush_text_batch(batch_key)

    base_handle.assert_awaited_once_with(first)
    assert getattr(first, "_telegram_team_constituent_message_ids") == ("900", "901")
    assert (
        getattr(first, "_telegram_team_constituent_ids_capability")
        is getattr(runner, "_telegram_team_constituent_ids_capability")
    )
    assert "constituent" not in repr(first.metadata).lower()
    dispatcher = runner._telegram_team_dispatcher
    assert dispatcher is not None
    assert dispatcher.resolve_reply_owner(
        _ALLOWED_CHAT, 900
    ) is None
    assert dispatcher.resolve_reply_owner(
        _ALLOWED_CHAT, 901
    ) is None


@pytest.mark.asyncio
async def test_interleaved_team_owners_do_not_coalesce_across_adapters():
    _, roster = _runner()
    engineering = roster["engineering"]
    design = roster["design"]
    _prepare_batching(engineering)
    _prepare_batching(design)

    engineering._enqueue_text_event(
        _event(engineering, 910, text="@Woz_Bot first")
    )
    engineering_task_1 = next(iter(engineering._pending_text_batch_tasks.values()))
    design._enqueue_text_event(_event(design, 920, text="@Virgil_Bot first"))
    design_task_1 = next(iter(design._pending_text_batch_tasks.values()))
    engineering._enqueue_text_event(
        _event(engineering, 911, text="@Woz_Bot second")
    )
    engineering_task_2 = next(iter(engineering._pending_text_batch_tasks.values()))
    design._enqueue_text_event(_event(design, 921, text="@Virgil_Bot second"))
    design_task_2 = next(iter(design._pending_text_batch_tasks.values()))

    assert len(engineering._pending_text_batches) == 1
    assert len(design._pending_text_batches) == 1
    engineering_batch = next(iter(engineering._pending_text_batches.values()))
    design_batch = next(iter(design._pending_text_batches.values()))
    assert getattr(engineering_batch, "_telegram_batch_message_ids") == (
        "910",
        "911",
    )
    assert getattr(design_batch, "_telegram_batch_message_ids") == ("920", "921")
    await _cancel_batch_tasks(
        engineering_task_1,
        engineering_task_2,
        design_task_1,
        design_task_2,
    )


@pytest.mark.asyncio
async def test_identical_media_group_ids_do_not_coalesce_across_sessions():
    _, roster = _runner()
    adapter = roster["engineering"]
    _prepare_batching(adapter)
    first_session = _event(
        adapter,
        930,
        text="@Woz_Bot first session",
        message_type=MessageType.PHOTO,
    )
    second_session = _event(
        adapter,
        940,
        text="@Woz_Bot second session",
        message_type=MessageType.PHOTO,
    )
    first_session.source.user_id = "111"
    second_session.source.user_id = "222"
    second_session.source.thread_id = "88"

    await adapter._queue_media_group_event("album-reused", first_session)
    first_task = next(iter(adapter._media_group_tasks.values()))
    await adapter._queue_media_group_event("album-reused", second_session)
    tasks = list(adapter._media_group_tasks.values())

    assert len(adapter._media_group_events) == 2
    batches = list(adapter._media_group_events.values())
    assert getattr(batches[0], "_telegram_batch_message_ids") == ("930",)
    assert getattr(batches[1], "_telegram_batch_message_ids") == ("940",)
    await _cancel_batch_tasks(first_task, *tasks)


@pytest.mark.asyncio
async def test_batch_identity_overflow_splits_without_discarding_message_ids():
    _, roster = _runner()
    adapter = roster["engineering"]
    _prepare_batching(adapter)
    scheduled = []

    for message_id in range(1, 66):
        adapter._enqueue_text_event(
            _event(adapter, message_id, text=f"@Woz_Bot chunk {message_id}")
        )
        scheduled.append(next(reversed(adapter._pending_text_batch_tasks.values())))

    assert len(adapter._pending_text_batches) == 2
    batches = list(adapter._pending_text_batches.values())
    assert getattr(batches[0], "_telegram_batch_message_ids") == tuple(
        str(message_id) for message_id in range(1, 65)
    )
    assert getattr(batches[1], "_telegram_batch_message_ids") == ("65",)
    assert {
        message_id
        for batch in batches
        for message_id in getattr(batch, "_telegram_batch_message_ids")
    } == {str(message_id) for message_id in range(1, 66)}
    await _cancel_batch_tasks(*scheduled)


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_kind", ["text", "command", "photo", "media-group"])
async def test_overflow_isolated_replay_reuses_pending_batch_without_payload_or_timer_duplication(
    monkeypatch,
    batch_kind,
):
    _, roster = _runner()
    adapter = roster["engineering"]
    _prepare_batching(adapter)
    base_handle = AsyncMock()
    monkeypatch.setattr(BasePlatformAdapter, "handle_message", base_handle)
    message_type = (
        MessageType.COMMAND
        if batch_kind == "command"
        else MessageType.TEXT
        if batch_kind == "text"
        else MessageType.PHOTO
    )
    originals = []

    try:
        for message_id in range(1, 66):
            text = (
                f"/status@Woz_Bot original-{message_id}"
                if batch_kind == "command"
                else f"@Woz_Bot original payload {message_id}"
            )
            event = _event(
                adapter,
                message_id,
                text=text,
                message_type=message_type,
            )
            event.media_urls = [f"/tmp/original-{message_id}.png"]
            event.media_types = ["image/png"]
            originals.append(event)
            await _enqueue_batch_event(adapter, batch_kind, event)

        pending, tasks = _pending_batch_state(adapter, batch_kind)
        isolated_key = next(
            key
            for key, batch in pending.items()
            if getattr(batch, "_telegram_batch_message_ids") == ("65",)
        )
        isolated_task = tasks[isolated_key]
        replay = _event(
            adapter,
            65,
            text=(
                "/status@Woz_Bot replay-payload"
                if batch_kind == "command"
                else "@Woz_Bot replay payload 65"
            ),
            message_type=message_type,
        )
        replay.media_urls = ["/tmp/replay-65.png"]
        replay.media_types = ["image/png"]

        await _enqueue_batch_event(adapter, batch_kind, replay)

        assert len(pending) == 2
        assert len(tasks) == 2
        assert tasks[isolated_key] is isolated_task
        assert isolated_key.endswith(":isolated:65")
        isolated = pending[isolated_key]
        assert isolated is originals[-1]
        assert getattr(isolated, "_telegram_batch_message_ids") == ("65",)
        assert "replay" not in (isolated.text or "")
        assert isolated.media_urls == ["/tmp/original-65.png"]

        await _cancel_batch_tasks(*list(tasks.values()))
        tasks.clear()
        for key in list(pending):
            await _flush_pending_batch(adapter, batch_kind, key)

        assert base_handle.await_count == 2
        assert [
            call.args[0].message_id for call in base_handle.await_args_list
        ] == ["1", "65"]
        if batch_kind == "command":
            assert all(
                call.args[0].message_type == MessageType.COMMAND
                for call in base_handle.await_args_list
            )
    finally:
        _pending, pending_tasks = _pending_batch_state(adapter, batch_kind)
        await _cancel_batch_tasks(*list(pending_tasks.values()))


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_kind", ["text", "photo", "media-group"])
async def test_overflow_sibling_overlap_is_rejected_and_disjoint_id_gets_one_stable_batch(
    batch_kind,
):
    _, roster = _runner()
    adapter = roster["engineering"]
    _prepare_batching(adapter)
    message_type = MessageType.TEXT if batch_kind == "text" else MessageType.PHOTO

    try:
        for message_id in range(1, 66):
            event = _event(
                adapter,
                message_id,
                text=f"original payload {message_id}",
                message_type=message_type,
            )
            event.media_urls = [f"/tmp/original-{message_id}.png"]
            event.media_types = ["image/png"]
            await _enqueue_batch_event(adapter, batch_kind, event)

        pending, tasks = _pending_batch_state(adapter, batch_kind)
        isolated_key = next(
            key
            for key, batch in pending.items()
            if getattr(batch, "_telegram_batch_message_ids") == ("65",)
        )
        isolated = pending[isolated_key]
        tasks_before = dict(tasks)
        pending_before = {
            key: (
                id(batch),
                getattr(batch, "_telegram_batch_message_ids"),
                batch.text,
                tuple(batch.media_urls),
                tuple(batch.media_types),
            )
            for key, batch in pending.items()
        }
        capability = adapter._telegram_batch_capability()
        overlapping = _event(
            adapter,
            66,
            text="partial overlap payload",
            message_type=message_type,
        )
        overlapping.media_urls = ["/tmp/partial-overlap.png"]
        overlapping.media_types = ["image/png"]
        setattr(overlapping, "_telegram_batch_message_ids", ("66", "65"))
        setattr(overlapping, "_telegram_batch_reply_to_message_ids", (None, None))
        setattr(overlapping, "_telegram_batch_identity_capability", capability)

        await _enqueue_batch_event(adapter, batch_kind, overlapping)

        assert len(pending) == 2
        assert tasks == tasks_before
        assert {
            key: (
                id(batch),
                getattr(batch, "_telegram_batch_message_ids"),
                batch.text,
                tuple(batch.media_urls),
                tuple(batch.media_types),
            )
            for key, batch in pending.items()
        } == pending_before
        assert pending[isolated_key] is isolated
        assert getattr(isolated, "_telegram_batch_message_ids") == ("65",)
        assert isolated.text == "original payload 65"
        assert isolated.media_urls == ["/tmp/original-65.png"]

        disjoint = _event(
            adapter,
            66,
            text="disjoint payload 66",
            message_type=message_type,
        )
        disjoint.media_urls = ["/tmp/disjoint-66.png"]
        disjoint.media_types = ["image/png"]
        await _enqueue_batch_event(adapter, batch_kind, disjoint)

        assert len(pending) == 3
        assert len(tasks) == 3
        disjoint_keys = [
            key
            for key, batch in pending.items()
            if getattr(batch, "_telegram_batch_message_ids") == ("66",)
        ]
        assert len(disjoint_keys) == 1
        assert disjoint_keys[0].endswith(":isolated:66")
        assert pending[disjoint_keys[0]] is disjoint
    finally:
        _pending, pending_tasks = _pending_batch_state(adapter, batch_kind)
        await _cancel_batch_tasks(*list(pending_tasks.values()))


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_kind", ["text", "photo", "media-group"])
async def test_overflow_pending_identity_cleanup_allows_key_reuse_after_flush_and_cancel(
    monkeypatch,
    batch_kind,
):
    _, roster = _runner()
    adapter = roster["engineering"]
    _prepare_batching(adapter)
    monkeypatch.setattr(BasePlatformAdapter, "handle_message", AsyncMock())
    message_type = MessageType.TEXT if batch_kind == "text" else MessageType.PHOTO

    try:
        for message_id in range(1, 66):
            await _enqueue_batch_event(
                adapter,
                batch_kind,
                _event(
                    adapter,
                    message_id,
                    text=f"payload {message_id}",
                    message_type=message_type,
                ),
            )
        pending, tasks = _pending_batch_state(adapter, batch_kind)
        isolated_key = next(
            key
            for key, batch in pending.items()
            if getattr(batch, "_telegram_batch_message_ids") == ("65",)
        )
        await _cancel_batch_tasks(*list(tasks.values()))
        tasks.clear()

        await _flush_pending_batch(adapter, batch_kind, isolated_key)
        assert isolated_key not in pending
        await _enqueue_batch_event(
            adapter,
            batch_kind,
            _event(
                adapter,
                65,
                text="legitimate replay after flush",
                message_type=message_type,
            ),
        )
        assert isolated_key in pending
        assert getattr(
            pending[isolated_key], "_telegram_batch_message_ids"
        ) == ("65",)

        adapter._drop_delayed_deliveries = True
        await adapter._cancel_pending_delivery_tasks()
        assert pending == {}
        assert tasks == {}
        adapter._held_inbound_events.clear()
        adapter._drop_delayed_deliveries = False

        later = _event(
            adapter,
            65,
            text="legitimate delivery after cancellation",
            message_type=message_type,
        )
        if batch_kind == "text":
            base_key = adapter._text_batch_key(later)
        elif batch_kind == "photo":
            base_key = adapter._team_compatible_telegram_batch_key(
                "overflow-photo",
                later,
            )
        else:
            base_key = (
                f"{adapter._text_batch_key(later)}:album:overflow-album"
            )
        await _enqueue_batch_event(adapter, batch_kind, later)
        assert list(pending) == [base_key]
        assert pending[base_key] is later
    finally:
        adapter._drop_delayed_deliveries = False
        _pending, pending_tasks = _pending_batch_state(adapter, batch_kind)
        await _cancel_batch_tasks(*list(pending_tasks.values()))


@pytest.mark.asyncio
async def test_submitted_overflow_team_command_replay_is_consumed_after_flush(
    monkeypatch,
):
    runner, roster = _runner()
    adapter = roster["engineering"]
    _prepare_batching(adapter)
    dispatcher = runner._telegram_team_dispatcher
    assert dispatcher is not None
    reserve = Mock(side_effect=AssertionError("commands must not reserve ingress"))
    claim = Mock(side_effect=AssertionError("commands must not claim ingress"))
    dispatcher.reserve_ingress_batch = reserve
    dispatcher.claim_ingress = claim
    base_events = []

    async def _base(_self, event):
        base_events.append(event)

    monkeypatch.setattr(BasePlatformAdapter, "handle_message", _base)

    for message_id in range(1, 66):
        adapter._enqueue_text_event(
            _event(
                adapter,
                message_id,
                text=f"/status@Woz_Bot chunk-{message_id}",
                message_type=MessageType.COMMAND,
            )
        )

    await _cancel_batch_tasks(*list(adapter._pending_text_batch_tasks.values()))
    adapter._pending_text_batch_tasks.clear()
    for key in list(adapter._pending_text_batches):
        await _flush_pending_batch(adapter, "command", key)

    assert [event.message_id for event in base_events] == ["1", "65"]

    adapter._text_batch_delay_seconds = 3600
    adapter._TEXT_BATCH_FAST_DELAY_S = 3600
    replay = _event(
        adapter,
        65,
        text="/status@Woz_Bot replay-65",
        message_type=MessageType.COMMAND,
    )
    adapter._enqueue_text_event(replay)
    await _cancel_batch_tasks(*list(adapter._pending_text_batch_tasks.values()))
    adapter._pending_text_batch_tasks.clear()
    replay_key = next(iter(adapter._pending_text_batches))
    await _flush_pending_batch(adapter, "command", replay_key)

    assert [event.message_id for event in base_events] == ["1", "65"]
    reserve.assert_not_called()
    claim.assert_not_called()


@pytest.mark.asyncio
async def test_inflight_team_command_replay_reaches_base_once(monkeypatch):
    _, roster = _runner()
    adapter = roster["engineering"]
    entered = asyncio.Event()
    release = asyncio.Event()
    base_events = []

    async def _blocking_base(_self, event):
        base_events.append(event)
        if len(base_events) == 1:
            entered.set()
            await release.wait()

    monkeypatch.setattr(BasePlatformAdapter, "handle_message", _blocking_base)
    original = _event(
        adapter,
        66,
        text="/status@Woz_Bot original",
        message_type=MessageType.COMMAND,
    )
    replay = _event(
        adapter,
        66,
        text="/status@Woz_Bot replay",
        message_type=MessageType.COMMAND,
    )

    original_task = asyncio.create_task(adapter.handle_message(original))
    await entered.wait()
    await adapter.handle_message(replay)

    assert base_events == [original]
    release.set()
    await original_task


@pytest.mark.asyncio
async def test_cancelled_team_command_retries_same_held_event_without_replay_piggyback(
    monkeypatch,
):
    _, roster = _runner()
    adapter = roster["engineering"]
    _prepare_batching(adapter)
    entered = asyncio.Event()
    never_release = asyncio.Event()
    base_events = []

    async def _base(_self, event):
        base_events.append(event)
        if len(base_events) == 1:
            entered.set()
            await never_release.wait()

    monkeypatch.setattr(BasePlatformAdapter, "handle_message", _base)
    original = _event(
        adapter,
        67,
        text="/status@Woz_Bot original",
        message_type=MessageType.COMMAND,
    )
    replay = _event(
        adapter,
        67,
        text="/status@Woz_Bot replay",
        message_type=MessageType.COMMAND,
    )

    original_task = asyncio.create_task(adapter.handle_message(original))
    await entered.wait()
    await adapter.handle_message(replay)

    original_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await original_task
    adapter._hold_inbound_event(original, where="test-cancel", schedule=False)
    assert adapter._held_inbound_events == [original]

    held_original = adapter._held_inbound_events.pop()
    await adapter.handle_message(held_original)
    await adapter.handle_message(replay)

    assert base_events == [original, original]
    original_token = getattr(original, "_telegram_team_command_submission_token")
    replay_token = getattr(replay, "_telegram_team_command_submission_token")
    assert original_token is not replay_token
    assert "submission" not in repr(original.metadata).lower()
    assert "submission" not in repr(replay.metadata).lower()
    assert (
        getattr(original, "_telegram_team_command_submission_token")
        is original_token
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_stage",
    ["ingress-consume", "ingress-error", "base-error"],
)
async def test_pre_acceptance_team_command_failure_releases_exact_token_for_retry(
    monkeypatch,
    failure_stage,
):
    _, roster = _runner()
    adapter = roster["engineering"]
    original_handler = adapter._team_ingress_handler
    assert original_handler is not None
    base_attempts = []

    if failure_stage.startswith("ingress"):
        async def _failed_ingress(_adapter, _event):
            if failure_stage == "ingress-error":
                raise RuntimeError("ingress failed")
            return True

        adapter.set_team_ingress_handler(_failed_ingress)

    async def _base(_self, event):
        base_attempts.append(event)
        if failure_stage == "base-error" and len(base_attempts) == 1:
            raise RuntimeError("base rejected submission")

    monkeypatch.setattr(BasePlatformAdapter, "handle_message", _base)
    original = _event(
        adapter,
        74,
        text="/status@Woz_Bot retry",
        message_type=MessageType.COMMAND,
    )

    if failure_stage == "base-error":
        with pytest.raises(RuntimeError, match="base rejected"):
            await adapter.handle_message(original)
    else:
        await adapter.handle_message(original)
        adapter.set_team_ingress_handler(original_handler)

    assert adapter._telegram_team_command_submitting == {}
    token = getattr(original, "_telegram_team_command_submission_token")
    await adapter.handle_message(original)
    await adapter.handle_message(
        _event(
            adapter,
            74,
            text="/status@Woz_Bot distinct replay",
            message_type=MessageType.COMMAND,
        )
    )

    expected_attempts = [original, original] if failure_stage == "base-error" else [original]
    assert base_attempts == expected_attempts
    assert getattr(original, "_telegram_team_command_submission_token") is token


@pytest.mark.asyncio
async def test_multi_id_team_command_submission_commits_every_id_atomically(
    monkeypatch,
):
    _, roster = _runner()
    adapter = roster["engineering"]
    base_events = []

    async def _base(_self, event):
        base_events.append(event)

    monkeypatch.setattr(BasePlatformAdapter, "handle_message", _base)
    command = _event(
        adapter,
        68,
        text="/status@Woz_Bot long command",
        message_type=MessageType.COMMAND,
    )
    capability = adapter._telegram_batch_capability()
    setattr(command, "_telegram_batch_message_ids", ("68", "69"))
    setattr(command, "_telegram_batch_reply_to_message_ids", (None, None))
    setattr(command, "_telegram_batch_identity_capability", capability)

    await adapter.handle_message(command)
    await adapter.handle_message(
        _event(
            adapter,
            68,
            text="/status@Woz_Bot replay first",
            message_type=MessageType.COMMAND,
        )
    )
    await adapter.handle_message(
        _event(
            adapter,
            69,
            text="/status@Woz_Bot replay second",
            message_type=MessageType.COMMAND,
        )
    )

    assert base_events == [command]
    assert list(adapter._telegram_team_command_committed) == [
        (_ALLOWED_CHAT, "68"),
        (_ALLOWED_CHAT, "69"),
    ]
    assert adapter._telegram_team_command_submitting == {}


@pytest.mark.asyncio
async def test_team_command_committed_cache_is_bounded_and_evicts_oldest(
    monkeypatch,
):
    _, roster = _runner()
    adapter = roster["engineering"]
    assert adapter._TEAM_COMMAND_SUBMISSION_DEDUPE_MAX >= 2048
    adapter._TEAM_COMMAND_SUBMISSION_DEDUPE_MAX = 2
    base_events = []

    async def _base(_self, event):
        base_events.append(event)

    monkeypatch.setattr(BasePlatformAdapter, "handle_message", _base)
    commands = [
        _event(
            adapter,
            message_id,
            text=f"/status@Woz_Bot command-{message_id}",
            message_type=MessageType.COMMAND,
        )
        for message_id in (75, 76, 77)
    ]

    for command in commands:
        await adapter.handle_message(command)

    assert list(adapter._telegram_team_command_committed) == [
        (_ALLOWED_CHAT, "76"),
        (_ALLOWED_CHAT, "77"),
    ]
    await adapter.handle_message(
        _event(
            adapter,
            75,
            text="/status@Woz_Bot oldest replay",
            message_type=MessageType.COMMAND,
        )
    )
    assert [event.message_id for event in base_events] == ["75", "76", "77", "75"]
    assert list(adapter._telegram_team_command_committed) == [
        (_ALLOWED_CHAT, "77"),
        (_ALLOWED_CHAT, "75"),
    ]


@pytest.mark.asyncio
async def test_invalid_overflow_team_command_does_not_mutate_or_evict_dedupe(
    monkeypatch,
):
    _, roster = _runner()
    adapter = roster["engineering"]
    base_events = []

    async def _base(_self, event):
        base_events.append(event)

    monkeypatch.setattr(BasePlatformAdapter, "handle_message", _base)
    valid = _event(
        adapter,
        70,
        text="/status@Woz_Bot valid",
        message_type=MessageType.COMMAND,
    )
    await adapter.handle_message(valid)
    committed_before = list(
        getattr(adapter, "_telegram_team_command_committed", {})
    )

    overflow = _event(
        adapter,
        71,
        text="/status@Woz_Bot overflow",
        message_type=MessageType.COMMAND,
    )
    capability = adapter._telegram_batch_capability()
    overflow_ids = tuple(str(message_id) for message_id in range(71, 136))
    setattr(overflow, "_telegram_batch_message_ids", overflow_ids)
    setattr(
        overflow,
        "_telegram_batch_reply_to_message_ids",
        (None,) * len(overflow_ids),
    )
    setattr(overflow, "_telegram_batch_identity_capability", capability)

    await adapter.handle_message(overflow)
    await adapter.handle_message(overflow)
    await adapter.handle_message(
        _event(
            adapter,
            70,
            text="/status@Woz_Bot committed replay",
            message_type=MessageType.COMMAND,
        )
    )

    assert base_events == [valid, overflow, overflow]
    assert list(adapter._telegram_team_command_committed) == committed_before
    assert adapter._telegram_team_command_submitting == {}


@pytest.mark.asyncio
async def test_non_team_command_replay_preserves_legacy_base_handling(monkeypatch):
    _, roster = _runner()
    adapter = roster["default"]
    base_handle = AsyncMock()
    monkeypatch.setattr(BasePlatformAdapter, "handle_message", base_handle)
    first = _event(
        adapter,
        72,
        chat_id=-999,
        text="/status",
        message_type=MessageType.COMMAND,
    )
    replay = _event(
        adapter,
        72,
        chat_id=-999,
        text="/status",
        message_type=MessageType.COMMAND,
    )

    await adapter.handle_message(first)
    await adapter.handle_message(replay)

    assert base_handle.await_args_list == [
        ((first,), {}),
        ((replay,), {}),
    ]


@pytest.mark.asyncio
async def test_team_handler_replacement_clears_only_submitting_command_tokens(
    monkeypatch,
):
    _, roster = _runner()
    adapter = roster["engineering"]
    handler = adapter._team_ingress_handler
    assert handler is not None
    entered = asyncio.Event()
    never_release = asyncio.Event()

    async def _blocking_base(_self, _event):
        entered.set()
        await never_release.wait()

    monkeypatch.setattr(BasePlatformAdapter, "handle_message", _blocking_base)
    command = _event(
        adapter,
        73,
        text="/status@Woz_Bot replace",
        message_type=MessageType.COMMAND,
    )
    task = asyncio.create_task(adapter.handle_message(command))
    await entered.wait()
    try:
        assert adapter._telegram_team_command_submitting

        adapter.set_team_ingress_handler(handler)

        assert adapter._telegram_team_command_submitting == {}
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_kwargs",
    [
        {"chat_id": 123, "chat_type": "dm", "raw_chat_type": "private"},
        {"chat_id": -999, "chat_type": "group", "raw_chat_type": "supergroup"},
        {"message_type": MessageType.COMMAND, "text": "/status@Ace_Bot"},
    ],
    ids=["dm", "outside-chat", "command"],
)
async def test_dm_outside_chat_and_commands_are_not_claimed(event_kwargs):
    runner, roster = _runner()
    adapter = roster["default"]
    claim = Mock(wraps=runner._telegram_team_dispatcher.claim_ingress)
    runner._telegram_team_dispatcher.claim_ingress = claim
    event = _event(adapter, 110, **event_kwargs)

    handler = adapter._team_ingress_handler
    assert handler is not None
    consumed = await handler(adapter, event)

    assert consumed is False
    claim.assert_not_called()
    assert "telegram_team_ingress_claimed" not in event.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/stop@Woz_Bot", "/new", "/status", "/approve"])
async def test_command_reply_to_known_root_uses_root_owner_session_without_reservation(
    command,
):
    runner, roster = _runner()
    adapter = roster["engineering"]
    dispatcher = runner._telegram_team_dispatcher
    assert dispatcher is not None
    assert dispatcher.record_root(_ALLOWED_CHAT, 700, "engineering") is True
    reserve = Mock(side_effect=AssertionError("commands must not reserve ingress"))
    claim = Mock(side_effect=AssertionError("commands must not claim ingress"))
    dispatcher.reserve_ingress_batch = reserve
    dispatcher.claim_ingress = claim
    event = _event(
        adapter,
        701,
        text=command,
        reply_to_message_id=700,
        message_type=MessageType.COMMAND,
    )

    handler = adapter._team_ingress_handler
    assert handler is not None
    consumed = await handler(adapter, event)

    assert consumed is False
    assert event.source.profile == "engineering"
    assert event.source.session_scope_id == f"telegram-team:{_ALLOWED_CHAT}:700"
    assert event.metadata["telegram_team_root_message_id"] == "700"
    assert event.metadata["telegram_team_owner_profile"] == "engineering"
    assert event.metadata["telegram_team_route_reason"] == "command_reply_to_root"
    assert "telegram_team_ingress_claimed" not in event.metadata
    command_key = build_session_key(
        event.source,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
        profile=adapter._session_key_profile(event.source),
    )
    root_source = dataclasses.replace(
        event.source,
        message_id="700",
        session_scope_id=f"telegram-team:{_ALLOWED_CHAT}:700",
    )
    root_key = build_session_key(
        root_source,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
        profile=adapter._session_key_profile(root_source),
    )
    assert command_key == root_key
    reserve.assert_not_called()
    claim.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_to_message_id", [None, 999])
async def test_bare_or_unrelated_command_keeps_legacy_session_and_does_not_reserve(
    reply_to_message_id,
):
    runner, roster = _runner()
    adapter = roster["default"]
    dispatcher = runner._telegram_team_dispatcher
    assert dispatcher is not None
    reserve = Mock(side_effect=AssertionError("commands must not reserve ingress"))
    claim = Mock(side_effect=AssertionError("commands must not claim ingress"))
    dispatcher.reserve_ingress_batch = reserve
    dispatcher.claim_ingress = claim
    event = _event(
        adapter,
        702,
        text="/status",
        reply_to_message_id=reply_to_message_id,
        message_type=MessageType.COMMAND,
    )
    source_before = event.source.to_dict()
    metadata_before = dict(event.metadata)

    handler = adapter._team_ingress_handler
    assert handler is not None
    assert await handler(adapter, event) is False

    assert event.source.to_dict() == source_before
    assert event.metadata == metadata_before
    reserve.assert_not_called()
    claim.assert_not_called()


@pytest.mark.asyncio
async def test_command_reply_to_known_root_is_consumed_by_nonowner_without_reservation():
    runner, roster = _runner()
    dispatcher = runner._telegram_team_dispatcher
    assert dispatcher is not None
    assert dispatcher.record_root(_ALLOWED_CHAT, 700, "engineering") is True
    reserve = Mock(side_effect=AssertionError("commands must not reserve ingress"))
    dispatcher.reserve_ingress_batch = reserve
    event = _event(
        roster["design"],
        703,
        text="/status",
        reply_to_message_id=700,
        message_type=MessageType.COMMAND,
    )

    handler = roster["design"]._team_ingress_handler
    assert handler is not None
    assert await handler(roster["design"], event) is True
    reserve.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_branch",
    [
        "resolve-owner-raises",
        "wrong-parent-owner",
        "record-alias-fails",
        "resolve-addressed-raises",
        "decision-owner-mismatch",
        "record-root-fails",
        "source-scope-raises",
    ],
)
async def test_every_post_reservation_ingress_failure_releases_atomically(
    monkeypatch,
    failure_branch,
):
    import gateway.telegram_team_routing as routing

    runner, roster = _runner()
    adapter = roster["engineering"]
    dispatcher = runner._telegram_team_dispatcher
    assert dispatcher is not None
    event = _event(adapter, 730, text="@Woz_Bot investigate")
    single_api_mock = None
    batch_api_mock = None

    if failure_branch == "resolve-owner-raises":
        dispatcher.resolve_reply_owner = Mock(side_effect=RuntimeError("resolve failed"))
    elif failure_branch == "wrong-parent-owner":
        event = _event(adapter, 730, reply_to_message_id=700)
        dispatcher.resolve_reply_owner = Mock(
            return_value=RootOwnership("700", "design")
        )
    elif failure_branch == "record-alias-fails":
        assert dispatcher.record_root(_ALLOWED_CHAT, 700, "engineering") is True
        event = _event(adapter, 730, reply_to_message_id=700)
        single_api_mock = Mock(
            side_effect=AssertionError("runner must use the atomic batch API")
        )
        batch_api_mock = Mock(return_value=False)
        dispatcher.record_inbound_alias = single_api_mock
        dispatcher.record_inbound_alias_batch = batch_api_mock
    elif failure_branch == "resolve-addressed-raises":
        monkeypatch.setattr(
            routing,
            "resolve_addressed_owner",
            Mock(side_effect=RuntimeError("routing failed")),
        )
    elif failure_branch == "decision-owner-mismatch":
        event = _event(adapter, 730, text="unaddressed")
    elif failure_branch == "record-root-fails":
        single_api_mock = Mock(
            side_effect=AssertionError("runner must use the atomic batch API")
        )
        batch_api_mock = Mock(return_value=False)
        dispatcher.record_root = single_api_mock
        dispatcher.record_root_batch = batch_api_mock
    else:
        assert event.source is not None
        source_type = type(event.source)

        class ExplodingSource(source_type):
            _explode_on_scope = False

            def __setattr__(self, name, value):
                if name == "session_scope_id" and self._explode_on_scope:
                    raise RuntimeError("source scope assignment failed")
                super().__setattr__(name, value)

        event.source.__class__ = ExplodingSource
        setattr(event.source, "_explode_on_scope", True)

    handler = adapter._team_ingress_handler
    assert handler is not None
    consumed = await handler(adapter, event)

    assert consumed is True
    if failure_branch == "record-alias-fails":
        assert single_api_mock is not None
        assert batch_api_mock is not None
        single_api_mock.assert_not_called()
        batch_api_mock.assert_called_once_with(
            _ALLOWED_CHAT,
            ("730",),
            "700",
        )
    elif failure_branch == "record-root-fails":
        assert single_api_mock is not None
        assert batch_api_mock is not None
        single_api_mock.assert_not_called()
        batch_api_mock.assert_called_once_with(
            _ALLOWED_CHAT,
            "730",
            "engineering",
            ("730",),
        )
    retry = dispatcher.reserve_ingress_batch(_ALLOWED_CHAT, [730], "engineering")
    assert retry.reserved is True
    assert retry.reservation is not None
    assert dispatcher.release_ingress(retry.reservation) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forgery",
    ["unknown-capability", "external-metadata", "external-runner-metadata"],
)
async def test_forged_batch_identity_is_consumed_without_reserving(forgery):
    runner, roster = _runner()
    adapter = roster["engineering"]
    dispatcher = runner._telegram_team_dispatcher
    assert dispatcher is not None
    reserve = Mock(side_effect=AssertionError("forged state must not reserve"))
    dispatcher.reserve_ingress_batch = reserve
    event = _event(adapter, 740, text="@Woz_Bot investigate")
    if forgery == "unknown-capability":
        setattr(event, "_telegram_batch_message_ids", ("740", "741"))
        setattr(event, "_telegram_batch_identity_capability", object())
    elif forgery == "external-metadata":
        event.metadata["telegram_batch_message_ids"] = ["740", "741"]
    else:
        event.metadata["_telegram_team_constituent_message_ids"] = ["740", "741"]

    handler = adapter._team_ingress_handler
    assert handler is not None
    assert await handler(adapter, event) is True
    reserve.assert_not_called()


@pytest.mark.asyncio
async def test_structural_ingress_records_three_hop_human_chain_with_interleaved_roots():
    runner, roster = _runner()
    dispatcher = runner._telegram_team_dispatcher

    engineering_root = _event(
        roster["engineering"], 100, text="@Woz_Bot own this"
    )
    design_root = _event(roster["design"], 200, text="@Virgil_Bot own this")
    engineering_reply_1 = _event(
        roster["engineering"], 101, reply_to_message_id=100
    )
    design_reply_1 = _event(roster["design"], 201, reply_to_message_id=200)
    engineering_reply_2 = _event(
        roster["engineering"], 102, reply_to_message_id=101
    )

    for adapter, event in (
        (roster["engineering"], engineering_root),
        (roster["design"], design_root),
        (roster["engineering"], engineering_reply_1),
        (roster["design"], design_reply_1),
        (roster["engineering"], engineering_reply_2),
    ):
        assert await adapter._team_ingress_handler(adapter, event) is False

    assert dispatcher.resolve_reply_owner(_ALLOWED_CHAT, 102) == RootOwnership(
        "100", "engineering"
    )
    assert dispatcher.resolve_reply_owner(_ALLOWED_CHAT, 201) == RootOwnership(
        "200", "design"
    )
    assert engineering_reply_2.source.session_scope_id == (
        f"telegram-team:{_ALLOWED_CHAT}:100"
    )
    assert engineering_reply_2.metadata["telegram_team_route_reason"] == (
        "reply_to_root_chain"
    )
    assert design_reply_1.metadata["telegram_team_root_message_id"] == "200"


@pytest.mark.asyncio
async def test_unaddressed_ingress_defers_immutable_root_binding():
    runner, roster = _runner()
    event = _event(roster["default"], 300, text="please investigate this")

    assert await roster["default"]._team_ingress_handler(
        roster["default"], event
    ) is False

    assert runner._telegram_team_dispatcher.resolve_reply_owner(
        _ALLOWED_CHAT, 300
    ) is None
    assert event.metadata == {
        "existing": "value",
        "telegram_team_root_message_id": "300",
        "telegram_team_owner_profile": "default",
        "telegram_team_route_reason": "unaddressed_ingress",
        "telegram_team_ingress_claimed": True,
        "telegram_team_root_binding_deferred": True,
    }
    assert event.source.session_scope_id == f"telegram-team:{_ALLOWED_CHAT}:300"


@pytest.mark.asyncio
async def test_suspended_runtime_consumes_allowed_team_ingress_fail_closed():
    runner, roster = _runner()
    runner._disable_telegram_team_runtime("test", log=False)
    claim = Mock(wraps=runner._telegram_team_dispatcher.claim_ingress)
    runner._telegram_team_dispatcher.claim_ingress = claim

    consumed = await roster["default"]._team_ingress_handler(
        roster["default"], _event(roster["default"], 301)
    )

    assert consumed is True
    claim.assert_not_called()


def test_build_target_event_uses_fresh_target_provenance_scope_and_safe_copies():
    runner, roster = _runner()
    original = _event(
        roster["default"],
        400,
        text="@Ace_Bot route me",
        reply_to_message_id=399,
    )
    original.source.profile_route_rejected = True
    original_profile = original.source.profile
    original_metadata = dict(original.metadata)
    target_handle = AsyncMock()
    roster["engineering"].handle_message = target_handle

    routed = runner.build_target_event(
        original,
        roster["engineering"],
        "engineering",
        "350",
        route_reason="semantic_specialist",
    )

    assert routed is not original
    assert routed.source is not original.source
    assert runner._adapter_for_source(original.source) is roster["default"]
    assert runner._adapter_for_source(routed.source) is roster["engineering"]
    assert routed.source._transport_adapter_ref() is roster["engineering"]
    assert routed.source.profile == "engineering"
    assert routed.source.profile_route_rejected is False
    assert routed.source.session_scope_id == f"telegram-team:{_ALLOWED_CHAT}:350"
    assert routed.source.thread_id == "77"
    assert routed.source.message_id == "400"
    assert routed.source.user_id == "123"
    assert routed.source.user_id_alt == "alice-alt"
    assert routed.text == original.text
    assert routed.raw_message is original.raw_message
    assert routed.timestamp is original.timestamp
    assert routed.reply_to_message_id == "399"
    assert routed.reply_to_text == original.reply_to_text
    assert routed.channel_prompt == original.channel_prompt
    assert routed.auto_skill == original.auto_skill
    assert routed.media_urls == original.media_urls
    assert routed.media_urls is not original.media_urls
    assert routed.media_types == original.media_types
    assert routed.media_types is not original.media_types
    assert routed.metadata["telegram_team_routed"] is True
    assert routed.metadata["telegram_team_owner_profile"] == "engineering"
    assert routed.metadata["telegram_team_root_message_id"] == "350"
    assert routed.metadata["telegram_team_route_reason"] == "semantic_specialist"
    assert routed.metadata["telegram_team_ingress_claimed"] is True
    assert original.metadata == original_metadata
    assert original.source.profile == original_profile
    assert original.source.session_scope_id is None
    assert "telegram_team_route_capability" not in routed.metadata
    assert "telegram_team_route_capability" not in routed.source.to_dict()
    target_handle.assert_not_awaited()


@pytest.mark.asyncio
async def test_forged_routed_marker_is_consumed_without_claiming():
    runner, roster = _runner()
    adapter = roster["engineering"]
    claim = Mock(side_effect=AssertionError("forged marker must not claim"))
    runner._telegram_team_dispatcher.claim_ingress = claim
    forged = _event(
        adapter,
        500,
        metadata={
            "telegram_team_routed": True,
            "telegram_team_owner_profile": "engineering",
            "telegram_team_root_message_id": "450",
            "telegram_team_route_reason": "semantic_specialist",
            "telegram_team_ingress_claimed": True,
        },
    )

    assert await adapter._team_ingress_handler(adapter, forged) is True
    claim.assert_not_called()


@pytest.mark.asyncio
async def test_valid_internal_routed_event_bypasses_second_claim_and_classification():
    runner, roster = _runner()
    original = _event(roster["default"], 600, text="unaddressed source")
    routed = runner.build_target_event(
        original,
        roster["design"],
        "design",
        "600",
        route_reason="semantic_specialist",
    )
    claim = Mock(side_effect=AssertionError("routed target must not claim again"))
    runner._telegram_team_dispatcher.claim_ingress = claim

    consumed = await roster["design"]._team_ingress_handler(
        roster["design"], routed
    )

    assert consumed is False
    claim.assert_not_called()


@pytest.mark.asyncio
async def test_internal_routed_event_with_mismatched_owner_is_consumed():
    runner, roster = _runner()
    original = _event(roster["default"], 700)
    routed = runner.build_target_event(
        original,
        roster["engineering"],
        "engineering",
        "700",
        route_reason="semantic_specialist",
    )
    routed.metadata["telegram_team_owner_profile"] = "design"
    claim = Mock(side_effect=AssertionError("malformed target must not claim"))
    runner._telegram_team_dispatcher.claim_ingress = claim

    assert await roster["engineering"]._team_ingress_handler(
        roster["engineering"], routed
    ) is True
    claim.assert_not_called()
