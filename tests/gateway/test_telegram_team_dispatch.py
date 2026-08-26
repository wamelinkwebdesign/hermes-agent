"""Central Telegram team ingress and fresh target-event construction tests."""

from __future__ import annotations

import time
import weakref
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
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

    consumed = await adapter._team_ingress_handler(adapter, event)

    assert consumed is False
    claim.assert_not_called()
    assert "telegram_team_ingress_claimed" not in event.metadata


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
