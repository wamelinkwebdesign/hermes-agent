"""Multiplex runtime validation and shared Telegram team-gate wiring."""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner, MultiplexConfigError
from gateway.telegram_team_routing import (
    TelegramTeamRouteContext,
    TelegramTeamDispatcher,
)
from plugins.platforms.telegram.adapter import TelegramAdapter


_ALLOWED_CHAT = "-1001234567890"
_DEFAULT = object()
_LIVE_USERNAMES = {
    "default": "Ace_Bot",
    "engineering": "Woz_Bot",
    "design": "Virgil_Bot",
}


def _team_raw() -> dict:
    return {
        "coordinator_profile": "default",
        "coordinator_username": "Ace_Bot",
        "members": {
            "default": "Ace_Bot",
            "engineering": "Woz_Bot",
            "design": "Virgil_Bot",
        },
        "allowed_chats": [_ALLOWED_CHAT],
    }


def _telegram_adapter(
    profile: str,
    token: str,
    *,
    live_username: object = _DEFAULT,
) -> TelegramAdapter:
    if live_username is _DEFAULT:
        live_username = _LIVE_USERNAMES.get(profile)
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token=token, extra={})
    adapter._team_route_gate = None
    adapter._team_ingress_handler = None
    adapter._bot_username_observed = (
        str(live_username).lstrip("@").lower() if live_username else None
    )
    adapter._bot = SimpleNamespace(username=live_username)
    adapter._bot_identity_checked_at = time.monotonic()
    adapter.set_owner_profile(profile)
    return adapter


def _runner(
    *,
    raw: object = _DEFAULT,
    multiplex: bool = True,
    allowlist: Any = _DEFAULT,
    active: str = "default",
    adapters: dict[str, Any] | None = None,
    telegram_enabled: bool = True,
    include_team_routing: bool = True,
) -> tuple[GatewayRunner, dict[str, Any]]:
    if raw is _DEFAULT:
        raw = _team_raw()
    if allowlist is _DEFAULT:
        allowlist = ["engineering", "design"]
    telegram_extra = {"team_routing": raw} if include_team_routing else {}
    config = GatewayConfig(
        multiplex_profiles=multiplex,
        multiplex_profile_allowlist=allowlist,
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=telegram_enabled,
                token="token-default",
                extra=telegram_extra,
            )
        },
    )
    roster = adapters or {
        "default": _telegram_adapter("default", "token-default"),
        "engineering": _telegram_adapter("engineering", "token-engineering"),
        "design": _telegram_adapter("design", "token-design"),
    }
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._active_profile_name = lambda: active
    if "default" in roster:
        runner.adapters[Platform.TELEGRAM] = roster["default"]
    for profile, adapter in roster.items():
        if profile != "default":
            runner._profile_adapters[profile] = {Platform.TELEGRAM: adapter}
    return runner, roster


def _context(
    profile: str,
    *,
    mentions: set[str] | None = None,
    reply_author_username: str | None = None,
    chat_id: str = _ALLOWED_CHAT,
    message_id: str | None = "500",
    reply_to_message_id: str | None = None,
) -> TelegramTeamRouteContext:
    usernames = {
        "default": "ace_bot",
        "engineering": "woz_bot",
        "design": "virgil_bot",
    }
    return TelegramTeamRouteContext(
        mentions=frozenset(mentions or set()),
        reply_author_username=reply_author_username,
        chat_id=chat_id,
        owner_profile=profile,
        owner_username=usernames[profile],
        message_id=message_id,
        reply_to_message_id=reply_to_message_id,
    )


def _gates(roster: dict[str, Any]) -> dict[str, Any]:
    return {
        profile: getattr(adapter, "_team_route_gate", None)
        for profile, adapter in roster.items()
    }


def _ingresses(roster: dict[str, Any]) -> dict[str, Any]:
    return {
        profile: getattr(adapter, "_team_ingress_handler", None)
        for profile, adapter in roster.items()
    }


def _stub_profile_adapter_dependencies(runner: Any) -> None:
    runner.session_store = object()
    runner._busy_text_modes_by_profile = {}
    runner._busy_text_mode = "interrupt"
    runner._make_profile_message_handler = lambda _profile: object()
    runner._make_profile_fatal_error_handler = lambda _profile, _platform: object()
    runner._make_profile_busy_session_handler = lambda _profile: object()
    runner._handle_reaction_event = object()
    runner._recover_telegram_topic_thread_id = object()
    runner._make_adapter_auth_check = lambda _platform, **_kwargs: object()
    runner._make_profile_platform_event_handler = lambda _profile: object()


def test_valid_complete_roster_installs_one_shared_gate_ingress_and_dispatcher():
    runner, roster = _runner()

    assert runner._validate_and_install_telegram_team_runtime() is True

    assert runner.telegram_team_routing_enabled is True
    assert runner.telegram_team_profiles == ("default", "design", "engineering")
    dispatcher = runner._telegram_team_dispatcher
    assert isinstance(dispatcher, TelegramTeamDispatcher)
    callbacks = list(_gates(roster).values())
    assert all(callable(callback) for callback in callbacks)
    assert len({id(callback) for callback in callbacks}) == 1
    ingress_callbacks = list(_ingresses(roster).values())
    assert all(callable(callback) for callback in ingress_callbacks)
    assert len({id(callback) for callback in ingress_callbacks}) == 1
    assert {
        profile: adapter._current_bot_username() for profile, adapter in roster.items()
    } == dict(runner._telegram_team_config.members)


def test_absent_team_routing_key_preserves_legacy_none_gate_and_ingress():
    runner, roster = _runner(include_team_routing=False)

    assert runner._prepare_telegram_team_runtime() is False
    assert runner._validate_and_install_telegram_team_runtime() is False
    assert all(gate is None for gate in _gates(roster).values())
    assert all(handler is None for handler in _ingresses(roster).values())


@pytest.mark.parametrize(
    "runner_kwargs",
    [
        {"raw": {"coordinator_profile": "default"}},
        {"raw": {**_team_raw(), "coordinator_username": "Other_Bot"}},
        {"telegram_enabled": False},
        {"multiplex": False},
        {"active": "engineering"},
        {"allowlist": None},
        {"allowlist": ["engineering"]},
    ],
    ids=[
        "malformed",
        "coordinator-username-mismatch",
        "telegram-disabled",
        "multiplex-off",
        "coordinator-not-active",
        "allowlist-absent",
        "allowlist-incomplete",
    ],
)
def test_present_but_statically_invalid_team_config_raises_token_safe_error(
    runner_kwargs, caplog
):
    secret = "static-token-sentinel-never-log-7f3c"
    runner, roster = _runner(**runner_kwargs)
    runner.config.platforms[Platform.TELEGRAM].token = secret
    caplog.set_level(logging.WARNING, logger="gateway.run")

    with pytest.raises(MultiplexConfigError) as exc_info:
        runner._prepare_telegram_team_runtime()

    assert "Telegram team routing" in str(exc_info.value)
    assert secret not in str(exc_info.value)
    assert secret not in caplog.text
    assert all(gate is None for gate in _gates(roster).values())


def test_incomplete_live_roster_keeps_reachable_shared_gates_pending_and_rejecting():
    roster = {
        "default": _telegram_adapter("default", "token-default"),
        "engineering": _telegram_adapter("engineering", "token-engineering"),
    }
    runner, roster = _runner(adapters=roster)

    assert runner._validate_and_install_telegram_team_runtime() is False

    callbacks = _gates(roster)
    assert runner.telegram_team_routing_enabled is False
    assert runner.telegram_team_profiles == ()
    assert all(callable(callback) for callback in callbacks.values())
    assert len({id(callback) for callback in callbacks.values()}) == 1
    assert all(
        callback(_context(profile)) is False for profile, callback in callbacks.items()
    )


@pytest.mark.parametrize("failure", ["missing", "wrong-type", "wrong-owner"])
def test_each_member_must_resolve_to_exact_owned_telegram_adapter(failure):
    roster: dict[str, object] = {
        "default": _telegram_adapter("default", "token-default"),
        "engineering": _telegram_adapter("engineering", "token-engineering"),
        "design": _telegram_adapter("design", "token-design"),
    }
    if failure == "missing":
        roster.pop("engineering")
    elif failure == "wrong-type":
        roster["engineering"] = SimpleNamespace(
            platform=Platform.TELEGRAM,
            config=PlatformConfig(enabled=True, token="token-engineering"),
        )
    else:
        roster["engineering"]._owner_profile = "other-profile"
    runner, roster = _runner(adapters=roster)

    assert runner._validate_and_install_telegram_team_runtime() is False
    assert runner.telegram_team_routing_enabled is False
    reachable_profiles = {
        profile
        for profile, adapter in roster.items()
        if isinstance(adapter, TelegramAdapter)
    }
    for profile, gate in _gates(roster).items():
        if profile in reachable_profiles:
            assert callable(gate)
            assert gate(_context(profile)) is False


@pytest.mark.parametrize("failure", ["empty", "duplicate"])
def test_team_tokens_must_be_nonempty_and_distinct(failure):
    shared = "telegram-secret-token-never-log"
    roster = {
        "default": _telegram_adapter("default", "token-default"),
        "engineering": _telegram_adapter(
            "engineering", "   " if failure == "empty" else shared
        ),
        "design": _telegram_adapter(
            "design", "token-design" if failure == "empty" else shared
        ),
    }
    runner, roster = _runner(adapters=roster)

    assert runner._validate_and_install_telegram_team_runtime() is False
    assert runner.telegram_team_routing_enabled is False
    callbacks = _gates(roster)
    assert all(callable(callback) for callback in callbacks.values())
    assert len({id(callback) for callback in callbacks.values()}) == 1
    assert all(
        callback(_context(profile)) is False for profile, callback in callbacks.items()
    )


def test_gate_installation_failure_keeps_surviving_callbacks_pending():
    runner, roster = _runner()
    design = roster["design"]
    original_setter = design.set_team_route_gate

    def fail_install(callback):
        if callback is not None:
            raise RuntimeError("setter failed")
        original_setter(None)

    design.set_team_route_gate = fail_install

    assert runner._validate_and_install_telegram_team_runtime() is False
    assert runner.telegram_team_routing_enabled is False
    assert callable(roster["default"]._team_route_gate)
    assert callable(roster["engineering"]._team_route_gate)
    assert roster["default"]._team_route_gate(_context("default")) is False
    assert roster["engineering"]._team_route_gate(_context("engineering")) is False


def test_runtime_revalidation_reuses_exactly_one_dispatcher(monkeypatch):
    import gateway.telegram_team_routing as routing

    real_dispatcher = routing.TelegramTeamDispatcher
    created = []

    class CountingDispatcher(real_dispatcher):
        def __init__(self):
            super().__init__()
            created.append(self)

    monkeypatch.setattr(routing, "TelegramTeamDispatcher", CountingDispatcher)
    runner, roster = _runner()

    assert runner._validate_and_install_telegram_team_runtime() is True
    first_callbacks = _gates(roster)
    assert runner._validate_and_install_telegram_team_runtime() is True

    assert created == [runner._telegram_team_dispatcher]
    assert _gates(roster) == first_callbacks


@pytest.mark.parametrize(
    ("mentions", "reply_author", "expected_owner"),
    [
        (set(), "Woz_Bot", "engineering"),
        ({"Virgil_Bot"}, None, "design"),
        ({"Woz_Bot", "Virgil_Bot"}, None, "default"),
        (set(), None, "default"),
    ],
    ids=["reply", "single-mention", "multiple-mentions", "unaddressed"],
)
def test_gate_accepts_only_resolved_owner(mentions, reply_author, expected_owner):
    runner, roster = _runner()
    assert runner._validate_and_install_telegram_team_runtime() is True

    decisions = {
        profile: adapter._team_route_gate(
            _context(
                profile,
                mentions=mentions,
                reply_author_username=reply_author,
            )
        )
        for profile, adapter in roster.items()
    }

    assert decisions == {profile: profile == expected_owner for profile in roster}


def test_gate_known_root_alias_precedes_reply_author_and_mentions():
    runner, roster = _runner()
    assert runner._validate_and_install_telegram_team_runtime() is True
    dispatcher = runner._telegram_team_dispatcher
    assert dispatcher.record_root(_ALLOWED_CHAT, 100, "engineering") is True
    assert dispatcher.record_inbound_alias(_ALLOWED_CHAT, 101, 100) is True

    decisions = {
        profile: adapter._team_route_gate(
            _context(
                profile,
                mentions={"Virgil_Bot"},
                reply_author_username="Ace_Bot",
                message_id="102",
                reply_to_message_id="101",
            )
        )
        for profile, adapter in roster.items()
    }

    assert decisions == {
        "default": False,
        "engineering": True,
        "design": False,
    }


def test_gate_keeps_interleaved_three_hop_human_reply_roots_separate():
    runner, roster = _runner()
    assert runner._validate_and_install_telegram_team_runtime() is True
    dispatcher = runner._telegram_team_dispatcher
    assert dispatcher.record_root(_ALLOWED_CHAT, 100, "engineering") is True
    assert dispatcher.record_root(_ALLOWED_CHAT, 200, "design") is True
    assert dispatcher.record_inbound_alias(_ALLOWED_CHAT, 101, 100) is True
    assert dispatcher.record_inbound_alias(_ALLOWED_CHAT, 201, 200) is True
    assert dispatcher.record_inbound_alias(_ALLOWED_CHAT, 102, 101) is True

    for reply_id, expected_owner in (("102", "engineering"), ("201", "design")):
        assert {
            profile: adapter._team_route_gate(
                _context(
                    profile,
                    mentions={"Ace_Bot"},
                    reply_author_username="Human_User",
                    message_id="300",
                    reply_to_message_id=reply_id,
                )
            )
            for profile, adapter in roster.items()
        } == {profile: profile == expected_owner for profile in roster}


def test_gate_rejects_every_team_adapter_outside_allowed_chats():
    runner, roster = _runner()
    assert runner._validate_and_install_telegram_team_runtime() is True

    assert {
        adapter._team_route_gate(_context(profile, chat_id="-999"))
        for profile, adapter in roster.items()
    } == {False}


def test_nonowner_arrival_cannot_steal_dispatcher_claim():
    runner, roster = _runner()
    assert runner._validate_and_install_telegram_team_runtime() is True
    runner._telegram_team_dispatcher.claim_ingress = Mock(
        side_effect=AssertionError("per-adapter gate must not claim ingress")
    )
    callback = roster["engineering"]._team_route_gate

    assert callback(_context("engineering")) is False
    assert callback(_context("default")) is True
    runner._telegram_team_dispatcher.claim_ingress.assert_not_called()


def test_replacement_receives_shared_callbacks_and_stale_callbacks_are_removed():
    runner, roster = _runner()
    unrelated_gate = object()
    unrelated_ingress = object()
    unrelated = SimpleNamespace(
        _team_route_gate=unrelated_gate,
        _team_ingress_handler=unrelated_ingress,
    )
    assert runner._validate_and_install_telegram_team_runtime() is True
    shared_callback = roster["engineering"]._team_route_gate
    shared_ingress = roster["engineering"]._team_ingress_handler
    stale = roster["engineering"]
    replacement = _telegram_adapter("engineering", "replacement-token")

    assert runner._install_telegram_team_gate("engineering", replacement) is True

    assert replacement._team_route_gate is shared_callback
    assert replacement._team_ingress_handler is shared_ingress
    assert stale._team_route_gate is None
    assert stale._team_ingress_handler is None
    assert unrelated._team_route_gate is unrelated_gate
    assert unrelated._team_ingress_handler is unrelated_ingress
    assert runner._telegram_team_gate_adapters["engineering"] is replacement


def test_replacement_ingress_install_failure_rolls_back_both_callbacks():
    runner, roster = _runner()
    assert runner._validate_and_install_telegram_team_runtime() is True
    stale = roster["engineering"]
    stale_gate = stale._team_route_gate
    stale_ingress = stale._team_ingress_handler
    replacement = _telegram_adapter("engineering", "replacement-token")

    def fail_ingress(callback):
        if callback is not None:
            raise RuntimeError("ingress setter failed")
        replacement._team_ingress_handler = None

    replacement.set_team_ingress_handler = fail_ingress

    assert runner._install_telegram_team_gate("engineering", replacement) is False

    assert replacement._team_route_gate is None
    assert replacement._team_ingress_handler is None
    assert stale._team_route_gate is stale_gate
    assert stale._team_ingress_handler is stale_ingress
    assert runner._telegram_team_gate_adapters["engineering"] is stale
    assert runner.telegram_team_routing_enabled is False


def test_pending_shared_gate_rejects_until_complete_roster_is_activated():
    runner, roster = _runner()

    assert runner._prepare_telegram_team_runtime() is True
    for profile, adapter in roster.items():
        assert runner._install_telegram_team_gate(profile, adapter) is True

    callback = roster["default"]._team_route_gate
    assert callback(_context("default")) is False
    assert runner._validate_and_install_telegram_team_runtime() is True
    assert callback(_context("default")) is True


def test_secondary_adapter_configuration_installs_replacement_gate_before_connect():
    runner, roster = _runner()
    assert runner._validate_and_install_telegram_team_runtime() is True
    shared_callback = roster["engineering"]._team_route_gate
    shared_ingress = roster["engineering"]._team_ingress_handler
    stale = roster["engineering"]
    replacement = _telegram_adapter(
        "engineering", "replacement-token", live_username=None
    )
    _stub_profile_adapter_dependencies(runner)

    runner._configure_profile_adapter(
        replacement,
        "engineering",
        Platform.TELEGRAM,
    )

    assert replacement._team_route_gate is shared_callback
    assert replacement._team_ingress_handler is shared_ingress
    assert stale._team_route_gate is None
    assert stale._team_ingress_handler is None
    assert replacement._owner_profile == "engineering"
    assert runner._telegram_team_gate_adapters["engineering"] is replacement
    assert replacement._current_bot_username() == ""
    assert runner.telegram_team_routing_enabled is False
    assert replacement._team_route_gate(_context("default")) is False


@pytest.mark.parametrize("token", ["   ", "token-design"], ids=["empty", "duplicate"])
def test_secondary_replacement_credentials_are_revalidated_before_connect(token):
    runner, roster = _runner()
    assert runner._validate_and_install_telegram_team_runtime() is True
    stale_gate = roster["engineering"]._team_route_gate
    replacement = _telegram_adapter("engineering", token)
    _stub_profile_adapter_dependencies(runner)

    with pytest.raises(MultiplexConfigError, match="engineering"):
        runner._configure_profile_adapter(
            replacement,
            "engineering",
            Platform.TELEGRAM,
        )

    assert replacement._team_route_gate is None
    assert roster["engineering"]._team_route_gate is stale_gate


def test_primary_replacement_credentials_are_revalidated_before_connect():
    runner, roster = _runner()
    assert runner._validate_and_install_telegram_team_runtime() is True
    stale_gate = roster["default"]._team_route_gate
    replacement = _telegram_adapter("default", "token-design")

    with pytest.raises(MultiplexConfigError, match="default"):
        runner._configure_primary_telegram_team_adapter(replacement)

    assert replacement._team_route_gate is None
    assert roster["default"]._team_route_gate is stale_gate


def test_member_outage_suspends_all_gates_and_valid_replacement_can_reactivate():
    runner, roster = _runner()
    assert runner._validate_and_install_telegram_team_runtime() is True
    dispatcher = runner._telegram_team_dispatcher
    shared_callback = roster["default"]._team_route_gate

    runner._remove_telegram_team_gate(roster["engineering"])

    assert runner.telegram_team_routing_enabled is False
    assert runner.telegram_team_profiles == ()
    assert runner._telegram_team_config is not None
    assert roster["engineering"]._team_route_gate is None
    assert roster["engineering"]._team_ingress_handler is None
    assert roster["default"]._team_route_gate is shared_callback
    assert roster["design"]._team_route_gate is shared_callback
    assert callable(roster["default"]._team_ingress_handler)
    assert (
        roster["default"]._team_ingress_handler
        is roster["design"]._team_ingress_handler
    )
    assert shared_callback(_context("default")) is False
    assert shared_callback(_context("design")) is False

    replacement = _telegram_adapter("engineering", "replacement-token")
    runner._profile_adapters["engineering"][Platform.TELEGRAM] = replacement
    assert runner._install_telegram_team_gate("engineering", replacement) is True
    assert runner._validate_and_install_telegram_team_runtime() is True
    assert runner._telegram_team_dispatcher is dispatcher
    current_roster = {**roster, "engineering": replacement}
    assert set(_gates(current_roster).values()) == {shared_callback}


def test_incomplete_startup_roster_can_revalidate_after_missing_member_connects():
    roster = {
        "default": _telegram_adapter("default", "token-default"),
        "engineering": _telegram_adapter("engineering", "token-engineering"),
    }
    runner, _ = _runner(adapters=roster)
    assert runner._validate_and_install_telegram_team_runtime() is False
    dispatcher = runner._telegram_team_dispatcher
    callback = roster["default"]._team_route_gate

    assert runner._telegram_team_config is not None
    assert callable(callback)
    assert callback(_context("default")) is False
    design = _telegram_adapter("design", "token-design")
    runner._profile_adapters["design"] = {Platform.TELEGRAM: design}

    assert runner._validate_and_install_telegram_team_runtime() is True
    assert runner._telegram_team_dispatcher is dispatcher
    assert roster["default"]._team_route_gate is callback
    assert design._team_route_gate is callback
    assert runner.telegram_team_profiles == ("default", "design", "engineering")


def _break_identity_freshness(
    adapter: TelegramAdapter, failure: str, secret: str
) -> None:
    if failure == "stale":
        adapter._bot_identity_checked_at = None
    elif failure == "missing-helper":
        setattr(adapter, "_bot_identity_is_fresh", None)
    else:

        def raise_with_secret():
            raise RuntimeError(secret)

        setattr(adapter, "_bot_identity_is_fresh", raise_with_secret)


def _restore_identity_freshness(adapter: TelegramAdapter, failure: str) -> None:
    if failure == "stale":
        adapter._bot_identity_checked_at = time.monotonic()
    else:
        delattr(adapter, "_bot_identity_is_fresh")


@pytest.mark.parametrize(
    "failure",
    ["stale", "missing-helper", "raising-helper"],
)
def test_activation_rejects_unverifiable_identity_freshness_without_network_or_secrets(
    failure, caplog
):
    secret = "identity-freshness-secret-never-log-6d2a"
    runner, roster = _runner()
    target = roster["engineering"]
    target.config.token = secret
    target._bot.get_me = Mock(side_effect=AssertionError("network call forbidden"))
    _break_identity_freshness(target, failure, secret)
    caplog.set_level(logging.WARNING, logger="gateway.run")

    assert runner._validate_and_install_telegram_team_runtime() is False

    callbacks = _gates(roster)
    assert runner.telegram_team_routing_enabled is False
    assert all(
        callback(_context(profile)) is False for profile, callback in callbacks.items()
    )
    target._bot.get_me.assert_not_called()
    assert secret not in caplog.text


@pytest.mark.parametrize(
    "failure",
    ["stale", "missing-helper", "raising-helper"],
)
def test_post_activation_identity_failure_rejects_all_gates_then_freshness_restores(
    failure,
):
    secret = "post-activation-identity-secret-never-log-a91e"
    runner, roster = _runner()
    assert runner._validate_and_install_telegram_team_runtime() is True
    dispatcher = runner._telegram_team_dispatcher
    shared_callback = roster["default"]._team_route_gate
    target = roster["design"]
    target._bot.get_me = Mock(side_effect=AssertionError("network call forbidden"))
    _break_identity_freshness(target, failure, secret)

    decisions = {
        profile: adapter._team_route_gate(_context(profile))
        for profile, adapter in roster.items()
    }

    assert decisions == {profile: False for profile in roster}
    assert runner.telegram_team_routing_enabled is False
    assert all(
        adapter._team_route_gate is shared_callback for adapter in roster.values()
    )
    target._bot.get_me.assert_not_called()

    _restore_identity_freshness(target, failure)

    assert runner._validate_and_install_telegram_team_runtime() is True
    assert runner._telegram_team_dispatcher is dispatcher
    assert all(
        adapter._team_route_gate is shared_callback for adapter in roster.values()
    )
    assert shared_callback(_context("default")) is True


@pytest.mark.parametrize(
    ("profile", "live_username", "observed_label"),
    [
        ("engineering", None, "missing"),
        ("design", "Unexpected_Bot", "unexpected_bot"),
    ],
    ids=["missing-live-username", "mismatched-live-username"],
)
def test_live_username_failure_suspends_all_gates_then_corrected_identity_reactivates(
    profile, live_username, observed_label, caplog
):
    secret = "live-identity-token-sentinel-never-log-91a2"
    roster = {
        "default": _telegram_adapter("default", secret),
        "engineering": _telegram_adapter("engineering", "token-engineering"),
        "design": _telegram_adapter("design", "token-design"),
    }
    roster[profile] = _telegram_adapter(
        profile,
        roster[profile].config.token,
        live_username=live_username,
    )
    runner, roster = _runner(adapters=roster)
    caplog.set_level(logging.WARNING, logger="gateway.run")

    assert runner._validate_and_install_telegram_team_runtime() is False

    dispatcher = runner._telegram_team_dispatcher
    callbacks = _gates(roster)
    shared_callback = callbacks["default"]
    assert all(callback is shared_callback for callback in callbacks.values())
    assert all(
        callback(_context(owner_profile)) is False
        for owner_profile, callback in callbacks.items()
    )
    assert profile in caplog.text
    assert runner._telegram_team_config.members[profile] in caplog.text
    assert observed_label in caplog.text
    assert secret not in caplog.text

    corrected = _LIVE_USERNAMES[profile].lower()
    roster[profile]._bot_username_observed = corrected
    roster[profile]._bot.username = corrected

    assert runner._validate_and_install_telegram_team_runtime() is True
    assert runner._telegram_team_dispatcher is dispatcher
    assert set(_gates(roster).values()) == {shared_callback}


def test_post_activation_username_rename_makes_every_team_gate_fail_closed():
    runner, roster = _runner()
    assert runner._validate_and_install_telegram_team_runtime() is True
    shared_callback = roster["default"]._team_route_gate
    runner._telegram_team_dispatcher.claim_ingress = Mock(
        side_effect=AssertionError("identity failure must not claim or reroute")
    )

    roster["design"]._bot_username_observed = "renamed_design_bot"
    roster["design"]._bot.username = "renamed_design_bot"

    decisions = {
        profile: adapter._team_route_gate(_context(profile))
        for profile, adapter in roster.items()
    }

    assert decisions == {profile: False for profile in roster}
    assert runner.telegram_team_routing_enabled is False
    assert all(
        adapter._team_route_gate is shared_callback for adapter in roster.values()
    )
    runner._telegram_team_dispatcher.claim_ingress.assert_not_called()


@pytest.mark.asyncio
async def test_secondary_reconnect_rejects_primary_token_inside_dynamic_profile_scope(
    monkeypatch, tmp_path
):
    import gateway.config as gateway_config
    import hermes_cli.profiles as profiles

    hermes_home = tmp_path / "hermes-home"
    profile_home = hermes_home / "profiles" / "design"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    runner, roster = _runner()
    del runner._active_profile_name
    assert runner._active_profile_name() == "default"
    assert runner._validate_and_install_telegram_team_runtime() is True
    old_adapter = roster["design"]
    old_gate = old_adapter._team_route_gate
    replacement = _telegram_adapter(
        "design",
        roster["default"].config.token,
        live_username="Virgil_Bot",
    )
    runner._running = True
    runner._profile_failed_platforms = {}
    monkeypatch.setattr(profiles, "get_profile_dir", lambda _profile: profile_home)
    profile_config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token=roster["default"].config.token,
            )
        }
    )
    monkeypatch.setattr(gateway_config, "load_gateway_config", lambda: profile_config)
    scoped_profiles = []

    def create_adapter(platform, config):
        scoped_profiles.append(runner._active_profile_name())
        return replacement

    runner._create_adapter = create_adapter
    _stub_profile_adapter_dependencies(runner)
    runner._connect_adapter_with_timeout = AsyncMock(return_value=True)
    runner._safe_adapter_disconnect = AsyncMock()

    async def stop_after_refusal(_delay):
        runner._running = False

    monkeypatch.setattr(asyncio, "sleep", stop_after_refusal)

    await runner._run_secondary_profile_reconnect("design", Platform.TELEGRAM)

    assert scoped_profiles == ["design"]
    runner._connect_adapter_with_timeout.assert_not_awaited()
    assert runner._profile_adapters["design"][Platform.TELEGRAM] is old_adapter
    assert runner._telegram_team_gate_adapters["design"] is old_adapter
    assert old_adapter._team_route_gate is old_gate
    assert replacement._team_route_gate is None
    assert runner.telegram_team_routing_enabled is True


@pytest.mark.asyncio
async def test_secondary_reconnect_revalidates_after_registering_adapter(
    monkeypatch, tmp_path
):
    import gateway.config as gateway_config
    import hermes_cli.profiles as profiles

    runner, roster = _runner()
    replacement = _telegram_adapter(
        "design", "replacement-token", live_username="Unexpected_Bot"
    )
    runner._running = True
    runner._profile_adapters["design"] = {}
    runner._profile_failed_platforms = {}
    monkeypatch.setattr(profiles, "get_profile_dir", lambda _profile: tmp_path)
    profile_config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="replacement-token",
            )
        }
    )
    monkeypatch.setattr(gateway_config, "load_gateway_config", lambda: profile_config)
    runner._create_adapter = lambda _platform, _config: replacement
    _stub_profile_adapter_dependencies(runner)
    runner._connect_adapter_with_timeout = AsyncMock(return_value=True)
    runner._sync_voice_mode_state_to_adapter = Mock()

    await runner._run_secondary_profile_reconnect("design", Platform.TELEGRAM)

    assert runner._profile_adapters["design"][Platform.TELEGRAM] is replacement
    current_roster = {**roster, "design": replacement}
    callbacks = _gates(current_roster)
    shared_callback = callbacks["default"]
    assert runner.telegram_team_routing_enabled is False
    assert all(callback is shared_callback for callback in callbacks.values())
    assert all(
        callback(_context(profile)) is False for profile, callback in callbacks.items()
    )


def test_unrelated_adapter_is_never_modified():
    runner, roster = _runner()
    unrelated_gate = object()
    unrelated = SimpleNamespace(_team_route_gate=unrelated_gate)
    runner._profile_adapters["observer"] = {Platform.DISCORD: unrelated}

    assert runner._validate_and_install_telegram_team_runtime() is True
    assert unrelated._team_route_gate is unrelated_gate


def test_validation_diagnostics_never_include_raw_tokens(caplog):
    secret = "raw-token-sentinel-do-not-log-7f3c"
    roster = {
        "default": _telegram_adapter("default", secret),
        "engineering": _telegram_adapter("engineering", secret),
        "design": _telegram_adapter("design", "different-token"),
    }
    runner, _ = _runner(adapters=roster)
    caplog.set_level(logging.WARNING, logger="gateway.run")

    assert runner._validate_and_install_telegram_team_runtime() is False

    assert secret not in caplog.text
    assert "default" in caplog.text
    assert "engineering" in caplog.text
