"""Multiplex runtime validation and shared Telegram team-gate wiring."""

from __future__ import annotations

import logging
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


def _telegram_adapter(profile: str, token: str) -> TelegramAdapter:
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token=token, extra={})
    adapter._team_route_gate = None
    adapter.set_owner_profile(profile)
    return adapter


def _runner(
    *,
    raw: object | None = None,
    multiplex: bool = True,
    allowlist: list[str] | None = None,
    active: str = "default",
    adapters: dict[str, Any] | None = None,
) -> tuple[GatewayRunner, dict[str, Any]]:
    if raw is None:
        raw = _team_raw()
    if allowlist is None:
        allowlist = ["engineering", "design"]
    config = GatewayConfig(
        multiplex_profiles=multiplex,
        multiplex_profile_allowlist=allowlist,
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="token-default",
                extra={"team_routing": raw},
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
    )


def _gates(roster: dict[str, Any]) -> dict[str, object]:
    return {
        profile: getattr(adapter, "_team_route_gate", None)
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


def test_valid_complete_roster_installs_one_shared_gate_and_dispatcher():
    runner, roster = _runner()

    assert runner._validate_and_install_telegram_team_runtime() is True

    assert runner.telegram_team_routing_enabled is True
    assert runner.telegram_team_profiles == ("default", "design", "engineering")
    dispatcher = runner._telegram_team_dispatcher
    assert isinstance(dispatcher, TelegramTeamDispatcher)
    callbacks = list(_gates(roster).values())
    assert all(callable(callback) for callback in callbacks)
    assert len({id(callback) for callback in callbacks}) == 1


@pytest.mark.parametrize(
    ("multiplex", "allowlist", "missing_profile"),
    [
        (False, ["engineering", "design"], None),
        (True, ["engineering"], None),
        (True, ["engineering", "design"], "design"),
    ],
    ids=["multiplex-off", "member-not-explicitly-allowlisted", "member-not-served"],
)
def test_incomplete_multiplex_runtime_preserves_legacy_absent_gate(
    multiplex, allowlist, missing_profile
):
    roster = {
        "default": _telegram_adapter("default", "token-default"),
        "engineering": _telegram_adapter("engineering", "token-engineering"),
        "design": _telegram_adapter("design", "token-design"),
    }
    if missing_profile:
        roster.pop(missing_profile)
    runner, roster = _runner(
        multiplex=multiplex,
        allowlist=allowlist,
        adapters=roster,
    )

    assert runner._validate_and_install_telegram_team_runtime() is False

    assert runner.telegram_team_routing_enabled is False
    assert runner.telegram_team_profiles == ()
    assert all(gate is None for gate in _gates(roster).values())


def test_team_members_require_an_explicit_allowlist_even_when_multiplex_serves_all():
    runner, roster = _runner()
    runner.config.multiplex_profile_allowlist = None

    assert runner._validate_and_install_telegram_team_runtime() is False
    assert runner.telegram_team_routing_enabled is False
    assert all(gate is None for gate in _gates(roster).values())


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
    assert all(gate is None for gate in _gates(roster).values())


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
    assert all(gate is None for gate in _gates(roster).values())


@pytest.mark.parametrize(
    "raw",
    [
        {"coordinator_profile": "default"},
        {
            **_team_raw(),
            "coordinator_username": "Other_Bot",
        },
    ],
    ids=["malformed", "coordinator-username-mismatch"],
)
def test_invalid_team_config_leaves_legacy_gate_absent(raw):
    runner, roster = _runner(raw=raw)

    assert runner._validate_and_install_telegram_team_runtime() is False
    assert runner.telegram_team_routing_enabled is False
    assert all(gate is None for gate in _gates(roster).values())


def test_coordinator_must_be_active_profile():
    runner, roster = _runner(active="engineering")

    assert runner._validate_and_install_telegram_team_runtime() is False
    assert runner.telegram_team_routing_enabled is False
    assert all(gate is None for gate in _gates(roster).values())


def test_gate_installation_rolls_back_atomically_when_one_adapter_rejects_callback():
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
    assert all(gate is None for gate in _gates(roster).values())


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

    assert decisions == {
        profile: profile == expected_owner for profile in roster
    }


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


def test_replacement_receives_shared_gate_and_stale_gate_is_removed():
    runner, roster = _runner()
    unrelated_gate = object()
    unrelated = SimpleNamespace(_team_route_gate=unrelated_gate)
    assert runner._validate_and_install_telegram_team_runtime() is True
    shared_callback = roster["engineering"]._team_route_gate
    stale = roster["engineering"]
    replacement = _telegram_adapter("engineering", "replacement-token")

    assert runner._install_telegram_team_gate("engineering", replacement) is True

    assert replacement._team_route_gate is shared_callback
    assert stale._team_route_gate is None
    assert unrelated._team_route_gate is unrelated_gate
    assert runner._telegram_team_gate_adapters["engineering"] is replacement


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
    stale = roster["engineering"]
    replacement = _telegram_adapter("engineering", "replacement-token")
    _stub_profile_adapter_dependencies(runner)

    runner._configure_profile_adapter(
        replacement,
        "engineering",
        Platform.TELEGRAM,
    )

    assert replacement._team_route_gate is shared_callback
    assert stale._team_route_gate is None
    assert replacement._owner_profile == "engineering"
    assert runner._telegram_team_gate_adapters["engineering"] is replacement


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

    runner._remove_telegram_team_gate(roster["engineering"])

    assert runner.telegram_team_routing_enabled is False
    assert runner.telegram_team_profiles == ()
    assert runner._telegram_team_config is not None
    assert all(gate is None for gate in _gates(roster).values())

    replacement = _telegram_adapter("engineering", "replacement-token")
    runner._profile_adapters["engineering"][Platform.TELEGRAM] = replacement
    assert runner._install_telegram_team_gate("engineering", replacement) is True
    assert runner._validate_and_install_telegram_team_runtime() is True
    assert runner._telegram_team_dispatcher is dispatcher
    current_roster = {**roster, "engineering": replacement}
    assert all(callable(gate) for gate in _gates(current_roster).values())


def test_incomplete_startup_roster_can_revalidate_after_missing_member_connects():
    roster = {
        "default": _telegram_adapter("default", "token-default"),
        "engineering": _telegram_adapter("engineering", "token-engineering"),
    }
    runner, _ = _runner(adapters=roster)
    assert runner._validate_and_install_telegram_team_runtime() is False
    dispatcher = runner._telegram_team_dispatcher

    assert runner._telegram_team_config is not None
    design = _telegram_adapter("design", "token-design")
    runner._profile_adapters["design"] = {Platform.TELEGRAM: design}

    assert runner._validate_and_install_telegram_team_runtime() is True
    assert runner._telegram_team_dispatcher is dispatcher
    assert runner.telegram_team_profiles == ("default", "design", "engineering")


@pytest.mark.asyncio
async def test_secondary_reconnect_revalidates_after_registering_adapter(
    monkeypatch, tmp_path
):
    import gateway.config as gateway_config
    import hermes_cli.profiles as profiles

    runner, _ = _runner()
    replacement = _telegram_adapter("design", "replacement-token")
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
    runner._configure_profile_adapter = Mock()
    runner._connect_adapter_with_timeout = AsyncMock(return_value=True)
    runner._validate_and_install_telegram_team_runtime = Mock(return_value=True)
    runner._sync_voice_mode_state_to_adapter = Mock()

    await runner._run_secondary_profile_reconnect("design", Platform.TELEGRAM)

    assert runner._profile_adapters["design"][Platform.TELEGRAM] is replacement
    runner._validate_and_install_telegram_team_runtime.assert_called_once_with()


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
