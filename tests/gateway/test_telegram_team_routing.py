"""Deterministic Telegram team-routing configuration and resolver tests."""

from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError

import pytest


def _routing_module():
    try:
        return importlib.import_module("gateway.telegram_team_routing")
    except ModuleNotFoundError:
        pytest.fail("gateway.telegram_team_routing is not implemented", pytrace=False)


def _valid_raw() -> dict[str, object]:
    return {
        "coordinator_profile": "default",
        "coordinator_username": "@Ace_Bot",
        "members": {
            "default": "ACE_BOT",
            "engineering": "Woz_Bot",
            "design-team": "Virgil_Bot",
        },
        "allowed_chats": [-1001234567890, "-987654321"],
    }


def _config():
    module = _routing_module()
    config = module.TelegramTeamConfig.from_raw(_valid_raw())
    assert config is not None
    return config


def test_team_config_normalizes_valid_values_and_is_immutable():
    module = _routing_module()

    config = module.TelegramTeamConfig.from_raw(_valid_raw())

    assert config is not None
    assert config.coordinator_profile == "default"
    assert config.coordinator_username == "ace_bot"
    assert dict(config.members) == {
        "default": "ace_bot",
        "engineering": "woz_bot",
        "design-team": "virgil_bot",
    }
    assert config.allowed_chats == frozenset({"-1001234567890", "-987654321"})
    with pytest.raises(FrozenInstanceError):
        config.coordinator_profile = "engineering"
    with pytest.raises(TypeError):
        config.members["default"] = "other_bot"


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        {**_valid_raw(), "unknown": True},
        {key: value for key, value in _valid_raw().items() if key != "coordinator_profile"},
        {key: value for key, value in _valid_raw().items() if key != "coordinator_username"},
        {key: value for key, value in _valid_raw().items() if key != "members"},
        {key: value for key, value in _valid_raw().items() if key != "allowed_chats"},
        {**_valid_raw(), "members": []},
        {**_valid_raw(), "allowed_chats": {}},
    ],
)
def test_team_config_rejects_unknown_missing_and_non_collection_fields(raw):
    module = _routing_module()

    assert module.TelegramTeamConfig.from_raw(raw) is None


@pytest.mark.parametrize(
    "profile",
    ["", "-engineering", "engineering!", " engineering", "a" * 65, 123],
)
def test_team_config_rejects_invalid_profile_names(profile):
    module = _routing_module()
    raw = _valid_raw()
    raw["members"] = {profile: "Woz_Bot", "default": "ACE_BOT"}

    assert module.TelegramTeamConfig.from_raw(raw) is None


def test_team_config_rejects_absent_coordinator_profile():
    module = _routing_module()
    raw = _valid_raw()
    raw["members"] = {"engineering": "Woz_Bot"}

    assert module.TelegramTeamConfig.from_raw(raw) is None


def test_team_config_rejects_coordinator_username_mismatch():
    module = _routing_module()
    raw = _valid_raw()
    raw["coordinator_username"] = "Other_Bot"

    assert module.TelegramTeamConfig.from_raw(raw) is None


@pytest.mark.parametrize(
    "username",
    ["", "abcd", "a" * 33, "bad-name", "@bad@name", " name_bot", 123],
)
def test_team_config_rejects_malformed_usernames(username):
    module = _routing_module()
    raw = _valid_raw()
    raw["members"] = {"default": "ACE_BOT", "engineering": username}

    assert module.TelegramTeamConfig.from_raw(raw) is None


def test_team_config_rejects_duplicate_usernames_after_normalization():
    module = _routing_module()
    raw = _valid_raw()
    raw["members"] = {
        "default": "ACE_BOT",
        "engineering": "@ace_bot",
    }

    assert module.TelegramTeamConfig.from_raw(raw) is None


@pytest.mark.parametrize(
    "allowed_chats",
    [
        [0],
        [1],
        ["0"],
        ["123"],
        ["-1.5"],
        [" -100"],
        [True],
        [-1.5],
        [None],
    ],
)
def test_team_config_rejects_invalid_allowed_group_ids(allowed_chats):
    module = _routing_module()
    raw = _valid_raw()
    raw["allowed_chats"] = allowed_chats

    assert module.TelegramTeamConfig.from_raw(raw) is None


def test_reply_to_team_bot_wins_over_mentions():
    module = _routing_module()

    decision = module.resolve_addressed_owner(
        mentions={"@Virgil_Bot"},
        reply_author_username="@WOZ_BOT",
        config=_config(),
        chat_id="-1001234567890",
    )

    assert decision == module.TeamRouteDecision(
        owner_profile="engineering",
        reason="reply_to_team_bot",
        accepted=True,
    )


def test_exactly_one_team_mention_owns_the_message():
    module = _routing_module()

    decision = module.resolve_addressed_owner(
        mentions={"@VIRGIL_BOT", "@human_handle", "foreign_bot"},
        reply_author_username="human_handle",
        config=_config(),
        chat_id="-1001234567890",
    )

    assert decision == module.TeamRouteDecision(
        owner_profile="design-team",
        reason="single_team_mention",
        accepted=True,
    )


def test_multiple_team_mentions_route_to_coordinator():
    module = _routing_module()

    decision = module.resolve_addressed_owner(
        mentions={"ace_bot", "@WOZ_BOT", "@someone_else"},
        reply_author_username=None,
        config=_config(),
        chat_id="-987654321",
    )

    assert decision == module.TeamRouteDecision(
        owner_profile="default",
        reason="multiple_team_mentions",
        accepted=True,
    )


def test_unaddressed_ingress_routes_to_coordinator():
    module = _routing_module()

    decision = module.resolve_addressed_owner(
        mentions={"@human_handle", "@foreign_bot"},
        reply_author_username="foreign_bot",
        config=_config(),
        chat_id="-987654321",
    )

    assert decision == module.TeamRouteDecision(
        owner_profile="default",
        reason="unaddressed_ingress",
        accepted=True,
    )


def test_outside_allowed_chats_is_not_accepted():
    module = _routing_module()

    decision = module.resolve_addressed_owner(
        mentions={"@WOZ_BOT"},
        reply_author_username="virgil_bot",
        config=_config(),
        chat_id="-555",
    )

    assert decision == module.TeamRouteDecision(
        owner_profile=None,
        reason="chat_not_allowed",
        accepted=False,
    )


def test_nested_team_routing_bridges_into_telegram_platform_extra(monkeypatch, tmp_path):
    from gateway.config import Platform, load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "telegram:\n"
        "  team_routing:\n"
        "    coordinator_profile: default\n"
        "    coordinator_username: Ace_Bot\n"
        "    members:\n"
        "      default: Ace_Bot\n"
        "      engineering: Woz_Bot\n"
        "    allowed_chats:\n"
        "      - -1001234567890\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config = load_gateway_config()

    assert config is not None
    telegram = config.platforms[Platform.TELEGRAM]
    assert telegram.extra["team_routing"] == {
        "coordinator_profile": "default",
        "coordinator_username": "Ace_Bot",
        "members": {
            "default": "Ace_Bot",
            "engineering": "Woz_Bot",
        },
        "allowed_chats": [-1001234567890],
    }
