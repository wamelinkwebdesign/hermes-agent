"""Deterministic Telegram team-routing configuration and resolver tests."""

from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor
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


def _dispatcher(**kwargs):
    return _routing_module().TelegramTeamDispatcher(**kwargs)


def test_ingress_claim_is_process_local_and_reports_the_first_adapter():
    module = _routing_module()
    dispatcher = _dispatcher()

    first = dispatcher.claim_ingress(-1001234567890, 501, "engineering")
    duplicate = dispatcher.claim_ingress("-1001234567890", "501", "design-team")

    assert first == module.IngressClaim(
        claimed=True,
        duplicate=False,
        adapter_profile="engineering",
    )
    assert duplicate == module.IngressClaim(
        claimed=False,
        duplicate=True,
        adapter_profile="engineering",
    )
    with pytest.raises(FrozenInstanceError):
        first.claimed = False


def test_concurrent_adapter_copies_produce_exactly_one_ingress_claim():
    dispatcher = _dispatcher()

    with ThreadPoolExecutor(max_workers=16) as pool:
        claims = list(
            pool.map(
                lambda profile: dispatcher.claim_ingress(-1001, 77, profile),
                [f"adapter-{index}" for index in range(32)],
            )
        )

    assert sum(claim.claimed for claim in claims) == 1
    assert sum(claim.duplicate for claim in claims) == 31
    assert {claim.adapter_profile for claim in claims} == {claims[0].adapter_profile}


@pytest.mark.parametrize(
    ("chat_id", "message_id", "profile"),
    [
        (0, 1, "engineering"),
        ("1001", 1, "engineering"),
        (-1001, 0, "engineering"),
        (-1001, "01", "engineering"),
        (-1001, True, "engineering"),
        (-1001, 1, "bad/profile"),
        (-1001, 1, ""),
    ],
)
def test_invalid_ingress_claims_fail_closed_without_consuming_capacity(
    chat_id,
    message_id,
    profile,
):
    module = _routing_module()
    dispatcher = _dispatcher(max_claims=1)

    rejected = dispatcher.claim_ingress(chat_id, message_id, profile)
    accepted = dispatcher.claim_ingress(-1001, 1, "engineering")

    assert rejected == module.IngressClaim(
        claimed=False,
        duplicate=False,
        adapter_profile=None,
    )
    assert accepted.claimed is True


def test_ingress_claims_evict_the_oldest_entry_deterministically():
    dispatcher = _dispatcher(max_claims=2)

    assert dispatcher.claim_ingress(-1001, 1, "default").claimed is True
    assert dispatcher.claim_ingress(-1001, 2, "default").claimed is True
    assert dispatcher.claim_ingress(-1001, 3, "default").claimed is True

    assert dispatcher.claim_ingress(-1001, 2, "engineering").duplicate is True
    assert dispatcher.claim_ingress(-1001, 1, "engineering").claimed is True


def test_root_and_arbitrarily_deep_human_aliases_keep_one_immutable_owner():
    module = _routing_module()
    dispatcher = _dispatcher()

    assert dispatcher.record_root(-1001, 100, "engineering") is True
    assert dispatcher.record_inbound_alias(-1001, 101, 100) is True
    assert dispatcher.record_inbound_alias(-1001, 102, 101) is True
    assert dispatcher.record_inbound_alias(-1001, 103, 102) is True

    ownership = module.RootOwnership(
        root_message_id="100",
        owner_profile="engineering",
    )
    assert dispatcher.resolve_reply_owner(-1001, 100) == ownership
    assert dispatcher.resolve_reply_owner(-1001, 103) == ownership
    with pytest.raises(FrozenInstanceError):
        ownership.owner_profile = "design-team"


def test_root_rebind_conflict_fails_without_mutating_the_original_owner():
    module = _routing_module()
    dispatcher = _dispatcher()

    assert dispatcher.record_root(-1001, 100, "engineering") is True
    assert dispatcher.record_root(-1001, 100, "design-team") is False

    assert dispatcher.resolve_reply_owner(-1001, 100) == module.RootOwnership(
        root_message_id="100",
        owner_profile="engineering",
    )


def test_inbound_alias_requires_a_resolved_parent_and_rejects_cross_root_rebind():
    module = _routing_module()
    dispatcher = _dispatcher()
    assert dispatcher.record_root(-1001, 100, "engineering") is True
    assert dispatcher.record_root(-1001, 200, "design-team") is True

    assert dispatcher.record_inbound_alias(-1001, 101, 999) is False
    assert dispatcher.resolve_reply_owner(-1001, 101) is None
    assert dispatcher.record_inbound_alias(-1001, 101, 100) is True
    assert dispatcher.record_inbound_alias(-1001, 101, 200) is False

    assert dispatcher.resolve_reply_owner(-1001, 101) == module.RootOwnership(
        root_message_id="100",
        owner_profile="engineering",
    )


def test_outbound_aliases_cover_preview_final_and_continuation_messages():
    module = _routing_module()
    dispatcher = _dispatcher()
    assert dispatcher.record_root(-1001, 100, "engineering") is True

    assert dispatcher.record_outbound_alias(-1001, [301, "302", 303], 100) is True

    expected = module.RootOwnership(root_message_id="100", owner_profile="engineering")
    assert dispatcher.resolve_reply_owner(-1001, 301) == expected
    assert dispatcher.resolve_reply_owner(-1001, 302) == expected
    assert dispatcher.resolve_reply_owner(-1001, 303) == expected


def test_outbound_alias_batch_is_atomic_on_invalid_id_or_conflict():
    module = _routing_module()
    dispatcher = _dispatcher()
    assert dispatcher.record_root(-1001, 100, "engineering") is True
    assert dispatcher.record_root(-1001, 200, "design-team") is True
    assert dispatcher.record_outbound_alias(-1001, [400], 200) is True

    assert dispatcher.record_outbound_alias(-1001, [301, 0, 302], 100) is False
    assert dispatcher.record_outbound_alias(-1001, [303, 400, 304], 100) is False

    assert dispatcher.resolve_reply_owner(-1001, 301) is None
    assert dispatcher.resolve_reply_owner(-1001, 303) is None
    assert dispatcher.resolve_reply_owner(-1001, 400) == module.RootOwnership(
        root_message_id="200",
        owner_profile="design-team",
    )


def test_aliases_are_chat_scoped_and_invalid_lookups_fail_closed():
    dispatcher = _dispatcher()
    assert dispatcher.record_root(-1001, 100, "engineering") is True

    assert dispatcher.resolve_reply_owner(-2002, 100) is None
    assert dispatcher.resolve_reply_owner("not-a-group", 100) is None
    assert dispatcher.resolve_reply_owner(-1001, 0) is None


def test_alias_bound_evicts_whole_oldest_root_family_without_dangling_aliases():
    module = _routing_module()
    dispatcher = _dispatcher(max_aliases=3)
    assert dispatcher.record_root(-1001, 100, "engineering") is True
    assert dispatcher.record_inbound_alias(-1001, 101, 100) is True
    assert dispatcher.record_root(-1001, 200, "design-team") is True

    assert dispatcher.record_outbound_alias(-1001, [201], 200) is True

    assert dispatcher.resolve_reply_owner(-1001, 100) is None
    assert dispatcher.resolve_reply_owner(-1001, 101) is None
    assert dispatcher.resolve_reply_owner(-1001, 200) == module.RootOwnership(
        root_message_id="200",
        owner_profile="design-team",
    )
    assert dispatcher.resolve_reply_owner(-1001, 201) == module.RootOwnership(
        root_message_id="200",
        owner_profile="design-team",
    )


def test_alias_batch_larger_than_capacity_fails_without_evicting_existing_roots():
    module = _routing_module()
    dispatcher = _dispatcher(max_aliases=2)
    assert dispatcher.record_root(-1001, 100, "engineering") is True

    assert dispatcher.record_outbound_alias(-1001, [301, 302], 100) is False

    assert dispatcher.resolve_reply_owner(-1001, 100) == module.RootOwnership(
        root_message_id="100",
        owner_profile="engineering",
    )
    assert dispatcher.resolve_reply_owner(-1001, 301) is None
