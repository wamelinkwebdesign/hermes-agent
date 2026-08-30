"""Deterministic Telegram team-routing configuration and resolver tests."""

from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from types import ModuleType

import pytest


def _routing_module() -> ModuleType:
    try:
        return importlib.import_module("gateway.telegram_team_routing")
    except ModuleNotFoundError:
        pytest.fail("gateway.telegram_team_routing is not implemented", pytrace=False)
        raise AssertionError("unreachable")


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
        {
            key: value
            for key, value in _valid_raw().items()
            if key != "coordinator_profile"
        },
        {
            key: value
            for key, value in _valid_raw().items()
            if key != "coordinator_username"
        },
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


def test_team_config_accepts_signed_64_bit_group_id_boundaries():
    module = _routing_module()
    raw = _valid_raw()
    raw["allowed_chats"] = [-(2**63), "-1"]

    config = module.TelegramTeamConfig.from_raw(raw)

    assert config is not None
    assert config.allowed_chats == frozenset({str(-(2**63)), "-1"})


@pytest.mark.parametrize("chat_id", [-(2**63) - 1, str(-(2**63) - 1)])
def test_team_config_rejects_group_ids_below_signed_64_bit_minimum(chat_id):
    module = _routing_module()
    raw = _valid_raw()
    raw["allowed_chats"] = [-1001, chat_id]

    assert module.TelegramTeamConfig.from_raw(raw) is None


def test_team_config_rejects_overlong_group_id_atomically():
    module = _routing_module()
    raw = _valid_raw()
    raw["allowed_chats"] = [-1001, "-" + "9" * 5000]

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


def test_nested_team_routing_bridges_into_telegram_platform_extra(
    monkeypatch, tmp_path
):
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


def _grant_arguments(adapter_capability):
    return {
        "chat_id": -1001,
        "root_message_id": 100,
        "owner_profile": "engineering",
        "current_message_id": 101,
        "constituent_message_ids": (101, 102),
        "target_adapter_capability": adapter_capability,
    }


def test_routed_authorization_grant_is_exact_and_single_use():
    dispatcher = _dispatcher()
    adapter_capability = object()
    arguments = _grant_arguments(adapter_capability)

    grant = dispatcher.issue_routed_authorization(**arguments)

    assert grant is not None
    assert "100" not in repr(grant)
    assert dispatcher.consume_routed_authorization(grant, **arguments) is True
    assert dispatcher.consume_routed_authorization(grant, **arguments) is False


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("chat_id", -2002),
        ("root_message_id", 200),
        ("owner_profile", "design-team"),
        ("current_message_id", 103),
        ("constituent_message_ids", (101, 103)),
        ("target_adapter_capability", object()),
    ],
)
def test_wrong_routed_authorization_binding_does_not_consume_valid_grant(
    field,
    wrong_value,
):
    dispatcher = _dispatcher()
    adapter_capability = object()
    arguments = _grant_arguments(adapter_capability)
    grant = dispatcher.issue_routed_authorization(**arguments)
    assert grant is not None
    wrong_arguments = {**arguments, field: wrong_value}

    assert dispatcher.consume_routed_authorization(grant, **wrong_arguments) is False
    assert dispatcher.consume_routed_authorization(grant, **arguments) is True


def test_routed_authorization_grants_are_bounded_and_invalid_issue_does_not_evict():
    dispatcher = _dispatcher(max_routed_grants=2)
    capability = object()
    first_args = _grant_arguments(capability)
    second_args = {
        **first_args,
        "current_message_id": 103,
        "constituent_message_ids": (103,),
    }
    third_args = {
        **first_args,
        "current_message_id": 104,
        "constituent_message_ids": (104,),
    }
    first = dispatcher.issue_routed_authorization(**first_args)
    second = dispatcher.issue_routed_authorization(**second_args)
    assert first is not None
    assert second is not None

    assert (
        dispatcher.issue_routed_authorization(**{
            **third_args,
            "constituent_message_ids": (104, 0),
        })
        is None
    )
    assert dispatcher.consume_routed_authorization(first, **first_args) is True

    replacement_first = dispatcher.issue_routed_authorization(**first_args)
    third = dispatcher.issue_routed_authorization(**third_args)
    assert replacement_first is not None
    assert third is not None
    assert dispatcher.consume_routed_authorization(second, **second_args) is False
    assert (
        dispatcher.consume_routed_authorization(replacement_first, **first_args) is True
    )
    assert dispatcher.consume_routed_authorization(third, **third_args) is True


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


def test_overlong_ingress_identifiers_fail_closed_without_exceptions():
    module = _routing_module()
    dispatcher = _dispatcher()
    rejected = module.IngressClaim(
        claimed=False,
        duplicate=False,
        adapter_profile=None,
    )

    assert dispatcher.claim_ingress("-" + "9" * 5000, 1, "engineering") == rejected
    assert dispatcher.claim_ingress(-1001, "9" * 10_000, "engineering") == rejected


@pytest.mark.parametrize(
    ("chat_id", "message_id"),
    [
        (-(2**63), 1),
        (str(-(2**63)), "1"),
        (-1, 2**63 - 1),
        ("-1", str(2**63 - 1)),
    ],
)
def test_ingress_claim_accepts_signed_64_bit_identifier_boundaries(chat_id, message_id):
    dispatcher_claim = _dispatcher().claim_ingress(
        chat_id,
        message_id,
        "engineering",
    )

    assert dispatcher_claim.claimed is True


@pytest.mark.parametrize(
    ("chat_id", "message_id"),
    [
        (-(2**63) - 1, 1),
        (str(-(2**63) - 1), "1"),
        (-1, 2**63),
        ("-1", str(2**63)),
    ],
)
def test_ingress_claim_rejects_identifiers_outside_signed_64_bit_range(
    chat_id,
    message_id,
):
    module = _routing_module()

    assert _dispatcher().claim_ingress(chat_id, message_id, "engineering") == (
        module.IngressClaim(claimed=False, duplicate=False, adapter_profile=None)
    )


def test_overlong_ingress_claim_does_not_consume_or_reorder_capacity():
    dispatcher = _dispatcher(max_claims=2)
    assert dispatcher.claim_ingress(-1001, 1, "default").claimed is True

    assert dispatcher.claim_ingress(-1001, "9" * 10_000, "default").claimed is False
    assert dispatcher.claim_ingress(-1001, 2, "default").claimed is True
    assert dispatcher.claim_ingress(-1001, 1, "engineering").duplicate is True

    assert dispatcher.claim_ingress(-1001, 3, "default").claimed is True
    assert dispatcher.claim_ingress(-1001, 1, "engineering").claimed is True


def test_ingress_claims_evict_the_oldest_entry_deterministically():
    dispatcher = _dispatcher(max_claims=2)

    assert dispatcher.claim_ingress(-1001, 1, "default").claimed is True
    assert dispatcher.claim_ingress(-1001, 2, "default").claimed is True
    assert dispatcher.claim_ingress(-1001, 3, "default").claimed is True

    assert dispatcher.claim_ingress(-1001, 2, "engineering").duplicate is True
    assert dispatcher.claim_ingress(-1001, 1, "engineering").claimed is True


def test_ingress_batch_reservation_blocks_duplicates_then_release_allows_retry():
    dispatcher = _dispatcher()

    first = dispatcher.reserve_ingress_batch(-1001, [900, 901], "engineering")
    assert first.reserved is True
    assert first.duplicate is False
    assert first.adapter_profile == "engineering"
    assert first.reservation is not None

    duplicate = dispatcher.reserve_ingress_batch(-1001, [901], "design-team")
    assert duplicate.reserved is False
    assert duplicate.duplicate is True
    assert duplicate.adapter_profile == "engineering"
    assert duplicate.reservation is None

    assert dispatcher.release_ingress(first.reservation) is True
    retry = dispatcher.reserve_ingress_batch(-1001, [900, 901], "design-team")
    assert retry.reserved is True
    assert retry.adapter_profile == "design-team"
    assert retry.reservation is not None

    assert dispatcher.commit_ingress(retry.reservation) is True
    replay = dispatcher.reserve_ingress_batch(-1001, [900], "engineering")
    assert replay.reserved is False
    assert replay.duplicate is True
    assert replay.adapter_profile == "design-team"


def test_constituent_ownership_preflight_distinguishes_none_same_partial_and_conflict():
    module = _routing_module()
    dispatcher = _dispatcher()
    assert dispatcher.record_root_batch(-1001, 100, "engineering", [100, 101])
    assert dispatcher.record_root_batch(-1001, 200, "design-team", [200, 201])

    assert dispatcher.inspect_constituent_ownership(-1001, [300, 301]) == (
        module.ConstituentOwnershipResult("none")
    )
    assert dispatcher.inspect_constituent_ownership(-1001, [100, 101]) == (
        module.ConstituentOwnershipResult(
            "same",
            module.RootOwnership("100", "engineering"),
        )
    )
    assert dispatcher.inspect_constituent_ownership(-1001, [300, 101]) == (
        module.ConstituentOwnershipResult(
            "partial",
            module.RootOwnership("100", "engineering"),
        )
    )
    assert dispatcher.inspect_constituent_ownership(-1001, [101, 201]) == (
        module.ConstituentOwnershipResult("conflict")
    )


def test_constituent_ownership_preflight_rejects_malformed_and_duplicate_heavy_ids():
    module = _routing_module()
    dispatcher = _dispatcher()
    assert dispatcher.record_root(-1001, 100, "engineering")
    invalid = module.ConstituentOwnershipResult("invalid")

    assert dispatcher.inspect_constituent_ownership(-1001, [100, 0]) == invalid
    assert dispatcher.inspect_constituent_ownership(-1001, "100") == invalid
    assert dispatcher.inspect_constituent_ownership("not-a-chat", [100]) == invalid

    class DuplicateHeavy:
        def __init__(self):
            self.next_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.next_calls += 1
            if self.next_calls > module.MAX_TELEGRAM_BATCH_MESSAGE_IDS + 1:
                raise AssertionError("preflight over-consumed hostile iterable")
            return 100

    duplicate_heavy = DuplicateHeavy()
    assert dispatcher.inspect_constituent_ownership(-1001, duplicate_heavy) == invalid
    assert duplicate_heavy.next_calls == module.MAX_TELEGRAM_BATCH_MESSAGE_IDS + 1


def test_ingress_reservation_rechecks_ownership_after_unbound_preflight():
    module = _routing_module()
    dispatcher = _dispatcher()
    assert dispatcher.inspect_constituent_ownership(-1001, [300, 301]) == (
        module.ConstituentOwnershipResult("none")
    )
    assert dispatcher.record_root_batch(-1001, 100, "engineering", [100, 301])

    attempt = dispatcher.reserve_ingress_batch(-1001, [300, 301], "default")

    assert attempt.reserved is False
    assert attempt.duplicate is True
    assert attempt.adapter_profile == "engineering"
    assert attempt.reservation is None


def test_routed_ownership_verification_requires_exact_complete_binding():
    dispatcher = _dispatcher()
    assert dispatcher.record_root_batch(-1001, 100, "engineering", [100, 101])
    assert dispatcher.record_root_batch(-1001, 200, "design-team", [200, 201])

    assert dispatcher.verify_routed_ownership(
        -1001,
        100,
        "engineering",
        [100, 101],
    )
    assert not dispatcher.verify_routed_ownership(
        -1001,
        100,
        "engineering",
        [100, 300],
    )
    assert not dispatcher.verify_routed_ownership(
        -1001,
        100,
        "engineering",
        [100, 201],
    )
    assert not dispatcher.verify_routed_ownership(
        -1001,
        200,
        "engineering",
        [100, 101],
    )
    assert not dispatcher.verify_routed_ownership(
        -2002,
        100,
        "engineering",
        [100, 101],
    )


@pytest.mark.parametrize(
    "message_ids",
    [[], "100", [100, 0], list(range(1, 66))],
)
def test_routed_ownership_verification_rejects_malformed_constituents(message_ids):
    dispatcher = _dispatcher()
    assert dispatcher.record_root(-1001, 100, "engineering")

    assert not dispatcher.verify_routed_ownership(
        -1001,
        100,
        "engineering",
        message_ids,
    )


def test_ingress_reservation_requires_exact_opaque_identity_and_dispatcher():
    dispatcher = _dispatcher()
    other = _dispatcher()
    attempt = dispatcher.reserve_ingress_batch(-1001, [900, 901], "engineering")
    reservation = attempt.reservation
    assert reservation is not None

    assert repr(reservation) == "<IngressReservation>"
    assert other.release_ingress(reservation) is False
    assert other.commit_ingress(reservation) is False
    assert dispatcher.release_ingress(object()) is False

    duplicate = dispatcher.reserve_ingress_batch(-1001, [900], "design-team")
    assert duplicate.duplicate is True
    assert dispatcher.release_ingress(reservation) is True
    assert dispatcher.release_ingress(reservation) is False


def test_ingress_reservation_identity_cannot_change_profile_or_id_family():
    dispatcher = _dispatcher()
    attempt = dispatcher.reserve_ingress_batch(-1001, [900, 901], "engineering")
    reservation = attempt.reservation
    assert reservation is not None

    with pytest.raises(AttributeError):
        reservation._keys = (("-1001", "999"),)
    with pytest.raises(AttributeError):
        reservation._adapter_profile = "design-team"

    assert dispatcher.release_ingress(reservation) is True


@pytest.mark.parametrize(
    "message_ids",
    [
        [],
        [900, 0],
        [900, True],
        [900, "0901"],
        list(range(1, 66)),
        "900",
        {"900": True},
    ],
)
def test_invalid_or_oversized_ingress_batch_is_atomic_and_preserves_claim_order(
    message_ids,
):
    dispatcher = _dispatcher(max_claims=2)
    assert dispatcher.claim_ingress(-1001, 10, "default").claimed is True
    assert dispatcher.claim_ingress(-1001, 11, "default").claimed is True

    rejected = dispatcher.reserve_ingress_batch(-1001, message_ids, "engineering")

    assert rejected.reserved is False
    assert rejected.duplicate is False
    assert rejected.adapter_profile is None
    assert rejected.reservation is None
    assert dispatcher.claim_ingress(-1001, 12, "default").claimed is True
    assert dispatcher.claim_ingress(-1001, 11, "engineering").duplicate is True
    assert dispatcher.claim_ingress(-1001, 10, "engineering").claimed is True


def test_pending_reservations_are_bounded_without_evicting_committed_claims():
    dispatcher = _dispatcher(max_claims=2)
    assert dispatcher.claim_ingress(-1001, 10, "default").claimed is True
    assert dispatcher.claim_ingress(-1001, 11, "default").claimed is True

    first = dispatcher.reserve_ingress_batch(-1001, [20, 21], "engineering")
    assert first.reserved is True
    overflow = dispatcher.reserve_ingress_batch(-1001, [22], "design-team")
    assert overflow.reserved is False
    assert overflow.duplicate is False
    assert dispatcher.claim_ingress(-1001, 10, "engineering").duplicate is True
    assert dispatcher.claim_ingress(-1001, 11, "engineering").duplicate is True

    assert dispatcher.release_ingress(first.reservation) is True
    assert dispatcher.reserve_ingress_batch(-1001, [22], "design-team").reserved is True


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


def test_overlong_root_and_resolve_identifiers_fail_closed_without_eviction():
    module = _routing_module()
    dispatcher = _dispatcher(max_aliases=2)
    assert dispatcher.record_root(-1001, 100, "engineering") is True
    assert dispatcher.record_root(-1001, 200, "design-team") is True

    overlong_chat_id = "-" + "9" * 5000
    overlong_message_id = "9" * 10_000
    assert dispatcher.record_root(-1001, overlong_message_id, "default") is False
    assert dispatcher.record_root(overlong_chat_id, 300, "default") is False
    assert dispatcher.resolve_reply_owner(-1001, overlong_message_id) is None
    assert dispatcher.resolve_reply_owner(overlong_chat_id, 100) is None

    assert dispatcher.resolve_reply_owner(-1001, 100) == module.RootOwnership(
        root_message_id="100",
        owner_profile="engineering",
    )
    assert dispatcher.resolve_reply_owner(-1001, 200) == module.RootOwnership(
        root_message_id="200",
        owner_profile="design-team",
    )


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


def test_inbound_alias_batch_late_conflict_is_atomic_and_valid_retry_succeeds():
    module = _routing_module()
    dispatcher = _dispatcher(max_aliases=6)
    assert dispatcher.record_root(-1001, 100, "engineering") is True
    assert dispatcher.record_root(-1001, 200, "design-team") is True
    assert dispatcher.record_inbound_alias(-1001, 400, 200) is True

    assert dispatcher.record_inbound_alias_batch(-1001, [301, 400, 302], 100) is False

    assert dispatcher.resolve_reply_owner(-1001, 301) is None
    assert dispatcher.resolve_reply_owner(-1001, 302) is None
    assert dispatcher.resolve_reply_owner(-1001, 400) == module.RootOwnership(
        root_message_id="200",
        owner_profile="design-team",
    )
    assert dispatcher.record_inbound_alias_batch(-1001, [301, 302], 100) is True
    assert dispatcher.resolve_reply_owner(-1001, 301) == module.RootOwnership(
        root_message_id="100",
        owner_profile="engineering",
    )


def test_rejected_inbound_alias_batch_preserves_family_order_and_capacity():
    module = _routing_module()
    dispatcher = _dispatcher(max_aliases=2)
    assert dispatcher.record_root(-1001, 100, "engineering") is True
    assert dispatcher.record_root(-1001, 200, "design-team") is True

    assert dispatcher.record_inbound_alias_batch(-1001, [301, 302], 100) is False
    assert dispatcher.resolve_reply_owner(-1001, 301) is None
    assert dispatcher.record_root(-1001, 300, "default") is True

    assert dispatcher.resolve_reply_owner(-1001, 100) is None
    assert dispatcher.resolve_reply_owner(-1001, 200) == module.RootOwnership(
        root_message_id="200",
        owner_profile="design-team",
    )
    assert dispatcher.resolve_reply_owner(-1001, 300) == module.RootOwnership(
        root_message_id="300",
        owner_profile="default",
    )


def test_inbound_alias_batch_iterator_failure_is_atomic_and_retryable():
    module = _routing_module()
    dispatcher = _dispatcher(max_aliases=4)
    assert dispatcher.record_root(-1001, 100, "engineering") is True

    class RaisingIterator:
        def __init__(self):
            self.next_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.next_calls += 1
            if self.next_calls == 1:
                return 301
            raise RuntimeError("iteration failed")

    message_ids = RaisingIterator()
    assert dispatcher.record_inbound_alias_batch(-1001, message_ids, 100) is False
    assert message_ids.next_calls == 2
    assert dispatcher.resolve_reply_owner(-1001, 301) is None
    assert dispatcher.resolve_reply_owner(-1001, 100) == module.RootOwnership(
        root_message_id="100",
        owner_profile="engineering",
    )
    assert dispatcher.record_inbound_alias_batch(-1001, [301], 100) is True


@pytest.mark.parametrize("batch_api", ["inbound", "root"])
def test_atomic_alias_batch_overflow_is_bounded_and_preserves_family_order(batch_api):
    module = _routing_module()
    dispatcher = _dispatcher(max_aliases=2)
    assert dispatcher.record_root(-1001, 100, "engineering") is True
    assert dispatcher.record_root(-1001, 200, "design-team") is True

    class GuardedInfiniteDuplicates:
        def __init__(self):
            self.next_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.next_calls += 1
            if self.next_calls > 3:
                raise AssertionError("alias iterator was over-consumed")
            return 301

    message_ids = GuardedInfiniteDuplicates()
    if batch_api == "inbound":
        result = dispatcher.record_inbound_alias_batch(-1001, message_ids, 100)
    else:
        result = dispatcher.record_root_batch(
            -1001,
            400,
            "default",
            message_ids,
        )

    assert result is False
    assert message_ids.next_calls == 3
    assert dispatcher.resolve_reply_owner(-1001, 301) is None
    assert dispatcher.resolve_reply_owner(-1001, 400) is None
    assert dispatcher.record_root(-1001, 300, "default") is True
    assert dispatcher.resolve_reply_owner(-1001, 100) is None
    assert dispatcher.resolve_reply_owner(-1001, 200) == module.RootOwnership(
        root_message_id="200",
        owner_profile="design-team",
    )


@pytest.mark.parametrize("batch_api", ["inbound", "root"])
@pytest.mark.parametrize("message_ids", [[], [301, 0, 302], "301", {301: True}])
def test_atomic_alias_batch_invalid_input_does_not_mutate_or_reorder(
    batch_api,
    message_ids,
):
    module = _routing_module()
    dispatcher = _dispatcher(max_aliases=2)
    assert dispatcher.record_root(-1001, 100, "engineering") is True
    assert dispatcher.record_root(-1001, 200, "design-team") is True

    if batch_api == "inbound":
        result = dispatcher.record_inbound_alias_batch(-1001, message_ids, 100)
    else:
        result = dispatcher.record_root_batch(
            -1001,
            400,
            "default",
            message_ids,
        )

    assert result is False
    assert dispatcher.resolve_reply_owner(-1001, 301) is None
    assert dispatcher.resolve_reply_owner(-1001, 400) is None
    assert dispatcher.record_root(-1001, 300, "default") is True
    assert dispatcher.resolve_reply_owner(-1001, 100) is None
    assert dispatcher.resolve_reply_owner(-1001, 200) == module.RootOwnership(
        root_message_id="200",
        owner_profile="design-team",
    )


def test_root_batch_late_conflict_does_not_evict_or_partially_bind():
    module = _routing_module()
    dispatcher = _dispatcher(max_aliases=4)
    assert dispatcher.record_root(-1001, 100, "engineering") is True
    assert dispatcher.record_root(-1001, 200, "design-team") is True
    assert dispatcher.record_inbound_alias(-1001, 400, 200) is True

    assert (
        dispatcher.record_root_batch(
            -1001,
            300,
            "default",
            [300, 301, 400],
        )
        is False
    )

    assert dispatcher.resolve_reply_owner(-1001, 100) == module.RootOwnership(
        root_message_id="100",
        owner_profile="engineering",
    )
    assert dispatcher.resolve_reply_owner(-1001, 300) is None
    assert dispatcher.resolve_reply_owner(-1001, 301) is None
    assert dispatcher.resolve_reply_owner(-1001, 400) == module.RootOwnership(
        root_message_id="200",
        owner_profile="design-team",
    )

    assert dispatcher.record_root_batch(-1001, 300, "default", [300, 301]) is True
    assert dispatcher.resolve_reply_owner(-1001, 100) is None
    assert dispatcher.resolve_reply_owner(-1001, 300) == module.RootOwnership(
        root_message_id="300",
        owner_profile="default",
    )
    assert dispatcher.resolve_reply_owner(-1001, 301) == module.RootOwnership(
        root_message_id="300",
        owner_profile="default",
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


def test_oversized_outbound_iterable_consumes_at_most_max_aliases_plus_one():
    module = _routing_module()
    dispatcher = _dispatcher(max_aliases=3)
    assert dispatcher.record_root(-1001, 100, "engineering") is True

    class CountingIterator:
        def __init__(self):
            self.next_calls = 0
            self._values = iter(range(301, 310))

        def __iter__(self):
            return self

        def __next__(self):
            self.next_calls += 1
            return next(self._values)

    message_ids = CountingIterator()

    assert dispatcher.record_outbound_alias(-1001, message_ids, 100) is False
    assert message_ids.next_calls == 4
    assert dispatcher.resolve_reply_owner(-1001, 301) is None
    assert dispatcher.resolve_reply_owner(-1001, 100) == module.RootOwnership(
        root_message_id="100",
        owner_profile="engineering",
    )
    assert dispatcher.record_outbound_alias(-1001, [301], 100) is True


def test_infinite_duplicate_outbound_iterable_stops_at_total_attempt_bound():
    dispatcher = _dispatcher(max_aliases=3)
    assert dispatcher.record_root(-1001, 100, "engineering") is True

    class GuardedInfiniteDuplicates:
        def __init__(self):
            self.next_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.next_calls += 1
            if self.next_calls > 4:
                raise AssertionError("outbound alias iterator was over-consumed")
            return 301

    message_ids = GuardedInfiniteDuplicates()

    assert dispatcher.record_outbound_alias(-1001, message_ids, 100) is False
    assert message_ids.next_calls == 4
    assert dispatcher.resolve_reply_owner(-1001, 301) is None


def test_outbound_iterator_construction_exception_fails_closed():
    module = _routing_module()
    dispatcher = _dispatcher(max_aliases=3)
    assert dispatcher.record_root(-1001, 100, "engineering") is True

    class RaisingIterable:
        def __iter__(self):
            raise RuntimeError("iterator unavailable")

    assert dispatcher.record_outbound_alias(-1001, RaisingIterable(), 100) is False
    assert dispatcher.resolve_reply_owner(-1001, 100) == module.RootOwnership(
        root_message_id="100",
        owner_profile="engineering",
    )
    assert dispatcher.record_outbound_alias(-1001, [301], 100) is True


def test_outbound_iteration_exception_is_atomic_and_later_batch_works():
    module = _routing_module()
    dispatcher = _dispatcher(max_aliases=3)
    assert dispatcher.record_root(-1001, 100, "engineering") is True

    class RaisingIterator:
        def __init__(self):
            self.next_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.next_calls += 1
            if self.next_calls == 1:
                return 301
            raise RuntimeError("iteration failed")

    message_ids = RaisingIterator()

    assert dispatcher.record_outbound_alias(-1001, message_ids, 100) is False
    assert message_ids.next_calls == 2
    assert dispatcher.resolve_reply_owner(-1001, 301) is None
    assert dispatcher.resolve_reply_owner(-1001, 100) == module.RootOwnership(
        root_message_id="100",
        owner_profile="engineering",
    )
    assert dispatcher.record_outbound_alias(-1001, [301], 100) is True


def test_unknown_outbound_root_does_not_construct_user_iterator():
    dispatcher = _dispatcher(max_aliases=3)

    class IteratorProbe:
        def __init__(self):
            self.iter_calls = 0

        def __iter__(self):
            self.iter_calls += 1
            return iter([301])

    message_ids = IteratorProbe()

    assert dispatcher.record_outbound_alias(-1001, message_ids, 999) is False
    assert message_ids.iter_calls == 0


def test_outbound_batch_revalidates_root_family_before_commit():
    module = _routing_module()
    dispatcher = _dispatcher(max_aliases=2)
    assert dispatcher.record_root(-1001, 100, "engineering") is True

    class ReplacingIterator:
        def __init__(self):
            self._replaced = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._replaced:
                raise StopIteration
            self._replaced = True
            assert dispatcher.record_root(-1001, 200, "design-team") is True
            assert dispatcher.record_root(-1001, 300, "default") is True
            assert dispatcher.record_root(-1001, 100, "engineering") is True
            return 301

    assert dispatcher.record_outbound_alias(-1001, ReplacingIterator(), 100) is False
    assert dispatcher.resolve_reply_owner(-1001, 301) is None
    assert dispatcher.resolve_reply_owner(-1001, 100) == module.RootOwnership(
        root_message_id="100",
        owner_profile="engineering",
    )


def test_rejected_outbound_batch_does_not_reorder_root_families():
    module = _routing_module()
    dispatcher = _dispatcher(max_aliases=2)
    assert dispatcher.record_root(-1001, 100, "engineering") is True
    assert dispatcher.record_root(-1001, 200, "design-team") is True

    assert dispatcher.record_outbound_alias(-1001, [301, 302, 303], 100) is False
    assert dispatcher.record_root(-1001, 300, "default") is True

    assert dispatcher.resolve_reply_owner(-1001, 100) is None
    assert dispatcher.resolve_reply_owner(-1001, 200) == module.RootOwnership(
        root_message_id="200",
        owner_profile="design-team",
    )
    assert dispatcher.resolve_reply_owner(-1001, 300) == module.RootOwnership(
        root_message_id="300",
        owner_profile="default",
    )
