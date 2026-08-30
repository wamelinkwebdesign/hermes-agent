"""Explicit thread handover between team members (WAM-46).

Replying to an answer and naming somebody else is how a human hands a
conversation over. Reply-chain ownership alone ignored that: the thread stayed
with the original owner and the named bot declined, so a live conversation
could never be passed on without starting again and re-explaining.

The rule is deliberately narrow. Exactly one named member, and not the current
owner, moves the thread. Naming nobody is an ordinary follow-up, naming the
current owner is emphasis, and naming several is ambiguous between "hand this
over" and "all of you look" — which cannot be told apart from mentions alone.
Ambiguity leaves the thread where it is, because a wrongly moved conversation
is far harder for a human to notice than one that did not move.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gateway import telegram_team_delivery as td
from gateway.platforms.base import BasePlatformAdapter
from gateway.session import SessionStore
from gateway.telegram_team_routing import (
    RootOwnership,
    TelegramTeamDispatcher,
    resolve_mention_handover,
)

from tests.gateway.test_telegram_team_dispatch import (
    _ALLOWED_CHAT,
    _event,
    _runner,
)


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(td, "_db_path", lambda: home / "state.db")
    yield


async def _own_root(roster, profile, message_id, text):
    """Give *profile* a root, then let its answer join the family."""
    adapter = roster[profile]
    event = _event(adapter, message_id, text=text)
    assert await adapter._team_ingress_handler(adapter, event) is False
    return event


class TestHandoverRule:
    """The decision itself, in isolation from ingress."""

    def _config(self):
        runner, _roster = _runner()
        return runner._telegram_team_config

    def test_one_other_member_named_is_a_handover(self):
        assert resolve_mention_handover(
            mentions={"Virgil_Bot"},
            config=self._config(),
            current_owner_profile="engineering",
        ) == "design"

    def test_naming_the_current_owner_is_not_a_handover(self):
        """Emphasis, not a move."""
        assert resolve_mention_handover(
            mentions={"Woz_Bot"},
            config=self._config(),
            current_owner_profile="engineering",
        ) is None

    def test_naming_nobody_is_not_a_handover(self):
        assert resolve_mention_handover(
            mentions=set(),
            config=self._config(),
            current_owner_profile="engineering",
        ) is None

    def test_naming_several_is_too_ambiguous_to_move_the_thread(self):
        assert resolve_mention_handover(
            mentions={"Virgil_Bot", "Ace_Bot"},
            config=self._config(),
            current_owner_profile="engineering",
        ) is None

    def test_current_owner_plus_another_is_still_ambiguous(self):
        """"@Woz thanks, @Virgil thoughts?" and "@Woz @Virgil look at this"
        are indistinguishable from mentions alone."""
        assert resolve_mention_handover(
            mentions={"Woz_Bot", "Virgil_Bot"},
            config=self._config(),
            current_owner_profile="engineering",
        ) is None

    def test_strangers_are_not_members(self):
        assert resolve_mention_handover(
            mentions={"Some_Stranger_Bot"},
            config=self._config(),
            current_owner_profile="engineering",
        ) is None

    def test_a_stranger_alongside_a_member_still_hands_over(self):
        """Only team members count toward the ambiguity test."""
        assert resolve_mention_handover(
            mentions={"Some_Stranger_Bot", "Virgil_Bot"},
            config=self._config(),
            current_owner_profile="engineering",
        ) == "design"

    def test_username_case_and_at_prefix_are_normalized(self):
        assert resolve_mention_handover(
            mentions={"@vIrGiL_bOt"},
            config=self._config(),
            current_owner_profile="engineering",
        ) == "design"


class TestTransferMovesTheWholeFamily:
    def test_root_and_every_alias_follow_the_new_owner(self):
        """A partial move would leave two bots each owning part of one
        conversation."""
        d = TelegramTeamDispatcher()
        assert d.record_root_batch(_ALLOWED_CHAT, "500", "engineering",
                                   ("500", "501", "502"))
        assert d.record_outbound_alias(_ALLOWED_CHAT, ["503"], "500")

        assert d.transfer_root_ownership(_ALLOWED_CHAT, "500", "design") is True

        for message_id in ("500", "501", "502", "503"):
            assert d.resolve_reply_owner(_ALLOWED_CHAT, message_id) == RootOwnership(
                "500", "design"
            ), message_id

    def test_transfer_is_idempotent(self):
        """A replayed handover is a no-op, not a failure the caller must
        distinguish from a real one."""
        d = TelegramTeamDispatcher()
        d.record_root(_ALLOWED_CHAT, "510", "engineering")
        assert d.transfer_root_ownership(_ALLOWED_CHAT, "510", "design") is True
        assert d.transfer_root_ownership(_ALLOWED_CHAT, "510", "design") is True
        assert d.resolve_reply_owner(_ALLOWED_CHAT, "510").owner_profile == "design"

    def test_transfer_can_move_the_thread_onward_again(self):
        d = TelegramTeamDispatcher()
        d.record_root(_ALLOWED_CHAT, "511", "engineering")
        assert d.transfer_root_ownership(_ALLOWED_CHAT, "511", "design")
        assert d.transfer_root_ownership(_ALLOWED_CHAT, "511", "default")
        assert d.resolve_reply_owner(_ALLOWED_CHAT, "511").owner_profile == "default"

    def test_unknown_root_is_refused(self):
        d = TelegramTeamDispatcher()
        assert d.transfer_root_ownership(_ALLOWED_CHAT, "999", "design") is False

    def test_other_families_are_untouched(self):
        d = TelegramTeamDispatcher()
        d.record_root(_ALLOWED_CHAT, "520", "engineering")
        d.record_root(_ALLOWED_CHAT, "530", "engineering")
        d.transfer_root_ownership(_ALLOWED_CHAT, "520", "design")
        assert d.resolve_reply_owner(_ALLOWED_CHAT, "530") == RootOwnership(
            "530", "engineering"
        )

    def test_a_transfer_never_crosses_chats(self):
        d = TelegramTeamDispatcher()
        d.record_root(_ALLOWED_CHAT, "540", "engineering")
        assert d.transfer_root_ownership("-1009999999999", "540", "design") is False
        assert d.resolve_reply_owner(_ALLOWED_CHAT, "540").owner_profile == (
            "engineering"
        )

    @pytest.mark.parametrize(
        "chat_id,root_id,profile",
        [
            (None, "550", "design"),
            (_ALLOWED_CHAT, None, "design"),
            (_ALLOWED_CHAT, "550", None),
            (_ALLOWED_CHAT, "550", ""),
            (_ALLOWED_CHAT, "not-a-message", "design"),
            ("not-a-chat", "550", "design"),
            (_ALLOWED_CHAT, "550", "bad profile!"),
        ],
    )
    def test_malformed_input_fails_closed(self, chat_id, root_id, profile):
        d = TelegramTeamDispatcher()
        d.record_root(_ALLOWED_CHAT, "550", "engineering")
        assert d.transfer_root_ownership(chat_id, root_id, profile) is False
        assert d.resolve_reply_owner(_ALLOWED_CHAT, "550").owner_profile == (
            "engineering"
        )


class TestHandoverThroughIngress:
    """The behaviour a human actually experiences in the group."""

    @pytest.mark.asyncio
    async def test_replying_to_an_answer_and_naming_another_bot_hands_over(
        self, monkeypatch
    ):
        runner, roster = _runner()
        dispatcher = runner._telegram_team_dispatcher
        runner.session_store = SessionStore.__new__(SessionStore)
        routed = []

        async def _capture(self, event):
            routed.append((getattr(self, "_owner_profile", None), event))

        monkeypatch.setattr(BasePlatformAdapter, "handle_message", _capture)

        await _own_root(roster, "engineering", 600, "@Woz_Bot is this migration safe?")
        assert dispatcher.record_outbound_alias(_ALLOWED_CHAT, ["601"], "600")

        woz = roster["engineering"]
        handover = _event(
            woz,
            602,
            text="@Virgil_Bot what do you think of this?",
            reply_to_message_id=601,
            reply_author_username="Woz_Bot",
        )
        # Woz consumes it: the event is redispatched to Virgil, not handled here.
        assert await woz._team_ingress_handler(woz, handover) is True

        assert len(routed) == 1
        owner, routed_event = routed[0]
        assert owner == "design"
        assert routed_event.metadata["telegram_team_owner_profile"] == "design"
        assert routed_event.metadata["telegram_team_route_reason"] == "mention_handover"

    @pytest.mark.asyncio
    async def test_the_thread_goes_with_it(self, monkeypatch):
        """The new owner inherits the root scope, so the conversation
        continues rather than forking into a fresh session."""
        runner, roster = _runner()
        runner.session_store = SessionStore.__new__(SessionStore)
        routed = []
        monkeypatch.setattr(
            BasePlatformAdapter,
            "handle_message",
            lambda self, event: routed.append(event) or _noop(),
        )

        async def _noop():
            return None

        await _own_root(roster, "engineering", 610, "@Woz_Bot look at this")
        dispatcher = runner._telegram_team_dispatcher
        dispatcher.record_outbound_alias(_ALLOWED_CHAT, ["611"], "610")

        woz = roster["engineering"]
        await woz._team_ingress_handler(
            woz,
            _event(
                woz,
                612,
                text="@Virgil_Bot thoughts?",
                reply_to_message_id=611,
                reply_author_username="Woz_Bot",
            ),
        )

        assert routed[0].metadata["telegram_team_root_message_id"] == "610"
        assert routed[0].source.session_scope_id == (
            f"telegram-team:{_ALLOWED_CHAT}:610"
        )

    @pytest.mark.asyncio
    async def test_ownership_actually_moves_for_later_replies(self, monkeypatch):
        runner, roster = _runner()
        dispatcher = runner._telegram_team_dispatcher
        runner.session_store = SessionStore.__new__(SessionStore)
        monkeypatch.setattr(BasePlatformAdapter, "handle_message", AsyncMock())

        await _own_root(roster, "engineering", 620, "@Woz_Bot start here")
        dispatcher.record_outbound_alias(_ALLOWED_CHAT, ["621"], "620")
        woz = roster["engineering"]
        await woz._team_ingress_handler(
            woz,
            _event(woz, 622, text="@Virgil_Bot over to you",
                   reply_to_message_id=621, reply_author_username="Woz_Bot"),
        )

        assert dispatcher.resolve_reply_owner(_ALLOWED_CHAT, "620") == RootOwnership(
            "620", "design"
        )

        # A plain follow-up now belongs to Virgil...
        virgil = roster["design"]
        follow_up = _event(virgil, 623, text="and the spacing?",
                           reply_to_message_id=622)
        assert await virgil._team_ingress_handler(virgil, follow_up) is False
        assert follow_up.metadata["telegram_team_route_reason"] == (
            "reply_to_root_chain"
        )

        # ...and Woz no longer accepts it.
        intruder = _event(woz, 624, text="mine again", reply_to_message_id=622)
        assert await woz._team_ingress_handler(woz, intruder) is True

    @pytest.mark.asyncio
    async def test_naming_the_current_owner_is_an_ordinary_follow_up(self):
        runner, roster = _runner()
        dispatcher = runner._telegram_team_dispatcher
        runner.session_store = SessionStore.__new__(SessionStore)

        await _own_root(roster, "engineering", 630, "@Woz_Bot start")
        dispatcher.record_outbound_alias(_ALLOWED_CHAT, ["631"], "630")
        woz = roster["engineering"]
        event = _event(woz, 632, text="@Woz_Bot and also this",
                       reply_to_message_id=631, reply_author_username="Woz_Bot")

        assert await woz._team_ingress_handler(woz, event) is False
        assert event.metadata["telegram_team_route_reason"] == "reply_to_root_chain"
        assert event.metadata["telegram_team_owner_profile"] == "engineering"

    @pytest.mark.asyncio
    async def test_naming_two_members_leaves_the_thread_where_it_is(self):
        runner, roster = _runner()
        dispatcher = runner._telegram_team_dispatcher
        runner.session_store = SessionStore.__new__(SessionStore)

        await _own_root(roster, "engineering", 640, "@Woz_Bot start")
        dispatcher.record_outbound_alias(_ALLOWED_CHAT, ["641"], "640")
        woz = roster["engineering"]
        event = _event(woz, 642, text="@Virgil_Bot @Ace_Bot who owns this?",
                       reply_to_message_id=641, reply_author_username="Woz_Bot")

        assert await woz._team_ingress_handler(woz, event) is False
        assert event.metadata["telegram_team_route_reason"] == "reply_to_root_chain"
        assert dispatcher.resolve_reply_owner(_ALLOWED_CHAT, "640").owner_profile == (
            "engineering"
        )

    @pytest.mark.asyncio
    async def test_a_handover_target_without_an_adapter_fails_closed(
        self, monkeypatch
    ):
        runner, roster = _runner()
        dispatcher = runner._telegram_team_dispatcher
        runner.session_store = SessionStore.__new__(SessionStore)

        await _own_root(roster, "engineering", 650, "@Woz_Bot start")
        dispatcher.record_outbound_alias(_ALLOWED_CHAT, ["651"], "650")
        monkeypatch.setattr(
            type(runner), "_telegram_team_adapter_for_profile",
            lambda self, profile: None,
        )

        woz = roster["engineering"]
        event = _event(woz, 652, text="@Virgil_Bot take it",
                       reply_to_message_id=651, reply_author_username="Woz_Bot")

        assert await woz._team_ingress_handler(woz, event) is True
        # The thread must not be left half-moved.
        assert dispatcher.resolve_reply_owner(_ALLOWED_CHAT, "650").owner_profile == (
            "engineering"
        )

    @pytest.mark.asyncio
    async def test_a_failed_transfer_publishes_no_routed_event(self, monkeypatch):
        runner, roster = _runner()
        dispatcher = runner._telegram_team_dispatcher
        runner.session_store = SessionStore.__new__(SessionStore)
        routed = []
        monkeypatch.setattr(
            BasePlatformAdapter, "handle_message",
            lambda self, event: routed.append(event) or AsyncMock()(),
        )

        await _own_root(roster, "engineering", 660, "@Woz_Bot start")
        dispatcher.record_outbound_alias(_ALLOWED_CHAT, ["661"], "660")
        monkeypatch.setattr(dispatcher, "transfer_root_ownership", lambda *a: False)

        woz = roster["engineering"]
        event = _event(woz, 662, text="@Virgil_Bot take it",
                       reply_to_message_id=661, reply_author_username="Woz_Bot")

        assert await woz._team_ingress_handler(woz, event) is True
        assert routed == []

    @pytest.mark.asyncio
    async def test_the_target_seeing_the_update_first_still_yields_one_handler(
        self, monkeypatch
    ):
        """Both bots receive the same Telegram update. Whichever reaches
        ingress first, the message must be handled exactly once."""
        runner, roster = _runner()
        dispatcher = runner._telegram_team_dispatcher
        runner.session_store = SessionStore.__new__(SessionStore)
        handled = []

        async def _capture(self, event):
            handled.append(getattr(self, "_owner_profile", None))

        monkeypatch.setattr(BasePlatformAdapter, "handle_message", _capture)

        await _own_root(roster, "engineering", 680, "@Woz_Bot start")
        dispatcher.record_outbound_alias(_ALLOWED_CHAT, ["681"], "680")

        woz, virgil = roster["engineering"], roster["design"]
        kwargs = dict(
            text="@Virgil_Bot over to you",
            reply_to_message_id=681,
            reply_author_username="Woz_Bot",
        )

        # Virgil sees it first and must decline: the thread is still Woz's.
        assert await virgil._team_ingress_handler(
            virgil, _event(virgil, 682, **kwargs)
        ) is True
        assert handled == []

        # Woz then performs the handover and redispatches once.
        assert await woz._team_ingress_handler(
            woz, _event(woz, 682, **kwargs)
        ) is True
        assert handled == ["design"]

        # Virgil re-seeing the original update after the move must not
        # produce a second turn.
        assert await virgil._team_ingress_handler(
            virgil, _event(virgil, 682, **kwargs)
        ) is True
        assert handled == ["design"]

    @pytest.mark.asyncio
    async def test_a_handover_owes_exactly_one_obligation_to_the_new_owner(
        self, monkeypatch
    ):
        """WAM-31 must still resolve to one outcome across the transfer."""
        runner, roster = _runner()
        dispatcher = runner._telegram_team_dispatcher
        runner.session_store = SessionStore.__new__(SessionStore)
        monkeypatch.setattr(BasePlatformAdapter, "handle_message", AsyncMock())

        await _own_root(roster, "engineering", 670, "@Woz_Bot start")
        dispatcher.record_outbound_alias(_ALLOWED_CHAT, ["671"], "670")
        woz = roster["engineering"]
        await woz._team_ingress_handler(
            woz,
            _event(woz, 672, text="@Virgil_Bot over to you",
                   reply_to_message_id=671, reply_author_username="Woz_Bot"),
        )

        with td._connect() as conn:
            rows = conn.execute(
                "SELECT root_message_id, head_message_id, owner_profile, "
                "route_reason, state FROM telegram_team_obligations "
                "WHERE head_message_id='672'"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0] == ("670", "672", "design", "mention_handover", "pending")
