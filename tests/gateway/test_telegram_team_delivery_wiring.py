"""End-to-end wiring for the Telegram team terminal-outcome ledger (WAM-31).

``test_telegram_team_delivery.py`` covers the ledger's own state machine.
This module covers the three seams that make it mean something in the
running gateway:

- central ingress records the debt the moment it accepts a message
- the end of a turn discharges it, speaking a fallback when the turn put
  nothing in front of the group
- a boot after a crash speaks for the debts the dead process left behind

The load-bearing property throughout is that the ledger is a safety net,
never a gate: every failure mode here must leave routing behaviour exactly
as it was.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway import telegram_team_delivery as td
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, SessionStore, build_session_key
from gateway.telegram_team_classifier import (
    ClassificationDecision,
    parse_classification_output,
)

from tests.gateway.test_telegram_team_dispatch import (
    _ALLOWED_CHAT,
    _adapter,
    _event,
    _runner,
)


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(td, "_db_path", lambda: home / "state.db")
    yield


def _rows():
    with td._connect() as conn:
        return conn.execute(
            """SELECT obligation_id, chat_id, root_message_id, head_message_id,
                      owner_profile, route_reason, state
               FROM telegram_team_obligations ORDER BY created_at ASC"""
        ).fetchall()


def _accept_unaddressed(monkeypatch, runner, roster, message_id, *, decision=None):
    """Drive one unaddressed message through central ingress."""
    import gateway.telegram_team_classifier as classifier_module
    import gateway.telegram_team_context as context_module

    runner.session_store = SessionStore.__new__(SessionStore)
    monkeypatch.setattr(
        classifier_module,
        "classify_new_root",
        AsyncMock(return_value=decision or ClassificationDecision("self")),
    )
    monkeypatch.setattr(context_module, "collect_team_context", Mock(return_value=""))
    return _event(roster["default"], message_id, text="please investigate this")


class TestIngressRecordsTheDebt:
    @pytest.mark.asyncio
    async def test_accepted_message_records_a_pending_obligation(self, monkeypatch):
        runner, roster = _runner()
        event = _accept_unaddressed(monkeypatch, runner, roster, 300)

        assert await roster["default"]._team_ingress_handler(
            roster["default"], event
        ) is False

        rows = _rows()
        assert len(rows) == 1
        (_oid, chat_id, root, head, owner, reason, state) = rows[0]
        assert chat_id == _ALLOWED_CHAT
        assert (root, head) == ("300", "300")
        assert owner == "default"
        assert reason == "semantic_self"
        assert state == "pending"

    @pytest.mark.asyncio
    async def test_obligation_id_matches_what_settling_recomputes(self, monkeypatch):
        """Ingress and the turn-end hook must agree on the row, or every
        answered message would look undischarged."""
        runner, roster = _runner()
        event = _accept_unaddressed(monkeypatch, runner, roster, 301)
        await roster["default"]._team_ingress_handler(roster["default"], event)

        assert _rows()[0][0] == td.compute_obligation_id(_ALLOWED_CHAT, "301", "301")

    @pytest.mark.asyncio
    async def test_specialist_redispatch_records_the_specialist_as_owner(
        self, monkeypatch
    ):
        runner, roster = _runner()
        monkeypatch.setattr(BasePlatformAdapter, "handle_message", AsyncMock())
        event = _accept_unaddressed(
            monkeypatch,
            runner,
            roster,
            302,
            decision=parse_classification_output(
                '{"decision":"owner","profile":"engineering"}',
                config=runner._telegram_team_config,
            ),
        )

        assert await roster["default"]._team_ingress_handler(
            roster["default"], event
        ) is True

        rows = _rows()
        assert len(rows) == 1
        assert rows[0][4] == "engineering"
        assert rows[0][5] == "semantic_specialist"

    @pytest.mark.asyncio
    async def test_replayed_update_does_not_open_a_second_debt(self, monkeypatch):
        runner, roster = _runner()
        event = _accept_unaddressed(monkeypatch, runner, roster, 303)
        await roster["default"]._team_ingress_handler(roster["default"], event)

        replay = _accept_unaddressed(monkeypatch, runner, roster, 303)
        await roster["default"]._team_ingress_handler(roster["default"], replay)

        assert len(_rows()) == 1

    @pytest.mark.asyncio
    async def test_rejected_event_records_nothing(self, monkeypatch):
        """A message routing never accepted owes nothing."""
        runner, roster = _runner()
        runner.session_store = SessionStore.__new__(SessionStore)
        # A specialist cannot own unaddressed ingress; this fails closed.
        event = _event(roster["engineering"], 304, text="unaddressed")

        assert await roster["engineering"]._team_ingress_handler(
            roster["engineering"], event
        ) is True
        assert _rows() == []

    @pytest.mark.asyncio
    async def test_disabled_ledger_still_routes_the_message(self, monkeypatch):
        runner, roster = _runner()
        monkeypatch.setattr(td, "ledger_enabled", lambda config=None: False)
        event = _accept_unaddressed(monkeypatch, runner, roster, 305)

        assert await roster["default"]._team_ingress_handler(
            roster["default"], event
        ) is False
        assert _rows() == []

    @pytest.mark.asyncio
    async def test_broken_ledger_never_consumes_an_accepted_message(self, monkeypatch):
        """The ledger is a safety net, not a gate: a message routing already
        accepted must not be dropped because sqlite is unhappy."""
        runner, roster = _runner()

        def _boom(**kwargs):
            raise RuntimeError("disk I/O error")

        monkeypatch.setattr(td, "record_obligation", _boom)
        event = _accept_unaddressed(monkeypatch, runner, roster, 306)

        assert await roster["default"]._team_ingress_handler(
            roster["default"], event
        ) is False
        assert event.metadata["telegram_team_route_reason"] == "semantic_self"


class TestTurnSettlesTheDebt:
    def _scoped_event(self, adapter, message_id=400, root="400"):
        event = _event(adapter, message_id)
        event.source.session_scope_id = f"telegram-team:{_ALLOWED_CHAT}:{root}"
        return event

    def _record(self, root="400", head="400", owner="default"):
        oid = td.compute_obligation_id(_ALLOWED_CHAT, root, head)
        td.record_obligation(
            obligation_id=oid,
            chat_id=_ALLOWED_CHAT,
            root_message_id=root,
            head_message_id=head,
            constituent_ids=(head,),
            owner_profile=owner,
            route_reason="semantic_self",
        )
        return oid

    @pytest.mark.asyncio
    async def test_delivered_turn_closes_as_answered(self):
        adapter = _adapter("default")
        oid = self._record()
        send = AsyncMock()
        adapter.send = send

        await adapter._settle_turn_delivery_obligation(
            self._scoped_event(adapter), delivered=True
        )

        assert _rows()[0][6] == "answered"
        send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_silent_turn_speaks_a_fallback_and_closes(self):
        adapter = _adapter("default")
        self._record()
        send = AsyncMock(return_value=SendResult(success=True))
        adapter.send = send

        await adapter._settle_turn_delivery_obligation(
            self._scoped_event(adapter), delivered=False
        )

        send.assert_awaited_once()
        assert send.await_args.kwargs["content"] == td.FALLBACK_NOTICE
        assert send.await_args.kwargs["chat_id"] == _ALLOWED_CHAT
        assert send.await_args.kwargs["reply_to"] == "400"
        assert _rows()[0][6] == "fallback"

    @pytest.mark.asyncio
    async def test_failed_fallback_send_is_not_a_silent_discharge(self):
        """A failed send must leave the debt open for a later boot."""
        adapter = _adapter("default")
        self._record()
        adapter.send = AsyncMock(
            return_value=SendResult(success=False, error="Not connected")
        )

        await adapter._settle_turn_delivery_obligation(
            self._scoped_event(adapter), delivered=False
        )

        assert _rows()[0][6] == "pending"

    @pytest.mark.asyncio
    async def test_raising_send_is_not_a_silent_discharge(self):
        adapter = _adapter("default")
        self._record()
        adapter.send = AsyncMock(side_effect=RuntimeError("transport exploded"))

        await adapter._settle_turn_delivery_obligation(
            self._scoped_event(adapter), delivered=False
        )

        assert _rows()[0][6] == "pending"

    @pytest.mark.asyncio
    async def test_already_answered_debt_never_earns_a_second_message(self):
        """The exactly-once guarantee, from the duplicate-adapter direction."""
        adapter = _adapter("default")
        oid = self._record()
        td.close_obligation(oid, outcome="answered")
        send = AsyncMock(return_value=SendResult(success=True))
        adapter.send = send

        await adapter._settle_turn_delivery_obligation(
            self._scoped_event(adapter), delivered=False
        )

        send.assert_not_awaited()
        assert _rows()[0][6] == "answered"

    @pytest.mark.asyncio
    async def test_two_settles_produce_exactly_one_fallback(self):
        adapter = _adapter("default")
        self._record()
        send = AsyncMock(return_value=SendResult(success=True))
        adapter.send = send
        event = self._scoped_event(adapter)

        await adapter._settle_turn_delivery_obligation(event, delivered=False)
        await adapter._settle_turn_delivery_obligation(event, delivered=False)

        assert send.await_count == 1

    @pytest.mark.asyncio
    async def test_non_team_event_owes_nothing(self):
        adapter = _adapter("default")
        send = AsyncMock()
        adapter.send = send
        event = _event(adapter, 500)
        event.source.session_scope_id = None

        await adapter._settle_turn_delivery_obligation(event, delivered=False)

        send.assert_not_awaited()
        assert _rows() == []

    @pytest.mark.asyncio
    async def test_foreign_scope_prefix_owes_nothing(self):
        adapter = _adapter("default")
        send = AsyncMock()
        adapter.send = send
        event = _event(adapter, 501)
        event.source.session_scope_id = "discord-team:-100:501"

        await adapter._settle_turn_delivery_obligation(event, delivered=False)

        send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_scope_transport_chat_mismatch_fails_closed(self):
        """A scope that disagrees with the transport must never be spoken
        into, nor used to discharge somebody else's debt."""
        adapter = _adapter("default")
        send = AsyncMock()
        adapter.send = send
        event = _event(adapter, 502)
        event.source.session_scope_id = "telegram-team:-1009999999999:502"

        await adapter._settle_turn_delivery_obligation(event, delivered=False)

        send.assert_not_awaited()

    @pytest.mark.parametrize(
        "scope",
        [
            "telegram-team:",
            "telegram-team:-1001234567890",
            "telegram-team:-1001234567890:400:extra",
            "telegram-team:not-a-chat:400",
            "telegram-team:-1001234567890:not-a-message",
        ],
    )
    @pytest.mark.asyncio
    async def test_malformed_scope_fails_closed(self, scope):
        adapter = _adapter("default")
        send = AsyncMock()
        adapter.send = send
        event = _event(adapter, 503)
        event.source.session_scope_id = scope

        await adapter._settle_turn_delivery_obligation(event, delivered=False)

        send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_disabled_ledger_never_speaks(self, monkeypatch):
        adapter = _adapter("default")
        self._record()
        send = AsyncMock()
        adapter.send = send
        monkeypatch.setattr(td, "ledger_enabled", lambda config=None: False)

        await adapter._settle_turn_delivery_obligation(
            self._scoped_event(adapter), delivered=False
        )

        send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unrecorded_message_never_speaks(self):
        """No debt, no fallback — the ledger only speaks for what ingress
        actually accepted."""
        adapter = _adapter("default")
        send = AsyncMock()
        adapter.send = send

        await adapter._settle_turn_delivery_obligation(
            self._scoped_event(adapter), delivered=False
        )

        send.assert_not_awaited()


class _ProbeAdapter(BasePlatformAdapter):
    """Minimal concrete adapter for exercising the base turn lifecycle."""

    def __init__(self) -> None:
        super().__init__(PlatformConfig(enabled=True, token="x"), Platform.SLACK)
        self.sent: list[str] = []
        self.settled: list[bool] = []
        self.settle_error: BaseException | None = None

    async def start(self):  # pragma: no cover - unused
        pass

    async def stop(self):  # pragma: no cover - unused
        pass

    async def connect(self):  # pragma: no cover - unused
        pass

    async def disconnect(self):  # pragma: no cover - unused
        pass

    async def get_chat_info(self, chat_id):  # pragma: no cover - unused
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id="m1")

    async def send_typing(self, chat_id, metadata=None):  # pragma: no cover - unused
        pass

    async def _settle_turn_delivery_obligation(self, event, *, delivered):
        if self.settle_error is not None:
            raise self.settle_error
        self.settled.append(delivered)


def _probe_event() -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.SLACK,
            user_id="U1",
            chat_id="C1",
            user_name="tester",
            chat_type="channel",
        ),
    )


class TestTurnLifecycleCallsTheHook:
    """The base-adapter contract the Telegram override depends on."""

    @pytest.mark.asyncio
    async def test_silent_turn_reports_nothing_delivered(self):
        """A turn that neither answered nor errored is the silent drop."""
        adapter = _ProbeAdapter()
        adapter.set_message_handler(AsyncMock(return_value=None))
        event = _probe_event()

        await adapter._process_message_background(
            event, build_session_key(event.source)
        )

        assert adapter.settled == [False]
        assert adapter.sent == []

    @pytest.mark.asyncio
    async def test_answered_turn_reports_delivered(self):
        adapter = _ProbeAdapter()
        adapter.set_message_handler(AsyncMock(return_value="the answer"))
        event = _probe_event()

        await adapter._process_message_background(
            event, build_session_key(event.source)
        )

        assert adapter.settled == [True]

    @pytest.mark.asyncio
    async def test_failed_turn_reports_delivered_because_of_the_error_notice(self):
        """The error notice the base adapter sends is itself a public
        outcome, so the hook must not treat that turn as silent — a fallback
        on top of it would be a duplicate public reply."""
        adapter = _ProbeAdapter()
        adapter.set_message_handler(AsyncMock(side_effect=RuntimeError("boom")))
        event = _probe_event()

        await adapter._process_message_background(
            event, build_session_key(event.source)
        )

        assert adapter.sent, "the base error notice should have been sent"
        assert adapter.settled == [True]

    @pytest.mark.asyncio
    async def test_hook_failure_never_breaks_the_turn(self):
        adapter = _ProbeAdapter()
        adapter.settle_error = RuntimeError("ledger exploded")
        adapter.set_message_handler(AsyncMock(return_value="the answer"))
        event = _probe_event()

        await adapter._process_message_background(
            event, build_session_key(event.source)
        )

        assert adapter.sent == ["the answer"]

    @pytest.mark.asyncio
    async def test_hook_runs_before_a_reraised_base_exception(self):
        """Shutdown must not strand an undischarged obligation."""
        adapter = _ProbeAdapter()
        adapter.set_message_handler(AsyncMock(side_effect=SystemExit(1)))
        event = _probe_event()

        with pytest.raises(SystemExit):
            await adapter._process_message_background(
                event, build_session_key(event.source)
            )

        assert adapter.settled == [True]


class TestBootSweepSpeaksForDeadProcesses:
    def _record_dead(self, runner, owner="default", root="700"):
        oid = td.compute_obligation_id(_ALLOWED_CHAT, root, root)
        td.record_obligation(
            obligation_id=oid,
            chat_id=_ALLOWED_CHAT,
            root_message_id=root,
            head_message_id=root,
            constituent_ids=(root,),
            owner_profile=owner,
            route_reason="semantic_self",
        )
        with td._connect() as conn:
            with conn:
                conn.execute(
                    "UPDATE telegram_team_obligations SET owner_pid=999999999, "
                    "owner_started_at=1 WHERE obligation_id=?",
                    (oid,),
                )
        return oid

    @pytest.mark.asyncio
    async def test_dead_owner_debt_is_spoken_by_its_own_adapter(self):
        runner, roster = _runner()
        self._record_dead(runner, owner="engineering")
        send = AsyncMock(return_value=SendResult(success=True))
        roster["engineering"].send = send

        claimed = await runner._claim_team_fallback_obligations()
        assert len(claimed) == 1
        assert await runner._deliver_team_fallbacks(claimed) == 1

        send.assert_awaited_once()
        assert send.await_args.kwargs["content"] == td.RECOVERED_FALLBACK_NOTICE
        assert _rows()[0][6] == "fallback"

    @pytest.mark.asyncio
    async def test_live_owner_debt_is_left_alone(self):
        """A live owner may still be mid-turn."""
        runner, roster = _runner()
        td.record_obligation(
            obligation_id="live-1",
            chat_id=_ALLOWED_CHAT,
            root_message_id="701",
            head_message_id="701",
            constituent_ids=("701",),
            owner_profile="default",
            route_reason="semantic_self",
        )

        assert await runner._claim_team_fallback_obligations() == []
        assert _rows()[0][6] == "pending"

    @pytest.mark.asyncio
    async def test_failed_boot_send_releases_rather_than_discharging(self):
        runner, roster = _runner()
        self._record_dead(runner, owner="engineering")
        roster["engineering"].send = AsyncMock(
            return_value=SendResult(success=False, error="Not connected")
        )

        claimed = await runner._claim_team_fallback_obligations()
        assert await runner._deliver_team_fallbacks(claimed) == 0
        assert _rows()[0][6] == "pending"

    @pytest.mark.asyncio
    async def test_routing_disabled_claims_nothing(self):
        runner, roster = _runner()
        self._record_dead(runner)
        runner._telegram_team_routing_enabled = False

        assert await runner._claim_team_fallback_obligations() == []
        assert _rows()[0][6] == "pending"

    @pytest.mark.asyncio
    async def test_disabled_ledger_claims_nothing(self, monkeypatch):
        runner, roster = _runner()
        self._record_dead(runner)
        monkeypatch.setattr(td, "ledger_enabled", lambda config=None: False)

        assert await runner._claim_team_fallback_obligations() == []

    @pytest.mark.asyncio
    async def test_missing_owner_adapter_releases_the_claim(self):
        runner, roster = _runner()
        oid = self._record_dead(runner, owner="engineering")
        claimed = await runner._claim_team_fallback_obligations()
        assert len(claimed) == 1
        # The adapter disappears between claim and send (reconnect churn).
        runner._telegram_team_adapter_for_profile = lambda profile: None

        assert await runner._deliver_team_fallbacks(claimed) == 0
        assert _rows()[0][6] == "pending"

    @pytest.mark.asyncio
    async def test_empty_claim_list_is_a_no_op(self):
        runner, _roster = _runner()
        assert await runner._deliver_team_fallbacks([]) == 0
