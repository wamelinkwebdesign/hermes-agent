"""Tests for the Telegram team terminal-outcome ledger (WAM-31).

Covers the acceptance criteria of the exactly-once terminal delivery task:

- one terminal outcome per accepted message, never zero and never two
- obligations survive restarts and are idempotent across replayed updates
- a replayed update can never resurrect an already-discharged obligation
- failed specialist delivery cannot become silent success: the fallback is
  claimed atomically and delivered at most once
- duplicate adapters racing the same root cannot both believe they owe the
  public reply
- state transitions, the attempts cap, the stale cutoff and retention are
  all bounded

The dead-owner cases are the restart contract: only a process that is
genuinely gone may have its debt taken over, because a live owner may still
be mid-turn.
"""

import os
import sqlite3
import threading
import time

import pytest

from gateway import telegram_team_delivery as td


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """Isolated state.db per test."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(td, "_db_path", lambda: home / "state.db")
    yield


def _record(
    oid="ob-1",
    chat_id="-1001",
    root="100",
    head="100",
    constituents=("100",),
    owner="ace",
    reason="unaddressed_ingress",
):
    return td.record_obligation(
        obligation_id=oid,
        chat_id=chat_id,
        root_message_id=root,
        head_message_id=head,
        constituent_ids=constituents,
        owner_profile=owner,
        route_reason=reason,
    )


def _row(oid):
    with td._connect() as conn:
        r = conn.execute(
            """SELECT state, outcome, attempts, owner_pid, owner_started_at,
                      last_error, constituent_ids
               FROM telegram_team_obligations WHERE obligation_id=?""",
            (oid,),
        ).fetchone()
    return None if r is None else {
        "state": r[0],
        "outcome": r[1],
        "attempts": r[2],
        "owner_pid": r[3],
        "owner_started_at": r[4],
        "last_error": r[5],
        "constituent_ids": r[6],
    }


def _make_owner_dead(oid):
    """Make the row look like it belongs to a process that no longer exists."""
    with td._connect() as conn:
        with conn:
            conn.execute(
                "UPDATE telegram_team_obligations SET owner_pid=999999999, "
                "owner_started_at=1 WHERE obligation_id=?",
                (oid,),
            )


def _age_row(oid, seconds):
    with td._connect() as conn:
        with conn:
            conn.execute(
                "UPDATE telegram_team_obligations SET created_at=? "
                "WHERE obligation_id=?",
                (time.time() - seconds, oid),
            )


def _set_attempts(oid, attempts):
    with td._connect() as conn:
        with conn:
            conn.execute(
                "UPDATE telegram_team_obligations SET attempts=? "
                "WHERE obligation_id=?",
                (attempts, oid),
            )


class TestObligationIdentity:
    def test_id_is_deterministic(self):
        first = td.compute_obligation_id("-1001", "100", "100")
        second = td.compute_obligation_id("-1001", "100", "100")
        assert first == second

    def test_follow_up_in_same_root_gets_its_own_obligation(self):
        """A second question under one root owes its own answer.

        Keying on the root alone would let the first answer silently
        discharge the follow-up.
        """
        root_id = td.compute_obligation_id("-1001", "100", "100")
        follow_up_id = td.compute_obligation_id("-1001", "100", "105")
        assert root_id != follow_up_id

    def test_same_message_ids_in_different_chats_never_collide(self):
        assert td.compute_obligation_id("-1001", "100", "100") != (
            td.compute_obligation_id("-1002", "100", "100")
        )


class TestRecording:
    def test_record_creates_pending_row(self):
        assert _record() is True
        row = _row("ob-1")
        assert row["state"] == "pending"
        assert row["outcome"] is None
        assert row["attempts"] == 0

    def test_duplicate_record_is_idempotent(self):
        assert _record() is True
        assert _record() is False
        row = _row("ob-1")
        assert row["state"] == "pending"

    def test_replay_cannot_resurrect_a_discharged_obligation(self):
        """The replay contract: a duplicate Telegram update after the answer
        went out must not reopen the debt and earn a second public reply."""
        _record()
        assert td.close_obligation("ob-1", outcome="answered") is True
        assert _record() is False
        assert _row("ob-1")["state"] == "answered"

    def test_constituent_ids_are_persisted_for_provenance(self):
        _record(constituents=("100", "101", "102"))
        assert _row("ob-1")["constituent_ids"] == '["100", "101", "102"]'

    def test_unserializable_constituents_do_not_break_recording(self):
        """Ledger failures must never block ingress."""
        assert _record(constituents=[object()]) is True
        assert _row("ob-1")["state"] == "pending"

    def test_route_reason_is_bounded(self):
        _record(reason="x" * 500)
        with td._connect() as conn:
            reason = conn.execute(
                "SELECT route_reason FROM telegram_team_obligations "
                "WHERE obligation_id=?",
                ("ob-1",),
            ).fetchone()[0]
        assert len(reason) == 128


class TestClosing:
    @pytest.mark.parametrize("outcome", ["answered", "clarified", "fallback"])
    def test_each_terminal_outcome_closes(self, outcome):
        _record()
        assert td.close_obligation("ob-1", outcome=outcome) is True
        row = _row("ob-1")
        assert row["state"] == outcome
        assert row["outcome"] == outcome

    def test_only_the_first_close_wins(self):
        """Two adapters racing one root must not both owe the public reply."""
        _record()
        assert td.close_obligation("ob-1", outcome="answered") is True
        assert td.close_obligation("ob-1", outcome="clarified") is False
        assert _row("ob-1")["outcome"] == "answered"

    def test_invalid_outcome_is_rejected(self):
        _record()
        assert td.close_obligation("ob-1", outcome="delivered") is False
        assert td.close_obligation("ob-1", outcome="abandoned") is False
        assert _row("ob-1")["state"] == "pending"

    def test_unknown_obligation_closes_nothing(self):
        assert td.close_obligation("missing", outcome="answered") is False

    def test_late_answer_still_closes_a_claimed_fallback(self):
        """The fallback was claimed but the real answer arrived; the debt is
        discharged and the caller decides whether the fallback still ships."""
        _record()
        _make_owner_dead("ob-1")
        assert len(td.sweep_recoverable()) == 1
        assert _row("ob-1")["state"] == "fallback_claimed"
        assert td.close_obligation("ob-1", outcome="answered") is True

    def test_concurrent_closers_produce_exactly_one_winner(self):
        _record()
        results = []
        barrier = threading.Barrier(8)

        def _close():
            barrier.wait()
            results.append(td.close_obligation("ob-1", outcome="answered"))

        threads = [threading.Thread(target=_close) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results.count(True) == 1


class TestOpenObligationsForRoot:
    def test_returns_only_open_rows_oldest_first(self):
        _record(oid="ob-1", head="100")
        _record(oid="ob-2", head="101")
        _record(oid="ob-3", head="102")
        _age_row("ob-1", 30)
        _age_row("ob-2", 20)
        _age_row("ob-3", 10)
        td.close_obligation("ob-2", outcome="answered")

        rows = td.open_obligations_for_root("-1001", "100")
        assert [r["obligation_id"] for r in rows] == ["ob-1", "ob-3"]

    def test_other_roots_and_chats_are_not_returned(self):
        _record(oid="ob-1", chat_id="-1001", root="100")
        _record(oid="ob-2", chat_id="-1001", root="200", head="200")
        _record(oid="ob-3", chat_id="-1002", root="100", head="100")

        rows = td.open_obligations_for_root("-1001", "100")
        assert [r["obligation_id"] for r in rows] == ["ob-1"]

    def test_claimed_rows_still_count_as_open(self):
        _record()
        _make_owner_dead("ob-1")
        td.sweep_recoverable()
        assert len(td.open_obligations_for_root("-1001", "100")) == 1


class TestSweepRecoverable:
    def test_live_owner_is_never_claimed(self):
        """A live owner may still be mid-turn; speaking beside it would be
        the duplicate public reply this project exists to prevent."""
        _record()
        assert td.sweep_recoverable() == []
        assert _row("ob-1")["state"] == "pending"

    def test_dead_owner_is_claimed_and_restamped(self):
        _record()
        _make_owner_dead("ob-1")
        claimed = td.sweep_recoverable()
        assert len(claimed) == 1
        assert claimed[0]["obligation_id"] == "ob-1"
        assert claimed[0]["notice"] == td.RECOVERED_FALLBACK_NOTICE
        row = _row("ob-1")
        assert row["state"] == "fallback_claimed"
        assert row["attempts"] == 1
        assert row["owner_pid"] == os.getpid()

    def test_second_sweep_does_not_reclaim_its_own_live_claim(self):
        _record()
        _make_owner_dead("ob-1")
        assert len(td.sweep_recoverable()) == 1
        assert td.sweep_recoverable() == []
        assert _row("ob-1")["attempts"] == 1

    def test_attempts_cap_abandons_without_speaking(self):
        _record()
        _make_owner_dead("ob-1")
        _set_attempts("ob-1", td.MAX_ATTEMPTS)
        assert td.sweep_recoverable() == []
        row = _row("ob-1")
        assert row["state"] == "abandoned"
        assert row["last_error"] == "attempts exhausted"

    def test_stale_row_abandons_rather_than_answering_late(self):
        """A fallback arriving hours later in a group chat is worse than a
        visible gap in the ledger."""
        _record()
        _make_owner_dead("ob-1")
        _age_row("ob-1", td.STALE_AFTER_SECONDS + 60)
        assert td.sweep_recoverable() == []
        row = _row("ob-1")
        assert row["state"] == "abandoned"
        assert row["last_error"] == "stale before any fallback"

    def test_undeliverable_profile_is_left_untouched(self):
        """attempts is the fallback budget and must only be spent on a real
        send, or a disconnected profile abandons having never spoken."""
        _record(owner="collectfyi")
        _make_owner_dead("ob-1")
        assert td.sweep_recoverable(deliverable_profiles={"ace"}) == []
        row = _row("ob-1")
        assert row["state"] == "pending"
        assert row["attempts"] == 0

    def test_deliverable_profile_is_claimed(self):
        _record(owner="collectfyi")
        _make_owner_dead("ob-1")
        claimed = td.sweep_recoverable(deliverable_profiles={"ace", "collectfyi"})
        assert [c["obligation_id"] for c in claimed] == ["ob-1"]

    def test_missing_own_start_stamp_fails_closed(self, monkeypatch):
        """Without a start time this process cannot prove its identity on the
        guarded UPDATE, so a concurrent sweep could double-claim."""
        _record()
        _make_owner_dead("ob-1")
        monkeypatch.setattr(td, "_owner_stamp", lambda: (os.getpid(), None))
        assert td.sweep_recoverable() == []
        assert _row("ob-1")["state"] == "pending"

    def test_other_live_owner_is_neither_claimed_nor_abandoned(self, monkeypatch):
        _record()
        with td._connect() as conn:
            with conn:
                conn.execute(
                    "UPDATE telegram_team_obligations SET owner_pid=?, "
                    "owner_started_at=? WHERE obligation_id=?",
                    (os.getpid(), 202, "ob-1"),
                )
        monkeypatch.setattr(td, "_owner_stamp", lambda: (54321, 909))
        monkeypatch.setattr(td, "_owner_alive", lambda pid, started: True)
        assert td.sweep_recoverable() == []
        assert _row("ob-1")["state"] == "pending"

    def test_closed_rows_are_never_swept(self):
        _record()
        td.close_obligation("ob-1", outcome="answered")
        _make_owner_dead("ob-1")
        assert td.sweep_recoverable() == []
        assert _row("ob-1")["state"] == "answered"

    def test_concurrent_sweeps_claim_each_row_once(self):
        for index in range(6):
            _record(oid=f"ob-{index}", head=str(100 + index))
            _make_owner_dead(f"ob-{index}")

        results = []
        barrier = threading.Barrier(4)

        def _sweep():
            barrier.wait()
            results.append(td.sweep_recoverable())

        threads = [threading.Thread(target=_sweep) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        claimed_ids = [row["obligation_id"] for batch in results for row in batch]
        assert sorted(claimed_ids) == sorted(f"ob-{i}" for i in range(6))
        assert len(claimed_ids) == len(set(claimed_ids))


class TestInProcessFallbackClaim:
    def test_claim_moves_pending_to_claimed(self):
        _record()
        assert td.claim_fallback("ob-1") is True
        row = _row("ob-1")
        assert row["state"] == "fallback_claimed"
        assert row["attempts"] == 1
        assert row["owner_pid"] == os.getpid()

    def test_claim_is_exactly_once(self):
        """Claiming before sending is what makes the fallback exactly-once."""
        _record()
        assert td.claim_fallback("ob-1") is True
        assert td.claim_fallback("ob-1") is False

    def test_claim_refuses_a_discharged_obligation(self):
        _record()
        td.close_obligation("ob-1", outcome="answered")
        assert td.claim_fallback("ob-1") is False
        assert _row("ob-1")["state"] == "answered"

    def test_claim_refuses_an_unknown_obligation(self):
        assert td.claim_fallback("missing") is False

    def test_claim_fails_closed_without_a_start_stamp(self, monkeypatch):
        """A claim this process cannot later prove it owns would strand the
        row in fallback_claimed forever."""
        _record()
        monkeypatch.setattr(td, "_owner_stamp", lambda: (os.getpid(), None))
        assert td.claim_fallback("ob-1") is False
        assert _row("ob-1")["state"] == "pending"

    def test_claim_then_release_restores_the_budget(self):
        _record()
        td.claim_fallback("ob-1")
        assert td.release_fallback_claim("ob-1", error="send failed") is True
        row = _row("ob-1")
        assert row["state"] == "pending"
        assert row["attempts"] == 0

    def test_failed_send_leaves_the_debt_open_for_a_later_boot(self):
        """A failed fallback send must not become a silent discharge."""
        _record()
        td.claim_fallback("ob-1")
        td.release_fallback_claim("ob-1", error="transport down")
        _make_owner_dead("ob-1")
        assert [c["obligation_id"] for c in td.sweep_recoverable()] == ["ob-1"]

    def test_concurrent_claims_produce_exactly_one_speaker(self):
        _record()
        results = []
        barrier = threading.Barrier(8)

        def _claim():
            barrier.wait()
            results.append(td.claim_fallback("ob-1"))

        threads = [threading.Thread(target=_claim) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results.count(True) == 1
        assert _row("ob-1")["attempts"] == 1


class TestFallbackCompletion:
    def test_delivered_only_from_claimed(self):
        _record()
        assert td.mark_fallback_delivered("ob-1") is False
        _make_owner_dead("ob-1")
        td.sweep_recoverable()
        assert td.mark_fallback_delivered("ob-1") is True
        row = _row("ob-1")
        assert row["state"] == "fallback"
        assert row["outcome"] == "fallback"

    def test_fallback_is_delivered_at_most_once(self):
        _record()
        _make_owner_dead("ob-1")
        td.sweep_recoverable()
        assert td.mark_fallback_delivered("ob-1") is True
        assert td.mark_fallback_delivered("ob-1") is False

    def test_release_returns_to_pending_without_spending_an_attempt(self):
        _record()
        _make_owner_dead("ob-1")
        td.sweep_recoverable()
        assert _row("ob-1")["attempts"] == 1
        assert td.release_fallback_claim("ob-1", error="adapter offline") is True
        row = _row("ob-1")
        assert row["state"] == "pending"
        assert row["attempts"] == 0
        assert row["last_error"] == "adapter offline"

    def test_release_is_fail_closed_to_the_claiming_process(self, monkeypatch):
        _record()
        _make_owner_dead("ob-1")
        td.sweep_recoverable()
        monkeypatch.setattr(td, "_owner_stamp", lambda: (54321, 909))
        assert td.release_fallback_claim("ob-1") is False
        assert _row("ob-1")["state"] == "fallback_claimed"

    def test_release_requires_a_claimed_row(self):
        _record()
        assert td.release_fallback_claim("ob-1") is False

    def test_release_error_is_bounded(self):
        _record()
        _make_owner_dead("ob-1")
        td.sweep_recoverable()
        td.release_fallback_claim("ob-1", error="e" * 900)
        assert len(_row("ob-1")["last_error"]) == 500

    def test_mark_abandoned_closes_an_open_row(self):
        _record()
        assert td.mark_abandoned("ob-1", error="operator drained") is True
        row = _row("ob-1")
        assert row["state"] == "abandoned"
        assert row["last_error"] == "operator drained"

    def test_mark_abandoned_cannot_reopen_a_closed_row(self):
        _record()
        td.close_obligation("ob-1", outcome="answered")
        assert td.mark_abandoned("ob-1") is False
        assert _row("ob-1")["state"] == "answered"


class TestRetention:
    def test_closed_rows_prune_after_retention(self):
        _record()
        td.close_obligation("ob-1", outcome="answered")
        with td._connect() as conn:
            with conn:
                conn.execute(
                    "UPDATE telegram_team_obligations SET updated_at=? "
                    "WHERE obligation_id=?",
                    (time.time() - td._RETENTION_SECONDS - 60, "ob-1"),
                )
        td._prune()
        assert _row("ob-1") is None

    def test_open_rows_survive_retention(self):
        _record()
        with td._connect() as conn:
            with conn:
                conn.execute(
                    "UPDATE telegram_team_obligations SET updated_at=? "
                    "WHERE obligation_id=?",
                    (time.time() - td._RETENTION_SECONDS - 60, "ob-1"),
                )
        td._prune()
        assert _row("ob-1") is not None

    def test_row_cap_drops_closed_rows_before_open_ones(self, monkeypatch):
        monkeypatch.setattr(td, "_MAX_ROWS", 3)
        for index in range(3):
            _record(oid=f"closed-{index}", head=str(200 + index))
            td.close_obligation(f"closed-{index}", outcome="answered")
        _record(oid="open-1", head="300")
        td._prune()
        assert _row("open-1") is not None
        with td._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM telegram_team_obligations"
            ).fetchone()[0]
        assert total <= 3

    def test_prune_failure_is_swallowed(self, monkeypatch):
        """Ledger failures must never propagate into ingress."""
        def _boom():
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(td, "_transaction", _boom)
        td._prune()


class TestConfigGate:
    def test_default_is_on(self):
        assert td.ledger_enabled({}) is True
        assert td.ledger_enabled({"gateway": {}}) is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", "OFF", False])
    def test_disabled_values(self, value):
        assert td.ledger_enabled({"gateway": {"telegram_team_delivery": value}}) is False

    @pytest.mark.parametrize("value", ["true", "1", "yes", True])
    def test_enabled_values(self, value):
        assert td.ledger_enabled({"gateway": {"telegram_team_delivery": value}}) is True

    def test_unreadable_config_defaults_on(self, monkeypatch):
        class _Hostile:
            def get(self, *args, **kwargs):
                raise RuntimeError("config exploded")

        assert td.ledger_enabled(_Hostile()) is True


class TestConnectionHygiene:
    def test_transaction_closes_the_connection(self):
        with td._transaction() as conn:
            conn.execute("SELECT 1")
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    def test_schema_failure_does_not_leak_the_connection(self, monkeypatch):
        opened = []
        real_connect = sqlite3.connect

        def _tracking_connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            opened.append(conn)
            return conn

        monkeypatch.setattr(sqlite3, "connect", _tracking_connect)
        monkeypatch.setattr(
            td,
            "_initialize_schema",
            lambda conn: (_ for _ in ()).throw(sqlite3.OperationalError("boom")),
        )
        with pytest.raises(sqlite3.OperationalError):
            td._connect()
        assert len(opened) == 1
        with pytest.raises(sqlite3.ProgrammingError):
            opened[0].execute("SELECT 1")


class TestOwnerAliveProbe:
    """``_owner_alive``'s no-start-time fallback must route through
    ``gateway.status``: ``os.kill(pid, 0)`` is not a no-op on Windows
    (bpo-14484), where it would Ctrl+C the gateway's own console group."""

    def _no_start_time(self, monkeypatch):
        from gateway import status

        monkeypatch.setattr(status, "get_process_start_time", lambda pid: None)

    def test_alive_when_pid_exists(self, monkeypatch):
        from gateway import status

        self._no_start_time(monkeypatch)
        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        assert td._owner_alive(12345, 999) is True

    def test_dead_when_pid_gone(self, monkeypatch):
        from gateway import status

        self._no_start_time(monkeypatch)
        monkeypatch.setattr(status, "_pid_exists", lambda pid: False)
        assert td._owner_alive(12345, 999) is False

    def test_raw_os_kill_probe_never_used(self, monkeypatch):
        from gateway import status

        self._no_start_time(monkeypatch)
        calls = []
        monkeypatch.setattr(status, "_pid_exists", lambda pid: calls.append(pid) or True)

        def _forbidden(*args, **kwargs):
            raise AssertionError("raw os.kill probe must not be used")

        monkeypatch.setattr(os, "kill", _forbidden)
        assert td._owner_alive(12345, 999) is True
        assert calls == [12345]

    @pytest.mark.parametrize("pid", [None, 0, "", "not-a-pid"])
    def test_unusable_pid_is_dead(self, pid):
        assert td._owner_alive(pid, 1) is False

    def test_mismatched_start_time_is_dead(self, monkeypatch):
        from gateway import status

        monkeypatch.setattr(status, "get_process_start_time", lambda pid: 500)
        assert td._owner_alive(12345, 999) is False
        assert td._owner_alive(12345, 500) is True


class TestDebugRows:
    def test_dump_is_json_and_bounded(self):
        import json

        for index in range(5):
            _record(oid=f"ob-{index}", head=str(100 + index))
        parsed = json.loads(td.debug_rows(limit=3))
        assert len(parsed) == 3
        assert {"id", "chat_id", "root", "head", "owner", "state"} <= set(parsed[0])
