"""Durable terminal-outcome ledger for Telegram team dispatch (WAM-31).

``gateway/delivery_ledger.py`` protects an answer that already exists: the
turn finished, the text is in hand, and only the platform ACK is owed. It
cannot protect the failure this module exists for — an accepted human
message whose turn never produced an answer at all. No obligation row is
ever written in that case, because the outbound ledger only records at send
time, so the message is dropped in total silence.

Central Telegram team ingress accepts a message the moment it commits a
root/alias binding and hands the event to exactly one owner. From that
instant the group is owed exactly one public terminal outcome. This module
records that debt durably so it survives a crash, a planned restart, and a
replayed Telegram update.

Rows live in the shared ``state.db`` under the same conventions as the
outbound ledger (WAL, owner pid + process-start-time liveness, capped
attempts, bounded retention), in their own table. The two lifecycles stay
separate on purpose: an inbound debt and an outbound send fail differently
and must never share a sweep.

    record_obligation()      state='pending'          at ingress accept
    close_obligation()       state='answered' |       on a terminal outcome
                                   'clarified' |
                                   'fallback'
    sweep_recoverable()      state='fallback_claimed' owner process is dead
    mark_fallback_delivered()state='fallback'         fallback reached Telegram

Two distinct mechanisms close the "silence" gap, and the split is
deliberate:

- **In-process (precise).** When an owner's turn ends without a terminal
  public outcome, the caller closes the obligation with a deterministic
  fallback immediately. No timeout is guessed, so a legitimately slow agent
  is never interrupted by a second public message.
- **Cross-restart (durable).** ``sweep_recoverable`` claims only rows whose
  owning process is **gone**. A dead process cannot still be thinking, so
  the fallback is unambiguous.

There is deliberately no "alive but overdue" sweep. A long agent turn is
normal, and firing a fallback next to a working agent would produce exactly
the duplicate public reply this project exists to prevent. Rows that outlive
``STALE_AFTER_SECONDS`` under a live owner therefore transition to
``abandoned`` and are logged, never answered: a fallback arriving hours late
in a group chat is worse than a visible gap in the ledger.

Everything here is best-effort. A ledger failure must never block ingress,
delay a send, or consume a message. Callers wrap every call in try/except.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DB_LOCK = threading.Lock()

# Recovery policy knobs (module constants, matching delivery_ledger.py:
# these bounds only matter on the rare recovery path and the whole module
# is already gated by ``gateway.telegram_team_delivery``).
MAX_ATTEMPTS = 3

# A team obligation is only worth answering publicly while the group still
# remembers the message. Past this, the row abandons instead of speaking.
STALE_AFTER_SECONDS = 6 * 60 * 60
_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_ROWS = 500

# States that still owe the group an outcome.
_OPEN_STATES = ("pending", "fallback_claimed")

# States that discharge the debt. ``abandoned`` closes the row without
# speaking (see the module docstring) and is kept briefly for inspection.
TERMINAL_OUTCOMES = frozenset({"answered", "clarified", "fallback"})
_CLOSED_STATES = tuple(sorted(TERMINAL_OUTCOMES)) + ("abandoned",)

# Public fallback copy. Deterministic and honest: it never claims the agent
# failed for a specific reason the gateway cannot actually know, and it
# never implies the message was received but ignored.
FALLBACK_NOTICE = (
    "⚠️ I couldn't complete a reply to this message. "
    "Nothing is queued on my side — please resend or rephrase if you still "
    "need an answer."
)

# Restart-specific variant. The gateway genuinely does not know how far the
# turn got before the process died, so the copy stays explicit about that
# rather than implying a clean failure.
RECOVERED_FALLBACK_NOTICE = (
    "⚠️ The gateway restarted before this message got an answer. "
    "Please resend it if you still need a reply."
)


def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        _initialize_schema(conn)
    except Exception:
        # A PRAGMA/DDL failure after a successful connect() must not leak the
        # just-opened connection back to the caller.
        conn.close()
        raise
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state import apply_wal_with_fallback

    apply_wal_with_fallback(conn, db_label="state.db (telegram_team_delivery)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS telegram_team_obligations (
            obligation_id TEXT PRIMARY KEY,
            chat_id TEXT NOT NULL,
            root_message_id TEXT NOT NULL,
            head_message_id TEXT NOT NULL,
            constituent_ids TEXT NOT NULL,
            owner_profile TEXT NOT NULL,
            route_reason TEXT NOT NULL,
            state TEXT NOT NULL,
            outcome TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            last_error TEXT
        )"""
    )
    # Sweeps and the per-root lookup are the only hot reads; both filter on
    # state first.
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_team_obligations_state
           ON telegram_team_obligations (state, updated_at)"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_team_obligations_root
           ON telegram_team_obligations (chat_id, root_message_id, created_at)"""
    )


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, and ALWAYS close it.

    ``sqlite3.Connection.__enter__``/``__exit__`` only commit or roll back the
    transaction; they do not close the connection. ``with _connect()`` alone
    therefore leaks the connection and its WAL/SHM descriptors until the GC
    runs, which on a long-lived gateway exhausts ``RLIMIT_NOFILE`` (#69567 in
    the cron ledger, and the same footgun is called out in
    ``delivery_ledger._transaction``). ``record_obligation`` runs on every
    accepted team message, so this module must not repeat it.
    """
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _owner_stamp() -> tuple[int, Optional[int]]:
    pid = os.getpid()
    try:
        from gateway.status import get_process_start_time

        return pid, get_process_start_time(pid)
    except Exception:
        return pid, None


def _owner_alive(pid: Any, started_at: Any) -> bool:
    """True when the recorded owning process still exists (pid + start time).

    Delegates the existence probe to ``gateway.status`` for the same reason
    the outbound ledger does: ``os.kill(pid, 0)`` is not a no-op on Windows
    (bpo-14484 maps signal 0 to ``GenerateConsoleCtrlEvent``), so a raw probe
    could Ctrl+C the gateway's own console group.
    """
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        from gateway.status import get_process_start_time

        current_start = get_process_start_time(pid)
    except Exception:
        current_start = None
    if current_start is None:
        try:
            from gateway.status import _pid_exists
        except Exception:
            if os.name == "nt":
                # Never fall back to a raw sig-0 probe on Windows.
                return False
            try:
                os.kill(pid, 0)  # windows-footgun: ok — POSIX-only fallback branch
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            except OSError:
                return False
            return True
        try:
            return bool(_pid_exists(pid))
        except Exception:
            return False
    if started_at is None:
        return True
    try:
        return int(current_start) == int(started_at)
    except (TypeError, ValueError):
        return True


def compute_obligation_id(
    chat_id: str,
    root_message_id: str,
    head_message_id: str,
) -> str:
    """Stable id for one accepted ingress batch.

    Keyed on the batch **head**, not the root: a follow-up reply inside an
    existing root family is its own accepted message and owes its own
    outcome, so keying on the root alone would let a second question be
    silently discharged by the first answer. Determinism is the point — a
    replayed Telegram update after a restart re-derives the same id and is
    recognised as the existing debt rather than opening a new one.
    """
    payload = f"{chat_id}|{root_message_id}|{head_message_id}"
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:24]


def record_obligation(
    *,
    obligation_id: str,
    chat_id: str,
    root_message_id: str,
    head_message_id: str,
    constituent_ids: Sequence[str],
    owner_profile: str,
    route_reason: str,
) -> bool:
    """Record that the group is owed one terminal outcome (state='pending').

    Returns ``True`` only when this call created the row. Re-recording is a
    deliberate no-op rather than an upsert: a replayed update must never
    resurrect an obligation that has already been discharged, which
    ``INSERT OR REPLACE`` (correct for the outbound ledger, where content
    differs per turn) would do here.
    """
    now = time.time()
    pid, started = _owner_stamp()
    try:
        serialized_ids = json.dumps([str(mid) for mid in constituent_ids])
    except Exception:
        serialized_ids = "[]"
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """INSERT INTO telegram_team_obligations
               (obligation_id, chat_id, root_message_id, head_message_id,
                constituent_ids, owner_profile, route_reason, state, outcome,
                attempts, created_at, updated_at, owner_pid, owner_started_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL, 0, ?, ?, ?, ?)
               ON CONFLICT(obligation_id) DO NOTHING""",
            (
                obligation_id,
                str(chat_id),
                str(root_message_id),
                str(head_message_id),
                serialized_ids,
                str(owner_profile),
                str(route_reason)[:128],
                now,
                now,
                pid,
                started,
            ),
        )
        created = bool(cursor.rowcount)
    _prune()
    return created


def close_obligation(obligation_id: str, *, outcome: str) -> bool:
    """Discharge an open obligation exactly once.

    Returns ``True`` only for the call that actually closed the row. Two
    adapters racing the same root therefore cannot both believe they owe the
    public reply, which is what keeps "one terminal outcome" true across a
    duplicated adapter or a replayed update.

    Closing is permitted from ``fallback_claimed`` as well as ``pending``: a
    real answer that lands just after a fallback was claimed still discharges
    the debt, and the caller decides whether the fallback send is still worth
    making.
    """
    if outcome not in TERMINAL_OUTCOMES:
        return False
    placeholders = ", ".join("?" for _ in _OPEN_STATES)
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            f"""UPDATE telegram_team_obligations
                SET state=?, outcome=?, updated_at=?
                WHERE obligation_id=? AND state IN ({placeholders})""",
            (outcome, outcome, time.time(), obligation_id, *_OPEN_STATES),
        )
    return bool(cursor.rowcount)


def open_obligations_for_root(
    chat_id: str,
    root_message_id: str,
) -> List[Dict[str, Any]]:
    """Return still-open obligations for one root, oldest first.

    The turn-completion hook knows the session scope
    (``telegram-team:<chat>:<root>``) but not necessarily which batch head
    triggered the turn, so it resolves the debt through this lookup.
    """
    placeholders = ", ".join("?" for _ in _OPEN_STATES)
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            f"""SELECT obligation_id, chat_id, root_message_id, head_message_id,
                       owner_profile, route_reason, state, attempts, created_at
                FROM telegram_team_obligations
                WHERE chat_id=? AND root_message_id=?
                  AND state IN ({placeholders})
                ORDER BY created_at ASC""",
            (str(chat_id), str(root_message_id), *_OPEN_STATES),
        ).fetchall()
    return [
        {
            "obligation_id": r[0],
            "chat_id": r[1],
            "root_message_id": r[2],
            "head_message_id": r[3],
            "owner_profile": r[4],
            "route_reason": r[5],
            "state": r[6],
            "attempts": r[7],
            "created_at": r[8],
        }
        for r in rows
    ]


def sweep_recoverable(
    now: Optional[float] = None,
    *,
    deliverable_profiles: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Claim open obligations owned by dead processes; return them for fallback.

    Claiming atomically re-stamps the owner to THIS process and increments
    ``attempts``, guarded on the previous owner stamp, so two gateways racing
    the same sweep cannot both deliver a fallback into the group.

    Only rows whose owning process is **gone** are claimed. A live owner may
    still be mid-turn, and a fallback posted beside a working agent is the
    duplicate public reply this project exists to prevent.

    ``deliverable_profiles`` restricts claiming to owner profiles whose
    adapter is actually connected this boot. ``attempts`` is the fallback
    budget and must only be spent on a real send, so a profile that failed to
    connect would otherwise burn one attempt per boot and abandon having
    never spoken once. Its rows are left for a later boot; the stale cutoff
    still bounds them.
    """
    now = now if now is not None else time.time()
    stale_cutoff = now - STALE_AFTER_SECONDS
    pid, started = _owner_stamp()
    if started is None:
        # Without a start time this process cannot prove its own identity on
        # the guarded UPDATE, so a concurrent sweep could double-claim.
        return []

    claimed: List[Dict[str, Any]] = []
    placeholders = ", ".join("?" for _ in _OPEN_STATES)
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            f"""SELECT obligation_id, chat_id, root_message_id, head_message_id,
                       owner_profile, route_reason, state, attempts,
                       created_at, owner_pid, owner_started_at
                FROM telegram_team_obligations
                WHERE state IN ({placeholders})
                ORDER BY created_at ASC""",
            _OPEN_STATES,
        ).fetchall()

        for row in rows:
            (
                obligation_id,
                chat_id,
                root_message_id,
                head_message_id,
                owner_profile,
                route_reason,
                state,
                attempts,
                created_at,
                owner_pid,
                owner_started_at,
            ) = row

            if _owner_alive(owner_pid, owner_started_at):
                continue

            if attempts >= MAX_ATTEMPTS or created_at < stale_cutoff:
                # Poison or too late to speak. Close without answering and
                # leave the row visible for inspection until retention prunes.
                conn.execute(
                    """UPDATE telegram_team_obligations
                       SET state='abandoned', updated_at=?, last_error=?
                       WHERE obligation_id=? AND state IN (?, ?)""",
                    (
                        now,
                        (
                            "attempts exhausted"
                            if attempts >= MAX_ATTEMPTS
                            else "stale before any fallback"
                        ),
                        obligation_id,
                        *_OPEN_STATES,
                    ),
                )
                logger.warning(
                    "Telegram team obligation %s abandoned without a public "
                    "outcome (owner profile '%s', attempts %s)",
                    obligation_id,
                    owner_profile,
                    attempts,
                )
                continue

            if (
                deliverable_profiles is not None
                and owner_profile not in deliverable_profiles
            ):
                continue

            cursor = conn.execute(
                """UPDATE telegram_team_obligations
                   SET state='fallback_claimed', attempts=attempts + 1,
                       updated_at=?, owner_pid=?, owner_started_at=?
                   WHERE obligation_id=? AND state=?
                     AND owner_pid IS ? AND owner_started_at IS ?""",
                (
                    now,
                    pid,
                    started,
                    obligation_id,
                    state,
                    owner_pid,
                    owner_started_at,
                ),
            )
            if not cursor.rowcount:
                # Lost the race to another sweep; leave it to the winner.
                continue

            claimed.append(
                {
                    "obligation_id": obligation_id,
                    "chat_id": chat_id,
                    "root_message_id": root_message_id,
                    "head_message_id": head_message_id,
                    "owner_profile": owner_profile,
                    "route_reason": route_reason,
                    "attempts": attempts + 1,
                    "created_at": created_at,
                    "notice": RECOVERED_FALLBACK_NOTICE,
                }
            )
    return claimed


def claim_fallback(obligation_id: str) -> bool:
    """Claim the right to speak a fallback for a still-open obligation.

    The in-process counterpart to ``sweep_recoverable``: a turn that ended
    without a public outcome claims here, sends, then calls
    ``mark_fallback_delivered`` (or ``release_fallback_claim`` if the send
    failed). Claiming *before* sending is what makes the fallback
    exactly-once — closing after a successful send would leave a crash
    between the two able to send a second fallback on the next boot, and
    closing without a claim would mark the debt discharged even when the
    send failed, turning a failure into a silent drop.

    Unlike the sweep this does not require a dead owner: the caller *is* the
    owner and has just finished the turn.
    """
    now = time.time()
    pid, started = _owner_stamp()
    if started is None:
        # Without a start time the matching release/deliver calls cannot
        # prove this process owns the claim. Fail closed rather than strand
        # the row in ``fallback_claimed``.
        return False
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """UPDATE telegram_team_obligations
               SET state='fallback_claimed', attempts=attempts + 1,
                   updated_at=?, owner_pid=?, owner_started_at=?
               WHERE obligation_id=? AND state='pending'""",
            (now, pid, started, obligation_id),
        )
    return bool(cursor.rowcount)


def mark_fallback_delivered(obligation_id: str) -> bool:
    """Close a claimed row after its fallback actually reached Telegram."""
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """UPDATE telegram_team_obligations
               SET state='fallback', outcome='fallback', updated_at=?
               WHERE obligation_id=? AND state='fallback_claimed'""",
            (time.time(), obligation_id),
        )
    return bool(cursor.rowcount)


def release_fallback_claim(obligation_id: str, error: str = "") -> bool:
    """Return an unsent claim to ``pending`` without spending an attempt.

    A claim that never reached a send must not consume the bounded fallback
    budget, or a profile that reconnects slowly would abandon having never
    spoken. Fail-closed to this exact process instance and the claimed state.
    """
    pid, started = _owner_stamp()
    if started is None:
        return False
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """UPDATE telegram_team_obligations
               SET state='pending', attempts=CASE
                       WHEN attempts > 0 THEN attempts - 1 ELSE 0 END,
                   updated_at=?, last_error=?
               WHERE obligation_id=? AND state='fallback_claimed'
                 AND owner_pid IS ? AND owner_started_at IS ?""",
            (time.time(), error[:500] if error else None, obligation_id, pid, started),
        )
    return bool(cursor.rowcount)


def mark_abandoned(obligation_id: str, error: str = "") -> bool:
    """Close a row without speaking (operator/None-deliverable escape hatch)."""
    placeholders = ", ".join("?" for _ in _OPEN_STATES)
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            f"""UPDATE telegram_team_obligations
                SET state='abandoned', updated_at=?, last_error=?
                WHERE obligation_id=? AND state IN ({placeholders})""",
            (time.time(), error[:500] if error else None, obligation_id, *_OPEN_STATES),
        )
    return bool(cursor.rowcount)


def _prune(now: Optional[float] = None) -> None:
    now = now if now is not None else time.time()
    cutoff = now - _RETENTION_SECONDS
    try:
        with _transaction() as conn:
            conn.execute(
                f"""DELETE FROM telegram_team_obligations
                    WHERE state IN ({", ".join("?" for _ in _CLOSED_STATES)})
                      AND updated_at < ?""",
                (*_CLOSED_STATES, cutoff),
            )
            total = conn.execute(
                "SELECT COUNT(*) FROM telegram_team_obligations"
            ).fetchone()[0]
            excess = max(0, total - _MAX_ROWS)
            if excess:
                # Drop closed rows before open ones: an undischarged debt is
                # the only row here worth keeping under pressure.
                conn.execute(
                    """DELETE FROM telegram_team_obligations
                       WHERE obligation_id IN (
                         SELECT obligation_id FROM telegram_team_obligations
                         ORDER BY CASE state
                                    WHEN 'answered' THEN 0
                                    WHEN 'clarified' THEN 0
                                    WHEN 'fallback' THEN 0
                                    WHEN 'abandoned' THEN 1
                                    ELSE 2
                                  END, updated_at ASC
                         LIMIT ?)""",
                    (excess,),
                )
    except Exception:
        logger.debug("Telegram team obligation prune failed", exc_info=True)


def ledger_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Read the ``gateway.telegram_team_delivery`` config gate (default on)."""
    try:
        if config is None:
            from hermes_cli.config import load_config

            config = load_config()
        gw = config.get("gateway") or {}
        value = gw.get("telegram_team_delivery", True)
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0", "no", "off"}
        return bool(value)
    except Exception:
        return True


def debug_rows(limit: int = 20) -> str:
    """Human-readable dump for ad-hoc inspection (sqlite3-free path)."""
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, chat_id, root_message_id, head_message_id,
                      owner_profile, route_reason, state, outcome, attempts,
                      created_at, updated_at, last_error
               FROM telegram_team_obligations
               ORDER BY updated_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return json.dumps(
        [
            {
                "id": r[0],
                "chat_id": r[1],
                "root": r[2],
                "head": r[3],
                "owner": r[4],
                "reason": r[5],
                "state": r[6],
                "outcome": r[7],
                "attempts": r[8],
                "created_at": r[9],
                "updated_at": r[10],
                "last_error": r[11],
            }
            for r in rows
        ],
        indent=2,
    )
