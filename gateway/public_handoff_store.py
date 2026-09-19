"""Bounded public-request spool, separate from private CLI session handoffs.

The local user's profile homes are the trust boundary, not model-writable wire
provenance. No general enqueue API/tool is exposed. DELETE journaling avoids
unsafe WAL versions on embedded runtimes. Every side-effect claim is a CAS.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time


# A fallback with no confirmed receipt is ambiguous even after a hard process exit.
_UNCERTAIN_ROW = (
    "(state='delivery_uncertain' OR (fallback_started_at IS NOT NULL "
    "AND COALESCE(json_extract(fallback_receipt,'$.state'),'') != 'delivered'))"
)

# Single clock: bounds generation AND delivery. The owner agent's run budget is
# derived from expires_at (minus a send margin) so a legitimate run can never
# outlive its lease and trip the recovery reaper. Producers must keep their
# enqueue deadline strictly below this cap; consumers reject rows above it.
MAX_REQUEST_TTL_SECONDS = 600


@dataclass(frozen=True)
class PublicRequest:
    version: int
    request_id: str
    source_profile: str
    target_profile: str
    platform: str
    chat_id: str
    thread_id: str | None
    message_id: str
    sender_id: str
    brief: str
    created_at: float
    expires_at: float
    root_request_id: str
    reply_to_message_id: str | None = None
    reply_to_sender_id: str | None = None

    def __post_init__(self):
        numeric = lambda value: isinstance(value, str) and re.fullmatch(r"[1-9][0-9]{0,19}", value)
        if (type(self.version) is not int or self.version != 1
                or self.source_profile not in {"default", "engineering"}
                or self.target_profile != "engineering" or self.platform != "telegram"
                or not isinstance(self.chat_id, str) or not re.fullmatch(r"-[1-9][0-9]{0,19}", self.chat_id)
                or not numeric(self.message_id) or not numeric(self.sender_id)
                or (self.thread_id is not None and not numeric(self.thread_id))
                or self.request_id != request_identity(self.chat_id, self.thread_id, self.message_id)
                or not isinstance(self.root_request_id, str)
                or not re.fullmatch(r"public-[a-f0-9]{64}", self.root_request_id)
                or not isinstance(self.brief, str) or not 0 < len(self.brief) <= 6000):
            raise ValueError("invalid_public_request")
        if (not all(type(t) in {int, float} and math.isfinite(t) and t > 0
                    for t in (self.created_at, self.expires_at))
                or not 0 < self.expires_at - self.created_at <= MAX_REQUEST_TTL_SECONDS):
            raise ValueError("invalid_public_deadline")
        if self.root_request_id == self.request_id:
            if self.source_profile != "default" or self.reply_to_message_id is not None or self.reply_to_sender_id is not None:
                raise ValueError("invalid_root_provenance")
        elif self.source_profile != "engineering" or not numeric(self.reply_to_message_id) or not numeric(self.reply_to_sender_id):
            raise ValueError("invalid_reply_provenance")


def request_identity(chat_id, thread_id, message_id):
    key = json.dumps(["telegram", chat_id, thread_id, message_id], separators=(",", ":"))
    return "public-" + hashlib.sha256(key.encode()).hexdigest()


class PublicHandoffStore:
    request_type = PublicRequest

    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self):
        # Do not create a missing profile home, follow a replacement symlink, or copy credentials.
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        os.close(fd)
        db = sqlite3.connect(self.path, timeout=3, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("""CREATE TABLE IF NOT EXISTS public_handoffs (
                request_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                root_request_id TEXT NOT NULL, state TEXT NOT NULL,
                created_at REAL NOT NULL, expires_at REAL NOT NULL,
                send_started_at REAL, answer TEXT, receipt TEXT, error TEXT,
                session_id TEXT, session_key TEXT, group_session_id TEXT,
                fallback_started_at REAL, fallback_receipt TEXT
            )""")
            yield db
        finally:
            db.close()

    @classmethod
    def decode(cls, row):
        if row is None:
            return None
        result = dict(row)
        result.update(asdict(cls.request_type(**json.loads(result.pop("payload")))))
        if result["receipt"]:
            result["receipt"] = json.loads(result["receipt"])
        return result

    def enqueue(self, request: PublicRequest):
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO public_handoffs "
                "(request_id,payload,root_request_id,state,created_at,expires_at) VALUES (?,?,?,'accepted',?,?)",
                (request.request_id, json.dumps(asdict(request)), request.root_request_id,
                 request.created_at, request.expires_at),
            )
            return self.decode(db.execute("SELECT * FROM public_handoffs WHERE request_id=?", (request.request_id,)).fetchone())

    def get(self, request_id):
        with self.connect() as db:
            return self.decode(db.execute("SELECT * FROM public_handoffs WHERE request_id=?", (request_id,)).fetchone())

    def pending(self):
        with self.connect() as db:
            return [self.decode(r) for r in db.execute(
                "SELECT * FROM public_handoffs WHERE state='accepted' ORDER BY created_at,request_id LIMIT 32")]

    def claim(self, request_id, *, expected_owner=None, expected_reason=None):
        with self.connect() as db:
            return db.execute(
                "UPDATE public_handoffs SET state='running' WHERE request_id=? AND state='accepted' "
                "AND expires_at>? "
                "AND (? IS NULL OR json_extract(payload,'$.target_profile')=?) "
                "AND (? IS NULL OR json_extract(payload,'$.reason_code')=?) "
                "AND NOT EXISTS (SELECT 1 FROM public_handoffs other "
                "WHERE other.root_request_id=public_handoffs.root_request_id "
                f"AND (other.state='running' OR {_UNCERTAIN_ROW}))",
                (request_id, time.time(), expected_owner, expected_owner, expected_reason, expected_reason),
            ).rowcount == 1

    def bind_session(self, request_id, session_id, session_key, group_session_id):
        with self.connect() as db:
            return db.execute(
                "UPDATE public_handoffs SET session_id=?, session_key=?, group_session_id=? "
                "WHERE request_id=? AND state='running' AND send_started_at IS NULL AND expires_at>?",
                (session_id, session_key, group_session_id, request_id, time.time()),
            ).rowcount == 1

    def begin_send(self, request_id, answer):
        with self.connect() as db:
            return db.execute(
                "UPDATE public_handoffs SET send_started_at=?, answer=? WHERE request_id=? "
                "AND state='running' AND send_started_at IS NULL AND expires_at>? "
                "AND session_id IS NOT NULL AND group_session_id IS NOT NULL",
                (time.time(), answer, request_id, time.time()),
            ).rowcount == 1

    def delivered(self, request_id, receipt):
        with self.connect() as db:
            return db.execute(
                "UPDATE public_handoffs SET state='delivered', receipt=? WHERE request_id=? "
                "AND state='running' AND send_started_at IS NOT NULL AND expires_at>?",
                (json.dumps(receipt), request_id, time.time()),
            ).rowcount == 1

    def fail(self, request_id, reason):
        with self.connect() as db:
            return db.execute(
                "UPDATE public_handoffs SET state=CASE WHEN send_started_at IS NULL "
                "THEN 'failed_definitive' ELSE 'delivery_uncertain' END, error=? "
                "WHERE request_id=? AND state IN ('accepted','running')",
                (reason, request_id),
            ).rowcount == 1

    def expire(self):
        with self.connect() as db:
            db.execute(
                "UPDATE public_handoffs SET state=CASE WHEN send_started_at IS NULL "
                "THEN 'failed_definitive' ELSE 'delivery_uncertain' END, error='deadline' "
                "WHERE state IN ('accepted','running') AND expires_at<=?", (time.time(),))

    def reply_owner(self, chat_id, thread_id, message_id, sender_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM public_handoffs WHERE json_extract(payload,'$.chat_id')=? "
                "AND json_extract(payload,'$.thread_id') IS ? AND ("
                "(json_extract(payload,'$.message_id')=? AND json_extract(payload,'$.sender_id')=? "
                ") OR "
                "(state='delivered' AND json_extract(receipt,'$.message_id')=? "
                "AND json_extract(receipt,'$.bot_id')=?)) LIMIT 1",
                (chat_id, thread_id, message_id, sender_id, message_id, sender_id),
            ).fetchone()
            return self.decode(row)

    def cancel(self, root_request_id, sender_id):
        with self.connect() as db:
            db.execute(
                "UPDATE public_handoffs SET state=CASE WHEN send_started_at IS NULL "
                "THEN 'failed_definitive' ELSE 'delivery_uncertain' END, error='cancelled' "
                "WHERE root_request_id=? AND state IN ('accepted','running') AND EXISTS "
                "(SELECT 1 FROM public_handoffs root WHERE root.request_id=? "
                "AND json_extract(root.payload,'$.sender_id')=?)",
                (root_request_id, root_request_id, sender_id),
            )

    def conversation_uncertain(self, root_request_id):
        with self.connect() as db:
            return db.execute(
                "SELECT 1 FROM public_handoffs WHERE root_request_id=? "
                f"AND {_UNCERTAIN_ROW} LIMIT 1", (root_request_id,)
            ).fetchone() is not None

    def history(self, root_request_id):
        with self.connect() as db:
            rows = list(db.execute(
                "SELECT * FROM public_handoffs WHERE root_request_id=? AND state='delivered' "
                "ORDER BY created_at DESC, request_id DESC LIMIT 4", (root_request_id,)))
        history = []
        for raw in reversed(rows):
            row = self.decode(raw)
            assert row is not None
            history.extend([{"role": "user", "content": row["brief"]},
                            {"role": "assistant", "content": row["answer"]}])
        return history

    def fallback_pending(self):
        with self.connect() as db:
            return [self.decode(r) for r in db.execute(
                "SELECT * FROM public_handoffs WHERE state='failed_definitive' "
                "AND fallback_started_at IS NULL ORDER BY created_at LIMIT 32")]

    def claim_fallback(self, request_id):
        with self.connect() as db:
            return db.execute(
                "UPDATE public_handoffs SET fallback_started_at=? WHERE request_id=? "
                "AND state='failed_definitive' AND fallback_started_at IS NULL "
                "AND NOT EXISTS (SELECT 1 FROM public_handoffs other "
                "WHERE other.root_request_id=public_handoffs.root_request_id "
                f"AND (other.state='running' OR {_UNCERTAIN_ROW}))",
                (time.time(), request_id),
            ).rowcount == 1

    def finish_fallback(self, request_id, receipt):
        with self.connect() as db:
            db.execute("UPDATE public_handoffs SET fallback_receipt=? WHERE request_id=? "
                       "AND fallback_started_at IS NOT NULL AND fallback_receipt IS NULL",
                       (json.dumps(receipt), request_id))
