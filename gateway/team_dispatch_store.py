"""Team envelopes reuse the public handoff CAS and uncertain-send quarantine."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import re
import time

from gateway.public_handoff_store import PublicHandoffStore, PublicRequest
from gateway.team_dispatch_policy import PROFILES, POSITIVE_ID


@dataclass(frozen=True)
class TeamRequest(PublicRequest):
    fingerprint: str = ""
    reason_code: str = "pending"
    reassigned: bool = False
    input_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.input_ids:
            object.__setattr__(self, "input_ids", (self.message_id,))
        # Keep the inherited wire bounds without the old Engineering-only pilot routing.
        PublicRequest(1, self.request_id, "default", "engineering", self.platform,
                      self.chat_id, self.thread_id, self.message_id, self.sender_id,
                      self.brief, self.created_at, self.expires_at, self.request_id)
        if (self.version != 2 or self.source_profile not in PROFILES or self.target_profile not in PROFILES
                or not re.fullmatch(r"[a-f0-9]{64}", self.fingerprint)
                or not re.fullmatch(r"public-[a-f0-9]{64}", self.root_request_id)
                or self.reason_code not in {"pending", "direct", "reply", "general", "single_domain",
                                            "cross_domain", "approval", "council", "ambiguous", "unsupported"}
                or type(self.reassigned) is not bool):
            raise ValueError("invalid_team_request")
        if (not isinstance(self.input_ids, (tuple, list)) or not 0 < len(self.input_ids) <= 32
                or self.input_ids[0] != self.message_id or len(set(self.input_ids)) != len(self.input_ids)
                or any(not isinstance(v, str) or not re.fullmatch(POSITIVE_ID, v) for v in self.input_ids)):
            raise ValueError("invalid_team_input_ids")
        for value in (self.reply_to_message_id, self.reply_to_sender_id):
            if value is not None and (not isinstance(value, str) or not re.fullmatch(POSITIVE_ID, value)):
                raise ValueError("invalid_team_reply")


class TeamStore(PublicHandoffStore):
    request_type = TeamRequest

    @contextmanager
    def connect(self):
        with super().connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS team_members (
                profile TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                bot_id TEXT NOT NULL, username TEXT NOT NULL, seen REAL NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS team_inputs (
                chat TEXT NOT NULL, thread TEXT NOT NULL, mid TEXT NOT NULL,
                sender TEXT NOT NULL, rid TEXT NOT NULL, PRIMARY KEY(chat,thread,mid))""")
            yield db

    def input_request(self, chat, thread, mid, sender):
        with self.connect() as db:
            row = db.execute("SELECT p.* FROM team_inputs i JOIN public_handoffs p ON p.request_id=i.rid "
                             "WHERE i.chat=? AND i.thread=? AND i.mid=? AND i.sender=?",
                             (chat, thread or "", mid, sender)).fetchone()
            return self.decode(row)

    def reply_owner(self, chat_id, thread_id, message_id, sender_id):
        return (self.input_request(chat_id, thread_id, message_id, sender_id)
                or super().reply_owner(chat_id, thread_id, message_id, sender_id))

    def enqueue(self, request):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                existing = db.execute("SELECT p.* FROM team_inputs i JOIN public_handoffs p ON p.request_id=i.rid "
                    "WHERE i.chat=? AND i.thread=? AND i.mid IN (" + ",".join("?" for _ in request.input_ids) + ") LIMIT 1",
                    (request.chat_id, request.thread_id or "", *request.input_ids)).fetchone()
                if existing is not None:
                    db.execute("COMMIT")
                    return self.decode(existing)
                db.execute("INSERT INTO public_handoffs (request_id,payload,root_request_id,state,created_at,expires_at) "
                           "VALUES (?,?,?,'accepted',?,?)", (request.request_id, json.dumps(asdict(request)),
                           request.root_request_id, request.created_at, request.expires_at))
                db.executemany("INSERT INTO team_inputs VALUES (?,?,?,?,?)",
                               [(request.chat_id, request.thread_id or "", mid, request.sender_id, request.request_id)
                                for mid in request.input_ids])
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return self.get(request.request_id)

    def register(self, profile, fingerprint, bot_id, username):
        if (profile not in PROFILES or not re.fullmatch(POSITIVE_ID, bot_id)
                or not re.fullmatch(r"[A-Za-z0-9_]{1,64}", username)):
            raise ValueError("invalid_team_member")
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO team_members VALUES (?,?,?,?,?)",
                       (profile, fingerprint, bot_id, username.lower(), time.time()))

    def members(self):
        with self.connect() as db:
            return {r["profile"]: dict(r) for r in db.execute("SELECT * FROM team_members WHERE seen>?",
                                                             (time.time() - 45,))}

    def route(self, row, owner, reason, parent=None, reassigned=False):
        payload = {k: row[k] for k in TeamRequest.__dataclass_fields__}
        payload.update(target_profile=owner, reason_code=reason, reassigned=reassigned)
        root = parent["root_request_id"] if parent else row["root_request_id"]
        payload["root_request_id"] = root
        TeamRequest(**payload)
        with self.connect() as db:
            return db.execute("UPDATE public_handoffs SET payload=?,root_request_id=?,state='accepted' "
                              "WHERE request_id=? AND state='running' AND send_started_at IS NULL AND expires_at>?",
                              (json.dumps(payload), root, row["request_id"], time.time())).rowcount == 1

    def unregister(self, profile):
        with self.connect() as db:
            db.execute("DELETE FROM team_members WHERE profile=?", (profile,))

    def recent(self, chat, thread, sender):
        with self.connect() as db:
            row = db.execute("SELECT * FROM public_handoffs WHERE state='delivered' "
                             "AND json_extract(payload,'$.chat_id')=? AND json_extract(payload,'$.thread_id') IS ? "
                             "AND json_extract(payload,'$.sender_id')=? AND send_started_at>=? "
                             "ORDER BY send_started_at DESC LIMIT 1", (chat, thread, sender, time.time() - 900)).fetchone()
            return self.decode(row)

    def lineage(self, root):
        with self.connect() as db:
            row = db.execute("SELECT * FROM public_handoffs WHERE root_request_id=? "
                             "AND json_extract(payload,'$.reassigned')=1 ORDER BY created_at DESC LIMIT 1",
                             (root,)).fetchone()
            if row is None:
                row = db.execute("SELECT * FROM public_handoffs WHERE request_id=?", (root,)).fetchone()
            return self.decode(row)

    def definitive_rejection(self, rid):
        # Only an authenticated Bot API rejection proves that nothing was delivered.
        with self.connect() as db:
            db.execute("UPDATE public_handoffs SET state='failed_definitive',error='send_rejected' "
                       "WHERE request_id=? AND state='running'", (rid,))
