"""Default-off Ace ingress, consumed by each existing standalone gateway.

The shared ledger contains public envelopes and receipts only. Each selected
profile runs its own normal agent and publishes through its own connected bot.
"""
import asyncio
import json
import logging
import re
import time

from gateway.config import Platform
from gateway.public_handoff_store import MAX_REQUEST_TTL_SECONDS, request_identity
from gateway.run_public_handoff import PublicHandoff
from gateway.session import SessionSource
from gateway.team_dispatch_policy import PROFILES, TeamPolicy, addressed_owner
from gateway.team_dispatch_store import TeamRequest, TeamStore
from hermes_constants import get_default_hermes_root, get_hermes_home, set_hermes_home_override, reset_hermes_home_override

logger = logging.getLogger(__name__)
FALLBACK = "Ace: deze aanvraag is niet voltooid. Er is geen antwoord van de specialist verstuurd. Probeer het opnieuw of vraag mij om hulp."
UNSUPPORTED = "Ace: automatische teamroutering ondersteunt dit type invoer nog niet. Stuur je vraag als tekst, of spreek de specialist rechtstreeks aan voor media of bediening."

# One clock bounds generation and delivery: the owner agent's run budget is
# expires_at minus a send margin, so the deadline must stay under the wire cap.
TEAM_DEADLINE_SECONDS = 300
assert TEAM_DEADLINE_SECONDS <= MAX_REQUEST_TTL_SECONDS, "team deadline exceeds request TTL cap"


def install_team_dispatch(runner, adapter):
    if adapter.platform != Platform.TELEGRAM:
        return
    policy = TeamPolicy.from_extra(adapter.config.extra)
    if runner.config.multiplex_profiles:
        if policy:
            raise ValueError("team_dispatch_requires_standalone_gateways")
        return
    home, root = get_hermes_home(), get_default_hermes_root()
    profile = next((p for p in PROFILES if home == (root if p == "default" else root / "profiles" / p)), None)
    if profile is None:
        if policy:
            raise ValueError("invalid_team_gateway_profile")
        return
    dispatch = TeamDispatch(runner, adapter, home, profile, TeamStore(root / "team_dispatch.sqlite"))
    runner._team_dispatch = dispatch
    adapter._team_dispatch_handler = dispatch.ingress
    adapter._team_batch_handler = dispatch.ingress_batch


class TeamDispatch(PublicHandoff):
    @property
    def policy(self):
        return TeamPolicy.from_extra(self.adapter.config.extra)

    def configured_policies(self):
        # Settings only, never sibling secrets or transcripts. This is a native
        # suppression fence, not authority to enable another profile's agent.
        import yaml
        root = self.home if self.profile == "default" else self.home.parent.parent
        policies = {}
        for profile in PROFILES:
            home = root if profile == "default" else root / "profiles" / profile
            path = home / "config.yaml"
            if not path.is_file():
                continue
            config = yaml.safe_load(path.read_text()) or {}
            extra = config.get("platforms", {}).get("telegram", {})
            policy = TeamPolicy.from_extra(extra)
            if policy and extra.get("enabled", False):
                policies[profile] = policy
        return policies

    def effective_policy(self):
        policies = self.configured_policies()
        if self.policy:
            policies[self.profile] = self.policy
        destinations = {d for policy in policies.values() for d in policy.destinations}
        return TeamPolicy(tuple(sorted(destinations, key=lambda d: (d[0], d[1] or "")))) if destinations else None

    def ready(self):
        return (self.adapter.config.enabled and self.adapter._bot is not None
                and self.adapter.is_connected
                and self.runner.adapters.get(Platform.TELEGRAM) is self.adapter)

    def register(self):
        policy = self.policy
        bot = self.adapter._bot
        if policy and self.ready():
            self.store.register(self.profile, policy.fingerprint, str(bot.id), bot.username)
        elif self.store.path.exists():
            self.store.unregister(self.profile)

    def members(self):
        members = self.store.members()
        configured = self.configured_policies()
        if (not self.policy or set(configured) != set(PROFILES)
                or any(p.fingerprint != self.policy.fingerprint for p in configured.values())
                or set(members) != set(PROFILES)
                or any(m["fingerprint"] != self.policy.fingerprint for m in members.values())
                or len({m["bot_id"] for m in members.values()}) != len(PROFILES)
                or len({m["username"] for m in members.values()}) != len(PROFILES)):
            raise ValueError("team_members_unavailable_or_mismatched")
        return members

    def authorized(self, message):
        policy = self.effective_policy()
        if (not policy or not self.ready()
                or not self.adapter._is_group_chat(message)):
            return False
        chat, thread = str(message.chat.id), self.adapter._effective_message_thread_id(message)
        source = self.adapter._source_from_message_for_auth(message)
        allowed = self.adapter._telegram_allowed_chats()
        return (policy.matches(chat, thread) and not source.is_bot and bool(source.user_id)
                and self.adapter._is_user_authorized_from_message(message)
                and self.runner._is_user_authorized_for_source(source, allow_adapter_delegation=False)
                and self.adapter._topic_gates_pass(thread, warn_non_numeric=False) is not False
                and (not allowed or chat in allowed))

    async def ingress(self, message):
        policy = self.effective_policy()
        chat, thread = str(message.chat.id), self.adapter._effective_message_thread_id(message)
        if not policy or not self.adapter._is_group_chat(message) or not policy.matches(chat, thread):
            return False
        self.register()
        # Once a destination is opted in, failed authorization never escapes to guest/native ingress.
        if not self.authorized(message):
            return True
        if self.store.input_request(chat, thread, str(message.message_id), str(message.from_user.id)):
            return True
        try:
            members = self.members()
        except ValueError:
            members = self.store.members()
            coherent = False
        else:
            coherent = True
        replied = getattr(message, "reply_to_message", None)
        text = getattr(message, "text", None) or ""
        if replied and text.strip().lower() == "/stop":
            from gateway.team_dispatch_batch import pending_reply
            pending = pending_reply(self.adapter, chat, thread, replied)
            if pending is not None:
                # Freeze pending aliases before cancellation. A later debounce
                # flush is deduplicated against this terminal row, not revived.
                self.store.enqueue(pending)
        parent = self.store.reply_owner(chat, thread, str(replied.message_id),
                    str(getattr(replied.from_user, "id", ""))) if replied else None
        if parent:
            parent = self.store.lineage(parent["root_request_id"])
        addressed = addressed_owner(message, members)
        owner = parent["target_profile"] if parent else addressed or "default"
        if self.profile != owner:
            return True
        # Direct media/commands keep their existing capable specialist path.
        if addressed and not parent and (not text or text.startswith("/")) and coherent:
            return False
        if parent and self.store.conversation_uncertain(parent["root_request_id"]):
            return True
        if text.strip().lower() == "/stop" and parent:
            self.store.cancel(parent["root_request_id"], str(message.from_user.id))
            return True
        now = time.time()
        rid = request_identity(chat, thread, str(message.message_id))
        reason = "reply" if parent else "direct" if addressed else "pending"
        reassigned = parent["reassigned"] if parent else False
        if parent and re.match(r"(?i)\s*(verkeerde agent|wrong agent)\b", text):
            owner = "default"
            reason = "ambiguous" if reassigned else "pending"
            reassigned = True
        if not text or len(text) > 6000 or text.startswith("/"):
            text, reason = "Unsupported public input", "unsupported"
        if not coherent:
            reason = "unsupported"
        # Construction failure = nothing was ever enqueued, so falling through to
        # normal Ace handling cannot double-deliver; silence is the worse fault.
        try:
            request = TeamRequest(2, rid, self.profile, owner, "telegram", chat, thread,
                str(message.message_id), str(message.from_user.id), text, now,
                now + TEAM_DEADLINE_SECONDS,
                parent["root_request_id"] if parent else rid,
                str(replied.message_id) if parent else None, str(replied.from_user.id) if parent else None,
                (self.policy or policy).fingerprint, reason, reassigned)
        except ValueError as exc:
            logger.error("team request construction rejected: %s", exc, exc_info=True)
            return False
        from gateway.team_dispatch_batch import queue_request
        queue_request(self.adapter, request, message)
        return True

    async def ingress_batch(self, event):
        from dataclasses import replace
        request = event._team_request
        policy = self.policy or self.effective_policy()
        if not policy or not self.authorized(event.raw_message) or request.fingerprint != policy.fingerprint:
            return
        brief = event.text
        reason = request.reason_code
        if not 0 < len(brief) <= 6000:
            brief, reason = "Unsupported public input", "unsupported"
        self.store.enqueue(replace(request, brief=brief, reason_code=reason, input_ids=tuple(event._team_input_ids)))

    def validate(self, row):
        if row["fingerprint"] != self.policy.fingerprint or not self.authorized(self.message_for(row)):
            raise ValueError("team_admission_revoked")
        members = self.members()
        if str(self.adapter._bot.id) != members[self.profile]["bot_id"]:
            raise ValueError("team_identity_changed")
        if self.store.conversation_uncertain(row["root_request_id"]):
            raise ValueError("team_conversation_quarantined")
        return members

    async def tick(self):
        if not self.effective_policy():
            if self.store.path.exists():
                self.store.unregister(self.profile)
            return
        self.register()
        # Wiring precedes connect/publication. Preserve queued work until ready.
        if not self.ready():
            return
        if not self.store.path.exists():
            return
        self.store.expire()
        if self.profile == "default":
            for row in self.store.fallback_pending():
                if len(self.tasks) < 4:
                    self.start_task(self.fallback(row))
        if getattr(self.runner, "_draining", False) or getattr(self.runner, "_external_drain_active", False):
            return
        for row in self.store.pending():
            if not self.policy and row["reason_code"] != "unsupported":
                continue
            if row["target_profile"] != self.profile or len(self.tasks) >= 4:
                continue
            if row["root_request_id"] != row["request_id"]:
                root = self.store.get(row["root_request_id"])
                if root and root["state"] in {"accepted", "running"}:
                    continue
            if self.store.claim(row["request_id"], expected_owner=row["target_profile"],
                                expected_reason=row["reason_code"]):
                self.start_task(self.receive(row))

    async def close(self):
        if self.store.path.exists():
            self.store.unregister(self.profile)
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def select(self, row):
        from gateway.team_dispatch_agent import classify, related
        parent = self.store.recent(row["chat_id"], row["thread_id"], row["sender_id"])
        if (not row["reassigned"] and parent and parent["target_profile"] != "default"
                and await related(self.runner, row["brief"], parent)):
            self.store.route(row, parent["target_profile"], "reply", parent, parent["reassigned"])
            return
        verdict = await classify(self.runner, row["brief"])
        owner = verdict["specialist"] if verdict["decision"] == "specialist" else "default"
        self.store.route(row, owner, verdict["reason_code"], reassigned=row["reassigned"])
        logger.info("team_dispatch_v1 classified request=%s owner=%s reason=%s", row["request_id"], owner, verdict["reason_code"])

    async def receive(self, row):
        rid = row["request_id"]
        token = set_hermes_home_override(self.home)
        try:
            self.validate(row)
            if row["reason_code"] == "reply":
                lineage = self.store.lineage(row["root_request_id"])
                if lineage and lineage["target_profile"] != self.profile:
                    self.store.route(row, lineage["target_profile"], "reply", lineage, lineage["reassigned"])
                    return
            if row["reason_code"] == "pending":
                if self.profile != "default":
                    raise ValueError("classifier_owner")
                await self.select(row)
                return
            if row["reason_code"] == "unsupported":
                raise ValueError("unsupported")
            source = SessionSource(platform=Platform.TELEGRAM, chat_id=row["chat_id"], chat_type="group",
                                   user_id=row["sender_id"], thread_id=row["thread_id"], message_id=row["message_id"])
            sid = "team-" + self.profile + "-" + row["root_request_id"]
            key = "telegram-team:" + row["chat_id"] + ":" + (row["thread_id"] or "main") + ":" + row["root_request_id"]
            db = self.runner.session_store._db
            db.create_session(sid, "telegram", session_key=key, profile_name=self.profile,
                              chat_id=source.chat_id, chat_type="group", thread_id=source.thread_id,
                              origin_json=json.dumps(source.to_dict()))
            if not self.store.bind_session(rid, sid, key, sid):
                raise ValueError("stale_session")
            from gateway.team_dispatch_agent import generate
            result = await generate(self, row, source, sid, key)
            answer = result.get("final_response")
            if (result.get("failed") or result.get("interrupted") or result.get("completed") is not True
                    or not isinstance(answer, str) or not answer.strip()
                    or answer.strip() in {"NO_REPLY", "[SILENT]"}
                    or len(answer.encode("utf-16-le")) > 8000):
                raise ValueError("invalid_team_result")
            self.validate(row)
            content, parse_mode = self.adapter.prepare_public_handoff(answer)
            if not self.store.begin_send(rid, answer):
                raise ValueError("stale_result")
            from telegram.error import BadRequest, Forbidden
            try:
                async with asyncio.timeout(15):
                    receipt = await self.adapter.send_public_handoff(row["chat_id"], row["thread_id"],
                                        row["message_id"], content, str(self.adapter._bot.id), parse_mode=parse_mode)
            except (BadRequest, Forbidden):
                self.store.definitive_rejection(rid)
                return
            receipt.update(target_profile=self.profile, session_id=sid, session_key=key)
            if not self.store.delivered(rid, receipt):
                self.store.fail(rid, "late_delivery")
            logger.info("team_dispatch_v1 result request=%s owner=%s delivered=%s", rid, self.profile, self.store.get(rid)["state"])
        except asyncio.CancelledError:
            self.store.fail(rid, "cancelled")
            raise
        except Exception as exc:
            # Do not put model errors, credentials or private paths in envelopes/fallbacks.
            logger.warning("team_dispatch_v1 failed request=%s phase=%s", rid, type(exc).__name__)
            self.store.fail(rid, "generation_or_admission_failed")
        finally:
            reset_hermes_home_override(token)

    async def fallback(self, row):
        if (not self.authorized(self.message_for(row))
                or self.store.conversation_uncertain(row["root_request_id"])
                or not self.store.claim_fallback(row["request_id"])):
            return
        try:
            async with asyncio.timeout(15):
                receipt = await self.adapter.send_public_handoff(row["chat_id"], row["thread_id"], row["message_id"],
                    UNSUPPORTED if row["reason_code"] == "unsupported" else FALLBACK, str(self.adapter._bot.id))
            receipt.update(state="delivered", target_profile="default")
        except (Exception, asyncio.CancelledError):
            receipt = {"state": "delivery_uncertain"}
        self.store.finish_fallback(row["request_id"], receipt)
