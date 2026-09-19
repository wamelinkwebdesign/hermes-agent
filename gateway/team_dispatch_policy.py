"""Explicit standalone Telegram team scope and metadata-first ownership."""
from dataclasses import dataclass
import hashlib
import json
import re

SPECIALISTS = ("revenue", "product", "design", "engineering", "finance-ops")
PROFILES = ("default", *SPECIALISTS)
POSITIVE_ID = r"[1-9][0-9]{0,19}"


@dataclass(frozen=True)
class TeamPolicy:
    destinations: tuple[tuple[str, str | None], ...]

    @classmethod
    def from_extra(cls, extra):
        raw = extra.get("team_dispatch")
        if raw is None or (isinstance(raw, dict) and raw.get("enabled") is False):
            return None
        if (not isinstance(raw, dict) or raw.get("enabled") is not True
                or raw.get("coordinator_profile") != "default"
                or not isinstance(raw.get("specialist_profiles"), list)
                or sorted(raw["specialist_profiles"]) != sorted(SPECIALISTS)):
            raise ValueError("invalid_team_config")
        dests = raw.get("destinations")
        if not isinstance(dests, list) or not 0 < len(dests) <= 64:
            raise ValueError("invalid_team_destinations")
        parsed = []
        for item in dests:
            if not isinstance(item, dict):
                raise ValueError("invalid_team_destination")
            chat, thread = item.get("chat_id"), item.get("thread_id")
            if (not isinstance(chat, str) or not re.fullmatch("-" + POSITIVE_ID, chat)
                    or (thread is not None and (not isinstance(thread, str)
                                               or not re.fullmatch(POSITIVE_ID, thread)))):
                raise ValueError("invalid_team_destination")
            parsed.append((chat, thread))
        if len(set(parsed)) != len(parsed):
            raise ValueError("duplicate_team_destination")
        legacy = extra.get("public_handoff") or {}
        if legacy.get("enabled") is True and (legacy.get("chat_id"), legacy.get("thread_id")) in parsed:
            raise ValueError("team_public_handoff_overlap")
        return cls(tuple(sorted(parsed, key=lambda v: (v[0], v[1] or ""))))

    def matches(self, chat_id, thread_id):
        return (chat_id, thread_id) in self.destinations

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.destinations).encode()).hexdigest()


def addressed_owner(message, members):
    """Only authenticated bot reply authors and current addressees, never quotes."""
    replied = getattr(message, "reply_to_message", None)
    author = getattr(replied, "from_user", None)
    if getattr(author, "is_bot", False):
        for profile, member in members.items():
            if str(author.id) == member["bot_id"]:
                return profile
    text = getattr(message, "text", None) or getattr(message, "caption", None) or ""
    # A leading addressee is unambiguous even on clients without entity metadata.
    prefix = re.match(r"^\s*((?:@[A-Za-z0-9_]+[\s,:]*)+)", text)
    mentions = set(re.findall(r"@([A-Za-z0-9_]+)", prefix[1] if prefix else ""))
    command = re.match(r"^/[A-Za-z0-9_]+@([A-Za-z0-9_]+)(?:\s|$)", text)
    if command:
        mentions.add(command[1])
    encoded = text.encode("utf-16-le")
    for entity in getattr(message, "entities", None) or getattr(message, "caption_entities", None) or ():
        if str(getattr(entity, "type", "")) != "mention":
            continue
        offset, length = getattr(entity, "offset", None), getattr(entity, "length", None)
        if type(offset) is not int or type(length) is not int or offset < 0 or length <= 0:
            continue
        if 2 * (offset + length) > len(encoded):
            continue
        try:
            before = encoded[:2 * offset].decode("utf-16-le")
            value = encoded[2 * offset:2 * (offset + length)].decode("utf-16-le")
        except UnicodeDecodeError:
            continue
        # Entities in quotes/forwarded prose aren't routing authority.
        if not any(c in before for c in '\n"`>') and value.startswith("@"):
            mentions.add(value[1:])
    owners = {p for p, m in members.items() if m["username"].lower() in {v.lower() for v in mentions}}
    return next(iter(owners)) if len(owners) == 1 else "default" if owners else None
