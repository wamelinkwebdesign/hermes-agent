"""Pure, deterministic ownership routing for Telegram bot teams."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

_PROFILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_USERNAME_RE = re.compile(r"[A-Za-z0-9_]{5,32}\Z")
_GROUP_CHAT_RE = re.compile(r"-[0-9]+\Z")


def _normalize_username(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    username = value[1:] if value.startswith("@") else value
    if not _USERNAME_RE.fullmatch(username):
        return None
    return username.lower()


def _normalize_group_chat_id(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value) if value < 0 else None
    if not isinstance(value, str) or not _GROUP_CHAT_RE.fullmatch(value):
        return None
    parsed = int(value)
    return str(parsed) if parsed < 0 else None


@dataclass(frozen=True)
class TelegramTeamConfig:
    """Validated immutable Telegram team-routing configuration."""

    coordinator_profile: str
    coordinator_username: str
    members: Mapping[str, str]
    allowed_chats: frozenset[str]

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"coordinator_profile", "coordinator_username", "members", "allowed_chats"}
    )

    @classmethod
    def from_raw(cls, raw: object) -> TelegramTeamConfig | None:
        """Validate *raw* atomically, returning ``None`` on any malformed field."""
        if not isinstance(raw, Mapping) or set(raw) != cls._FIELDS:
            return None

        coordinator_profile = raw.get("coordinator_profile")
        if not isinstance(coordinator_profile, str) or not _PROFILE_RE.fullmatch(coordinator_profile):
            return None

        coordinator_username = _normalize_username(raw.get("coordinator_username"))
        if coordinator_username is None:
            return None

        raw_members = raw.get("members")
        if not isinstance(raw_members, Mapping):
            return None

        members: dict[str, str] = {}
        seen_usernames: set[str] = set()
        for profile, raw_username in raw_members.items():
            if not isinstance(profile, str) or not _PROFILE_RE.fullmatch(profile):
                return None
            username = _normalize_username(raw_username)
            if username is None or username in seen_usernames:
                return None
            members[profile] = username
            seen_usernames.add(username)

        if members.get(coordinator_profile) != coordinator_username:
            return None

        raw_allowed_chats = raw.get("allowed_chats")
        if not isinstance(raw_allowed_chats, list):
            return None
        allowed_chats: set[str] = set()
        for raw_chat_id in raw_allowed_chats:
            chat_id = _normalize_group_chat_id(raw_chat_id)
            if chat_id is None:
                return None
            allowed_chats.add(chat_id)

        return cls(
            coordinator_profile=coordinator_profile,
            coordinator_username=coordinator_username,
            members=MappingProxyType(members),
            allowed_chats=frozenset(allowed_chats),
        )


@dataclass(frozen=True)
class TelegramTeamRouteContext:
    """Stable adapter-to-runner input for one Telegram group message."""

    mentions: frozenset[str]
    reply_author_username: str | None
    chat_id: str
    owner_profile: str
    owner_username: str


@dataclass(frozen=True)
class TeamRouteDecision:
    """Deterministic routing result for one Telegram group message."""

    owner_profile: str | None
    reason: str
    accepted: bool


def resolve_addressed_owner(
    *,
    mentions: set[str],
    reply_author_username: str | None,
    config: TelegramTeamConfig,
    chat_id: str,
) -> TeamRouteDecision:
    """Resolve one public owner using reply, mention, then ingress precedence."""
    normalized_chat_id = _normalize_group_chat_id(chat_id)
    if normalized_chat_id not in config.allowed_chats:
        return TeamRouteDecision(None, "chat_not_allowed", False)

    profiles_by_username = {username: profile for profile, username in config.members.items()}

    reply_username = _normalize_username(reply_author_username)
    if reply_username in profiles_by_username:
        return TeamRouteDecision(
            profiles_by_username[reply_username],
            "reply_to_team_bot",
            True,
        )

    mentioned_profiles = {
        profiles_by_username[username]
        for value in mentions
        if (username := _normalize_username(value)) in profiles_by_username
    }
    if len(mentioned_profiles) == 1:
        return TeamRouteDecision(
            next(iter(mentioned_profiles)),
            "single_team_mention",
            True,
        )
    if len(mentioned_profiles) > 1:
        return TeamRouteDecision(
            config.coordinator_profile,
            "multiple_team_mentions",
            True,
        )
    return TeamRouteDecision(
        config.coordinator_profile,
        "unaddressed_ingress",
        True,
    )
