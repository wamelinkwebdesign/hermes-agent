"""Pure, deterministic ownership routing for Telegram bot teams."""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

_PROFILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_USERNAME_RE = re.compile(r"[A-Za-z0-9_]{5,32}\Z")
_GROUP_CHAT_RE = re.compile(r"-[0-9]+\Z")
_MESSAGE_ID_RE = re.compile(r"[1-9][0-9]*\Z")

_DEFAULT_MAX_CLAIMS = 4096
_DEFAULT_MAX_ALIASES = 8192


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


def _normalize_message_id(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value) if value > 0 else None
    if not isinstance(value, str) or not _MESSAGE_ID_RE.fullmatch(value):
        return None
    return value


def _normalize_profile(value: object) -> str | None:
    if not isinstance(value, str) or not _PROFILE_RE.fullmatch(value):
        return None
    return value


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


@dataclass(frozen=True)
class IngressClaim:
    """Result of attempting to claim one Telegram update in this process.

    ``adapter_profile`` names the first claimant for valid updates. Duplicates
    report that original profile, not the adapter that lost the claim.
    Invalid input fails closed with all status fields false/``None``.
    """

    claimed: bool
    duplicate: bool
    adapter_profile: str | None


@dataclass(frozen=True)
class RootOwnership:
    """Canonical Telegram root message and the profile that owns its replies."""

    root_message_id: str
    owner_profile: str


@dataclass
class _RootFamily:
    ownership: RootOwnership
    aliases: set[tuple[str, str]]


class TelegramTeamDispatcher:
    """Bounded, process-local Telegram ingress and reply ownership state.

    One lock protects both maps. Claims evict in insertion order. Aliases evict
    as complete root families, ordered by the family's most recent successful
    extension, so no surviving alias can ever point at an evicted root. The
    state intentionally contains only Telegram IDs, profile names, and order
    metadata; it stores no message text and has no persistence or TTL behavior.
    """

    def __init__(
        self,
        max_claims: int = _DEFAULT_MAX_CLAIMS,
        max_aliases: int = _DEFAULT_MAX_ALIASES,
    ) -> None:
        if (
            isinstance(max_claims, bool)
            or not isinstance(max_claims, int)
            or max_claims <= 0
        ):
            raise ValueError("max_claims must be a positive integer")
        if (
            isinstance(max_aliases, bool)
            or not isinstance(max_aliases, int)
            or max_aliases <= 0
        ):
            raise ValueError("max_aliases must be a positive integer")

        self._max_claims = max_claims
        self._max_aliases = max_aliases
        self._lock = threading.Lock()
        self._claims: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._aliases: dict[tuple[str, str], RootOwnership] = {}
        self._families: OrderedDict[tuple[str, str], _RootFamily] = OrderedDict()

    def claim_ingress(
        self,
        chat_id: object,
        inbound_message_id: object,
        adapter_profile: object,
    ) -> IngressClaim:
        """Claim a Telegram update once across every adapter using this dispatcher."""
        normalized_chat_id = _normalize_group_chat_id(chat_id)
        normalized_message_id = _normalize_message_id(inbound_message_id)
        normalized_profile = _normalize_profile(adapter_profile)
        if (
            normalized_chat_id is None
            or normalized_message_id is None
            or normalized_profile is None
        ):
            return IngressClaim(claimed=False, duplicate=False, adapter_profile=None)

        key = (normalized_chat_id, normalized_message_id)
        with self._lock:
            first_profile = self._claims.get(key)
            if first_profile is not None:
                return IngressClaim(
                    claimed=False,
                    duplicate=True,
                    adapter_profile=first_profile,
                )

            self._claims[key] = normalized_profile
            while len(self._claims) > self._max_claims:
                self._claims.popitem(last=False)

        return IngressClaim(
            claimed=True,
            duplicate=False,
            adapter_profile=normalized_profile,
        )

    def record_root(
        self,
        chat_id: object,
        root_message_id: object,
        owner_profile: object,
    ) -> bool:
        """Bind a root message to itself and one immutable owner profile."""
        normalized_chat_id = _normalize_group_chat_id(chat_id)
        normalized_root_id = _normalize_message_id(root_message_id)
        normalized_profile = _normalize_profile(owner_profile)
        if (
            normalized_chat_id is None
            or normalized_root_id is None
            or normalized_profile is None
        ):
            return False

        root_key = (normalized_chat_id, normalized_root_id)
        ownership = RootOwnership(normalized_root_id, normalized_profile)
        with self._lock:
            existing = self._aliases.get(root_key)
            if existing is not None:
                return existing == ownership and existing.root_message_id == normalized_root_id

            evictions = self._plan_family_evictions(added_aliases=1, keep_family=None)
            if evictions is None:
                return False
            self._evict_families(evictions)
            self._aliases[root_key] = ownership
            self._families[root_key] = _RootFamily(ownership, {root_key})
            return True

    def record_inbound_alias(
        self,
        chat_id: object,
        message_id: object,
        parent_message_id: object,
    ) -> bool:
        """Map a human reply to the already-resolved root of its parent."""
        normalized_chat_id = _normalize_group_chat_id(chat_id)
        normalized_message_id = _normalize_message_id(message_id)
        normalized_parent_id = _normalize_message_id(parent_message_id)
        if (
            normalized_chat_id is None
            or normalized_message_id is None
            or normalized_parent_id is None
        ):
            return False

        message_key = (normalized_chat_id, normalized_message_id)
        parent_key = (normalized_chat_id, normalized_parent_id)
        with self._lock:
            ownership = self._aliases.get(parent_key)
            if ownership is None:
                return False
            existing = self._aliases.get(message_key)
            if existing is not None:
                return existing == ownership

            root_key = (normalized_chat_id, ownership.root_message_id)
            return self._record_aliases_locked(root_key, ownership, [message_key])

    def record_outbound_alias(
        self,
        chat_id: object,
        message_ids: object,
        root_message_id: object,
    ) -> bool:
        """Atomically map every bot output ID to one existing root family."""
        normalized_chat_id = _normalize_group_chat_id(chat_id)
        normalized_root_id = _normalize_message_id(root_message_id)
        if normalized_chat_id is None or normalized_root_id is None:
            return False
        if (
            isinstance(message_ids, (str, bytes, bytearray, Mapping))
            or not isinstance(message_ids, Iterable)
        ):
            return False

        normalized_ids: list[str] = []
        seen_ids: set[str] = set()
        for value in message_ids:
            normalized_id = _normalize_message_id(value)
            if normalized_id is None:
                return False
            if normalized_id not in seen_ids:
                normalized_ids.append(normalized_id)
                seen_ids.add(normalized_id)
        if not normalized_ids:
            return False

        root_key = (normalized_chat_id, normalized_root_id)
        message_keys = [(normalized_chat_id, message_id) for message_id in normalized_ids]
        with self._lock:
            ownership = self._aliases.get(root_key)
            if ownership is None or ownership.root_message_id != normalized_root_id:
                return False
            return self._record_aliases_locked(root_key, ownership, message_keys)

    def resolve_reply_owner(
        self,
        chat_id: object,
        reply_to_message_id: object,
    ) -> RootOwnership | None:
        """Resolve any known human or bot reply target to its root owner."""
        normalized_chat_id = _normalize_group_chat_id(chat_id)
        normalized_message_id = _normalize_message_id(reply_to_message_id)
        if normalized_chat_id is None or normalized_message_id is None:
            return None
        with self._lock:
            return self._aliases.get((normalized_chat_id, normalized_message_id))

    def _record_aliases_locked(
        self,
        root_key: tuple[str, str],
        ownership: RootOwnership,
        message_keys: list[tuple[str, str]],
    ) -> bool:
        family = self._families.get(root_key)
        if family is None or family.ownership != ownership:
            return False

        new_keys: list[tuple[str, str]] = []
        for message_key in message_keys:
            existing = self._aliases.get(message_key)
            if existing is not None:
                if existing != ownership:
                    return False
                continue
            new_keys.append(message_key)

        if not new_keys:
            return True
        if len(family.aliases) + len(new_keys) > self._max_aliases:
            return False

        evictions = self._plan_family_evictions(
            added_aliases=len(new_keys),
            keep_family=root_key,
        )
        if evictions is None:
            return False

        self._evict_families(evictions)
        for message_key in new_keys:
            self._aliases[message_key] = ownership
            family.aliases.add(message_key)
        self._families.move_to_end(root_key)
        return True

    def _plan_family_evictions(
        self,
        *,
        added_aliases: int,
        keep_family: tuple[str, str] | None,
    ) -> tuple[tuple[str, str], ...] | None:
        excess = len(self._aliases) + added_aliases - self._max_aliases
        if excess <= 0:
            return ()

        planned: list[tuple[str, str]] = []
        reclaimed = 0
        for root_key, family in self._families.items():
            if root_key == keep_family:
                continue
            planned.append(root_key)
            reclaimed += len(family.aliases)
            if reclaimed >= excess:
                return tuple(planned)
        return None

    def _evict_families(self, root_keys: tuple[tuple[str, str], ...]) -> None:
        for root_key in root_keys:
            family = self._families.pop(root_key)
            for alias_key in family.aliases:
                self._aliases.pop(alias_key, None)


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
