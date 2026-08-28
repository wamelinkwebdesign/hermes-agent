"""Pure, deterministic ownership routing for Telegram bot teams."""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, Literal, cast

_PROFILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_USERNAME_RE = re.compile(r"[A-Za-z0-9_]{5,32}\Z")
_GROUP_CHAT_RE = re.compile(r"-[0-9]+\Z")
_MESSAGE_ID_RE = re.compile(r"[1-9][0-9]*\Z")

_MIN_GROUP_CHAT_ID = -(2**63)
_MAX_MESSAGE_ID = 2**63 - 1
_MAX_GROUP_CHAT_ID_LENGTH = len(str(_MIN_GROUP_CHAT_ID))
_MAX_MESSAGE_ID_LENGTH = len(str(_MAX_MESSAGE_ID))

_DEFAULT_MAX_CLAIMS = 4096
_DEFAULT_MAX_ALIASES = 8192
MAX_TELEGRAM_BATCH_MESSAGE_IDS = 64


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
        return str(value) if _MIN_GROUP_CHAT_ID <= value < 0 else None
    if (
        not isinstance(value, str)
        or len(value) > _MAX_GROUP_CHAT_ID_LENGTH
        or not _GROUP_CHAT_RE.fullmatch(value)
    ):
        return None
    parsed = int(value)
    return str(parsed) if _MIN_GROUP_CHAT_ID <= parsed < 0 else None


def _normalize_message_id(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value) if 0 < value <= _MAX_MESSAGE_ID else None
    if (
        not isinstance(value, str)
        or len(value) > _MAX_MESSAGE_ID_LENGTH
        or not _MESSAGE_ID_RE.fullmatch(value)
    ):
        return None
    return value if int(value) <= _MAX_MESSAGE_ID else None


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

    _FIELDS: ClassVar[frozenset[str]] = frozenset({
        "coordinator_profile",
        "coordinator_username",
        "members",
        "allowed_chats",
    })

    @classmethod
    def from_raw(cls, raw: object) -> TelegramTeamConfig | None:
        """Validate *raw* atomically, returning ``None`` on any malformed field."""
        if not isinstance(raw, Mapping) or set(raw) != cls._FIELDS:
            return None

        coordinator_profile = raw.get("coordinator_profile")
        if not isinstance(coordinator_profile, str) or not _PROFILE_RE.fullmatch(
            coordinator_profile
        ):
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
    """Stable adapter-to-runner input for one Telegram group message.

    Message identifiers are normalized only from the current Telegram Message
    and its immediate ``reply_to_message`` target. Invalid identifiers become
    ``None``; quoted text is never a routing input.
    """

    mentions: frozenset[str]
    reply_author_username: str | None
    chat_id: str
    owner_profile: str
    owner_username: str
    message_id: str | None = None
    reply_to_message_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "message_id", _normalize_message_id(self.message_id))
        object.__setattr__(
            self,
            "reply_to_message_id",
            _normalize_message_id(self.reply_to_message_id),
        )


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


@dataclass(frozen=True, eq=False, repr=False)
class IngressReservation:
    """Opaque immutable identity for one pending atomic ingress reservation."""

    _dispatcher_identity: object
    _keys: tuple[tuple[str, str], ...]
    _adapter_profile: str

    def __repr__(self) -> str:
        return "<IngressReservation>"


@dataclass(frozen=True)
class IngressReservationResult:
    """Result of atomically reserving one bounded Telegram ingress batch."""

    reserved: bool
    duplicate: bool
    adapter_profile: str | None
    reservation: IngressReservation | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class RootOwnership:
    """Canonical Telegram root message and the profile that owns its replies."""

    root_message_id: str
    owner_profile: str


@dataclass(frozen=True)
class ConstituentOwnershipResult:
    """Atomic ownership preflight result for one bounded ingress batch."""

    status: Literal["invalid", "none", "same", "partial", "conflict"]
    ownership: RootOwnership | None = None


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
        self._reservation_identity = object()
        self._claims: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._pending_reservations: dict[tuple[str, str], IngressReservation] = {}
        self._aliases: dict[tuple[str, str], RootOwnership] = {}
        self._families: OrderedDict[tuple[str, str], _RootFamily] = OrderedDict()

    def reserve_ingress_batch(
        self,
        chat_id: object,
        inbound_message_ids: object,
        adapter_profile: object,
    ) -> IngressReservationResult:
        """Atomically reserve a bounded batch without evicting committed claims."""
        rejected = IngressReservationResult(False, False, None)
        normalized_chat_id = _normalize_group_chat_id(chat_id)
        normalized_profile = _normalize_profile(adapter_profile)
        if normalized_chat_id is None or normalized_profile is None:
            return rejected
        if isinstance(inbound_message_ids, (str, bytes, bytearray, Mapping)):
            return rejected
        try:
            iterator = iter(cast(Iterable[object], inbound_message_ids))
        except Exception:
            return rejected

        normalized_ids: list[str] = []
        seen_ids: set[str] = set()
        for attempt in range(MAX_TELEGRAM_BATCH_MESSAGE_IDS + 1):
            try:
                value = next(iterator)
            except StopIteration:
                break
            except Exception:
                return rejected
            if attempt == MAX_TELEGRAM_BATCH_MESSAGE_IDS:
                return rejected
            try:
                normalized_id = _normalize_message_id(value)
            except Exception:
                return rejected
            if normalized_id is None:
                return rejected
            if normalized_id not in seen_ids:
                normalized_ids.append(normalized_id)
                seen_ids.add(normalized_id)
        if not normalized_ids or len(normalized_ids) > self._max_claims:
            return rejected

        keys = tuple((normalized_chat_id, message_id) for message_id in normalized_ids)
        with self._lock:
            for key in keys:
                bound_ownership = self._aliases.get(key)
                if bound_ownership is not None:
                    root_key = (normalized_chat_id, bound_ownership.root_message_id)
                    family = self._families.get(root_key)
                    if (
                        self._aliases.get(root_key) != bound_ownership
                        or family is None
                        or family.ownership != bound_ownership
                        or key not in family.aliases
                    ):
                        return rejected
                    return IngressReservationResult(
                        False,
                        True,
                        bound_ownership.owner_profile,
                    )
                committed_profile = self._claims.get(key)
                if committed_profile is not None:
                    return IngressReservationResult(
                        False,
                        True,
                        committed_profile,
                    )
                pending = self._pending_reservations.get(key)
                if pending is not None:
                    return IngressReservationResult(
                        False,
                        True,
                        pending._adapter_profile,
                    )
            if len(self._pending_reservations) + len(keys) > self._max_claims:
                return rejected

            reservation = IngressReservation(
                self._reservation_identity,
                keys,
                normalized_profile,
            )
            for key in keys:
                self._pending_reservations[key] = reservation
            return IngressReservationResult(
                True,
                False,
                normalized_profile,
                reservation,
            )

    def commit_ingress(self, reservation: object) -> bool:
        """Commit only the exact pending reservation returned by this dispatcher."""
        if (
            not isinstance(reservation, IngressReservation)
            or reservation._dispatcher_identity is not self._reservation_identity
        ):
            return False
        with self._lock:
            if not reservation._keys or any(
                self._pending_reservations.get(key) is not reservation
                for key in reservation._keys
            ):
                return False
            for key in reservation._keys:
                self._pending_reservations.pop(key, None)
                self._claims[key] = reservation._adapter_profile
            while len(self._claims) > self._max_claims:
                self._claims.popitem(last=False)
            return True

    def release_ingress(self, reservation: object) -> bool:
        """Release only the exact pending reservation returned by this dispatcher."""
        if (
            not isinstance(reservation, IngressReservation)
            or reservation._dispatcher_identity is not self._reservation_identity
        ):
            return False
        with self._lock:
            if not reservation._keys or any(
                self._pending_reservations.get(key) is not reservation
                for key in reservation._keys
            ):
                return False
            for key in reservation._keys:
                self._pending_reservations.pop(key, None)
            return True

    def claim_ingress(
        self,
        chat_id: object,
        inbound_message_id: object,
        adapter_profile: object,
    ) -> IngressClaim:
        """Claim a Telegram update once across every adapter using this dispatcher."""
        attempt = self.reserve_ingress_batch(
            chat_id,
            [inbound_message_id],
            adapter_profile,
        )
        if attempt.reserved and attempt.reservation is not None:
            if self.commit_ingress(attempt.reservation):
                return IngressClaim(
                    claimed=True,
                    duplicate=False,
                    adapter_profile=attempt.adapter_profile,
                )
            self.release_ingress(attempt.reservation)
            return IngressClaim(False, False, None)
        return IngressClaim(
            claimed=False,
            duplicate=attempt.duplicate,
            adapter_profile=attempt.adapter_profile,
        )

    def inspect_constituent_ownership(
        self,
        chat_id: object,
        message_ids: object,
    ) -> ConstituentOwnershipResult:
        """Atomically inspect every bounded constituent without mutating state."""
        invalid = ConstituentOwnershipResult("invalid")
        normalized_chat_id = _normalize_group_chat_id(chat_id)
        if normalized_chat_id is None or isinstance(
            message_ids, (str, bytes, bytearray, Mapping)
        ):
            return invalid
        try:
            iterator = iter(cast(Iterable[object], message_ids))
        except Exception:
            return invalid

        normalized_ids: list[str] = []
        seen_ids: set[str] = set()
        for attempt in range(MAX_TELEGRAM_BATCH_MESSAGE_IDS + 1):
            try:
                value = next(iterator)
            except StopIteration:
                break
            except Exception:
                return invalid
            if attempt == MAX_TELEGRAM_BATCH_MESSAGE_IDS:
                return invalid
            try:
                normalized_id = _normalize_message_id(value)
            except Exception:
                return invalid
            if normalized_id is None:
                return invalid
            if normalized_id not in seen_ids:
                normalized_ids.append(normalized_id)
                seen_ids.add(normalized_id)
        if not normalized_ids:
            return invalid

        keys = tuple((normalized_chat_id, message_id) for message_id in normalized_ids)
        with self._lock:
            ownerships: list[RootOwnership] = []
            for key in keys:
                ownership = self._aliases.get(key)
                if ownership is None:
                    continue
                root_key = (normalized_chat_id, ownership.root_message_id)
                family = self._families.get(root_key)
                if (
                    self._aliases.get(root_key) != ownership
                    or family is None
                    or family.ownership != ownership
                    or key not in family.aliases
                ):
                    return invalid
                ownerships.append(ownership)

            if not ownerships:
                return ConstituentOwnershipResult("none")
            first = ownerships[0]
            if any(ownership != first for ownership in ownerships[1:]):
                return ConstituentOwnershipResult("conflict")
            status: Literal["same", "partial"] = (
                "same" if len(ownerships) == len(keys) else "partial"
            )
            return ConstituentOwnershipResult(status, first)

    def verify_routed_ownership(
        self,
        chat_id: object,
        root_message_id: object,
        owner_profile: object,
        message_ids: object,
    ) -> bool:
        """Verify a routed event's complete constituent binding from one snapshot."""
        normalized_root_id = _normalize_message_id(root_message_id)
        normalized_profile = _normalize_profile(owner_profile)
        if normalized_root_id is None or normalized_profile is None:
            return False
        result = self.inspect_constituent_ownership(chat_id, message_ids)
        return result.status == "same" and result.ownership == RootOwnership(
            normalized_root_id,
            normalized_profile,
        )

    def record_root(
        self,
        chat_id: object,
        root_message_id: object,
        owner_profile: object,
    ) -> bool:
        """Bind a root message to itself and one immutable owner profile."""
        return self.record_root_batch(
            chat_id,
            root_message_id,
            owner_profile,
            [root_message_id],
        )

    def _normalize_alias_batch(self, message_ids: object) -> tuple[str, ...] | None:
        """Materialize one hostile iterable within a total-attempt bound."""
        if isinstance(message_ids, (str, bytes, bytearray, Mapping)):
            return None
        try:
            iterator = iter(cast(Iterable[object], message_ids))
        except Exception:
            return None

        normalized_ids: list[str] = []
        seen_ids: set[str] = set()
        for attempt in range(self._max_aliases + 1):
            try:
                value = next(iterator)
            except StopIteration:
                break
            except Exception:
                return None
            if attempt == self._max_aliases:
                return None
            try:
                normalized_id = _normalize_message_id(value)
            except Exception:
                return None
            if normalized_id is None:
                return None
            if normalized_id not in seen_ids:
                normalized_ids.append(normalized_id)
                seen_ids.add(normalized_id)
        return tuple(normalized_ids) if normalized_ids else None

    def record_root_batch(
        self,
        chat_id: object,
        root_message_id: object,
        owner_profile: object,
        alias_ids: object,
    ) -> bool:
        """Atomically bind one root and every constituent alias in its family."""
        normalized_chat_id = _normalize_group_chat_id(chat_id)
        normalized_root_id = _normalize_message_id(root_message_id)
        normalized_profile = _normalize_profile(owner_profile)
        normalized_alias_ids = self._normalize_alias_batch(alias_ids)
        if (
            normalized_chat_id is None
            or normalized_root_id is None
            or normalized_profile is None
            or normalized_alias_ids is None
        ):
            return False

        if normalized_root_id not in normalized_alias_ids:
            normalized_alias_ids = (normalized_root_id, *normalized_alias_ids)
        if len(normalized_alias_ids) > self._max_aliases:
            return False

        root_key = (normalized_chat_id, normalized_root_id)
        ownership = RootOwnership(normalized_root_id, normalized_profile)
        message_keys = [
            (normalized_chat_id, message_id) for message_id in normalized_alias_ids
        ]
        with self._lock:
            existing = self._aliases.get(root_key)
            if existing is not None:
                family = self._families.get(root_key)
                if (
                    existing != ownership
                    or existing.root_message_id != normalized_root_id
                    or family is None
                    or family.ownership != ownership
                ):
                    return False
                return self._record_aliases_locked(root_key, ownership, message_keys)

            if any(message_key in self._aliases for message_key in message_keys):
                return False

            evictions = self._plan_family_evictions(
                added_aliases=len(message_keys),
                keep_family=None,
            )
            if evictions is None:
                return False
            self._evict_families(evictions)
            for message_key in message_keys:
                self._aliases[message_key] = ownership
            self._families[root_key] = _RootFamily(ownership, set(message_keys))
            return True

    def record_inbound_alias(
        self,
        chat_id: object,
        message_id: object,
        parent_message_id: object,
    ) -> bool:
        """Map a human reply to the already-resolved root of its parent."""
        return self.record_inbound_alias_batch(
            chat_id,
            [message_id],
            parent_message_id,
        )

    def record_inbound_alias_batch(
        self,
        chat_id: object,
        message_ids: object,
        parent_message_id: object,
    ) -> bool:
        """Atomically map all human constituents to one resolved parent root."""
        normalized_chat_id = _normalize_group_chat_id(chat_id)
        normalized_parent_id = _normalize_message_id(parent_message_id)
        normalized_message_ids = self._normalize_alias_batch(message_ids)
        if (
            normalized_chat_id is None
            or normalized_parent_id is None
            or normalized_message_ids is None
        ):
            return False

        parent_key = (normalized_chat_id, normalized_parent_id)
        message_keys = [
            (normalized_chat_id, message_id) for message_id in normalized_message_ids
        ]
        with self._lock:
            ownership = self._aliases.get(parent_key)
            if ownership is None:
                return False
            root_key = (normalized_chat_id, ownership.root_message_id)
            if (
                self._aliases.get(root_key) != ownership
                or self._families.get(root_key) is None
            ):
                return False
            return self._record_aliases_locked(root_key, ownership, message_keys)

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

        root_key = (normalized_chat_id, normalized_root_id)
        with self._lock:
            ownership = self._aliases.get(root_key)
            family = self._families.get(root_key)
            if (
                ownership is None
                or ownership.root_message_id != normalized_root_id
                or family is None
                or family.ownership != ownership
            ):
                return False

        if isinstance(message_ids, (str, bytes, bytearray, Mapping)):
            return False

        try:
            iterator = iter(cast(Iterable[object], message_ids))
        except Exception:
            return False

        normalized_ids: list[str] = []
        seen_ids: set[str] = set()
        for attempt in range(self._max_aliases + 1):
            try:
                value = next(iterator)
            except StopIteration:
                break
            except Exception:
                return False
            if attempt == self._max_aliases:
                return False
            try:
                normalized_id = _normalize_message_id(value)
            except Exception:
                return False
            if normalized_id is None:
                return False
            if normalized_id not in seen_ids:
                normalized_ids.append(normalized_id)
                seen_ids.add(normalized_id)
        if not normalized_ids:
            return False

        message_keys = [
            (normalized_chat_id, message_id) for message_id in normalized_ids
        ]
        with self._lock:
            if (
                self._aliases.get(root_key) != ownership
                or self._families.get(root_key) is not family
            ):
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

    profiles_by_username = {
        username: profile for profile, username in config.members.items()
    }

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
