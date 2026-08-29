"""Read-only, root-scoped context for Telegram team classification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any, cast

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.telegram_team_routing import (
    _normalize_group_chat_id,
    _normalize_message_id,
    _normalize_profile,
)

MAX_TEAM_CONTEXT_MESSAGES = 12
MAX_TEAM_CONTEXT_CHARS = 12_000
_CONTEXT_HEADER = "[UNTRUSTED TELEGRAM TEAM CONTEXT]"
_ALLOWED_ROLES = frozenset({"user", "assistant"})
_STRICT_SECRET_NAME = (
    r"(?:password|passwd|passphrase|secret|token|access[_-]?token|"
    r"refresh[_-]?token|api[_-]?key|apikey|private[_-]?key|"
    r"client[_-]?secret|credential|auth|pw)"
)
_STRICT_SECRET_KEY = (
    rf"(?:{_STRICT_SECRET_NAME}|"
    rf"[A-Za-z0-9][A-Za-z0-9_.-]{{0,63}}{_STRICT_SECRET_NAME})"
)
_STRICT_ASSIGNMENT_START_RE = re.compile(
    rf"(?<![A-Za-z0-9_.-])(?P<key_quote>[\"']?)"
    rf"(?P<key>{_STRICT_SECRET_KEY})(?P=key_quote)[ \t]*[:=][ \t]*",
    re.IGNORECASE,
)
_AMBIGUOUS_OPEN_SECRET_KEY_RE = re.compile(
    rf"(?<![A-Za-z0-9_.-])[\"']{_STRICT_SECRET_KEY}[ \t]*[:=]",
    re.IGNORECASE,
)
_AMBIGUOUS_CLOSE_SECRET_KEY_RE = re.compile(
    rf"(?<![A-Za-z0-9_.\-\"']){_STRICT_SECRET_KEY}[\"'][ \t]*[:=]",
    re.IGNORECASE,
)
_STRICT_AUTHORIZATION_START_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:Proxy-)?Authorization[ \t]*[:=][ \t]*",
    re.IGNORECASE,
)
_STRICT_BEARER_START_RE = re.compile(
    r"(?<![A-Za-z0-9_-])Bearer[ \t]+",
    re.IGNORECASE,
)
_CANONICAL_REDACTION_SENTINEL_RE = re.compile(
    r"«redacted(?:-secret|:[^»]*)?»",
    re.IGNORECASE,
)
_CANONICAL_MASK_ARTIFACT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"eyJ[A-Za-z0-9_-]{3}\.\.\.[A-Za-z0-9_=-]{4}|"
    r"(?:bot)?[0-9]{8,}:\*{3}"
    r")(?![A-Za-z0-9_-])"
)
_STRICT_REDACTION = "[REDACTED]"
_MAX_STRICT_PARSE_CHARS = MAX_TEAM_CONTEXT_CHARS


@dataclass
class TeamContextSafety:
    """Mutable per-call safety signal for the classifier routing boundary."""

    unsafe: bool = False
    redaction_failed: bool = False


class _TeamContextRedactionError(Exception):
    pass


def _bounded_field_end(text: str, start: int) -> int:
    """Return the end of one physical field and folded continuations."""
    field_end = text.find("\n", start)
    if field_end < 0:
        return len(text)
    while field_end < len(text):
        continuation_start = field_end + 1
        if continuation_start >= len(text) or text[continuation_start] not in " \t":
            break
        next_end = text.find("\n", continuation_start)
        field_end = len(text) if next_end < 0 else next_end
    return field_end


def _redact_bounded_fields(text: str, start_pattern: re.Pattern[str]) -> str:
    """Mask complete physical fields, including folded continuation lines."""
    rendered: list[str] = []
    cursor = 0
    while match := start_pattern.search(text, cursor):
        rendered.append(text[cursor : match.end()])
        rendered.append(_STRICT_REDACTION)
        cursor = _bounded_field_end(text, match.end())
    rendered.append(text[cursor:])
    return "".join(rendered)


def _redact_authorization_fields(text: str) -> str:
    """Mask Authorization values for every authentication scheme."""
    return _redact_bounded_fields(text, _STRICT_AUTHORIZATION_START_RE)


def _quoted_value_end(text: str, start: int, quote: str) -> int:
    """Return the end of one escaped, possibly multiline quoted value."""
    escaped = False
    for index in range(start + 1, len(text)):
        character = text[index]
        if character == "\x00" or (ord(character) < 0x20 and character not in "\t\r\n"):
            raise _TeamContextRedactionError
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == quote:
            end = index + 1
            if end < len(text) and (text[end].isalnum() or text[end] in "_.-\\\"'"):
                raise _TeamContextRedactionError
            return end
    raise _TeamContextRedactionError


def _strict_redact_credential_assignments(text: str) -> str:
    """Parse and completely mask bounded sensitive assignments and headers."""
    if len(text) > _MAX_STRICT_PARSE_CHARS:
        raise _TeamContextRedactionError
    text.encode("utf-8", errors="strict")
    redacted = _redact_authorization_fields(text)
    redacted = _redact_bounded_fields(redacted, _STRICT_BEARER_START_RE)
    rendered: list[str] = []
    cursor = 0
    while match := _STRICT_ASSIGNMENT_START_RE.search(redacted, cursor):
        if (
            not match.group("key_quote")
            and match.start() > 0
            and redacted[match.start() - 1] in {'"', "'"}
        ):
            raise _TeamContextRedactionError
        rendered.append(redacted[cursor : match.end()])
        value_start = match.end()
        if value_start >= len(redacted):
            raise _TeamContextRedactionError
        opening = redacted[value_start]
        if opening in {'"', "'"}:
            value_end = _quoted_value_end(redacted, value_start, opening)
        else:
            value_end = value_start
            while (
                value_end < len(redacted)
                and not redacted[value_end].isspace()
                and redacted[value_end] not in "&#;,?"
            ):
                if redacted[value_end] == "\x00":
                    raise _TeamContextRedactionError
                value_end += 1
            if value_end == value_start:
                raise _TeamContextRedactionError
        rendered.append(_STRICT_REDACTION)
        cursor = _bounded_field_end(redacted, value_end)
    rendered.append(redacted[cursor:])
    joined = "".join(rendered)
    if _AMBIGUOUS_OPEN_SECRET_KEY_RE.search(
        joined
    ) or _AMBIGUOUS_CLOSE_SECRET_KEY_RE.search(joined):
        raise _TeamContextRedactionError
    without_sentinels = _CANONICAL_REDACTION_SENTINEL_RE.sub(
        _STRICT_REDACTION,
        joined,
    )
    return _CANONICAL_MASK_ARTIFACT_RE.sub(_STRICT_REDACTION, without_sentinels)


def redact_team_classifier_text(text: object) -> str | None:
    """Apply canonical then strict, fragment-free classifier redaction."""
    if type(text) is not str or len(text) > _MAX_STRICT_PARSE_CHARS:
        return None
    try:
        from agent.redact import redact_sensitive_text

        redacted = redact_sensitive_text(
            text,
            force=True,
            file_read=True,
            redact_url_credentials=True,
        )
        if type(redacted) is not str:
            return None
        strict_redacted = _strict_redact_credential_assignments(redacted)
        return strict_redacted if type(strict_redacted) is str else None
    except Exception:
        return None


def _is_exact_shared_root_source(
    source: object,
    root_message_id: str,
    coordinator_profile: str,
) -> bool:
    if not isinstance(source, SessionSource):
        return False
    chat_id = _normalize_group_chat_id(source.chat_id)
    expected_scope = f"telegram-team:{source.chat_id}:{root_message_id}"
    return bool(
        source.platform is Platform.TELEGRAM
        and source.chat_type == "group"
        and chat_id is not None
        and chat_id == source.chat_id
        and source.profile == coordinator_profile
        and source.session_scope_id == expected_scope
    )


def _same_shared_root(
    left: SessionSource,
    right: object,
    coordinator_profile: str,
) -> bool:
    if not isinstance(right, SessionSource):
        return False
    return bool(
        right.platform is Platform.TELEGRAM
        and right.chat_type == "group"
        and right.chat_id == left.chat_id
        and right.thread_id == left.thread_id
        and right.scope_id == left.scope_id
        and right.profile == left.profile == coordinator_profile
        and right.session_scope_id == left.session_scope_id
    )


def _render_message(role: str, content: str) -> str:
    label = f"[UNTRUSTED {role}] "
    # Keep every physical line inside the same explicit data label. Collapsing
    # whitespace also prevents transcript text from manufacturing a trusted-
    # looking delimiter on a fresh line.
    redacted = redact_team_classifier_text(content)
    if redacted is None:
        raise _TeamContextRedactionError
    normalized = " ".join(redacted.split())
    return f"{label}{normalized}" if normalized else ""


def _render_immediate_reply(
    source: SessionSource,
    reply_to_message_id: object,
    reply_to_text: object,
    raw_reply_to_message_id: object,
    *,
    raw_reply_present: object,
    raw_reply_provenance_safe: object,
    raw_reply_to_text: object,
    raw_reply_chat_id: object,
    raw_reply_thread_id: object,
    raw_reply_author_is_bot: object,
    raw_external_reply_present: object,
    require_complete_reply_provenance: object,
) -> tuple[bool, str | None]:
    """Render one validated immediate Telegram reply as explicitly untrusted."""
    if (
        reply_to_message_id is None
        and reply_to_text is None
        and raw_reply_to_message_id is None
    ):
        if require_complete_reply_provenance is True and (
            raw_reply_provenance_safe is not True
            or raw_reply_present is not False
            or raw_reply_to_text is not None
            or raw_reply_chat_id is not None
            or raw_reply_thread_id is not None
            or raw_reply_author_is_bot is not None
            or raw_external_reply_present is not False
        ):
            return False, None
        return True, None
    normalized_event_id = _normalize_message_id(reply_to_message_id)
    normalized_raw_id = _normalize_message_id(raw_reply_to_message_id)
    if (
        normalized_event_id is None
        or normalized_raw_id is None
        or normalized_event_id != normalized_raw_id
        or type(reply_to_text) is not str
        or len(reply_to_text) > MAX_TEAM_CONTEXT_CHARS
    ):
        return False, None

    if require_complete_reply_provenance is True:
        normalized_raw_chat_id = _normalize_group_chat_id(raw_reply_chat_id)
        expected_thread_id = (
            _normalize_message_id(source.thread_id)
            if source.thread_id is not None
            else None
        )
        normalized_raw_thread_id = (
            _normalize_message_id(raw_reply_thread_id)
            if raw_reply_thread_id is not None
            else None
        )
        if (
            raw_reply_provenance_safe is not True
            or raw_reply_present is not True
            or type(raw_external_reply_present) is not bool
            or raw_external_reply_present
            or type(raw_reply_to_text) is not str
            or raw_reply_to_text != reply_to_text
            or len(raw_reply_to_text) > MAX_TEAM_CONTEXT_CHARS
            or not reply_to_text.strip()
            or not raw_reply_to_text.strip()
            or normalized_raw_chat_id != source.chat_id
            or normalized_raw_thread_id != expected_thread_id
            or raw_reply_author_is_bot is not False
        ):
            return False, None
    elif require_complete_reply_provenance is not False:
        return False, None

    redacted = redact_team_classifier_text(reply_to_text)
    if redacted is None:
        raise _TeamContextRedactionError
    normalized = " ".join(redacted.split())
    if not normalized:
        return True, None
    return (
        True,
        f"[UNTRUSTED immediate reply id={normalized_event_id}] {normalized}",
    )


def _bounded_render(
    rows: list[Mapping[str, Any]],
    max_chars: int,
    trailing_material: str | None = None,
) -> str:
    rendered: list[str] = []
    for row in rows:
        role = row.get("role")
        if type(role) is not str:
            raise ValueError("invalid message role")
        if role not in _ALLOWED_ROLES:
            continue
        content = row.get("content")
        if type(content) is not str:
            raise ValueError("invalid message content")
        if row.get("display_kind") is not None:
            continue
        if any(row.get(key) for key in ("tool_calls", "tool_call_id", "tool_name")):
            continue
        material = _render_message(role, content[:MAX_TEAM_CONTEXT_CHARS])
        if material:
            rendered.append(material)

    if trailing_material:
        rendered.append(trailing_material)

    if not rendered:
        return ""

    budget = max_chars - len(_CONTEXT_HEADER) - 1
    if budget <= 0:
        return ""

    selected_newest_first: list[str] = []
    for material in reversed(rendered):
        separator_cost = 1 if selected_newest_first else 0
        available = budget - separator_cost
        if available <= 0:
            break
        if len(material) <= available:
            selected_newest_first.append(material)
            budget -= len(material) + separator_cost
            continue
        if not selected_newest_first:
            role_prefix_end = material.find("] ") + 2
            if role_prefix_end > 1 and available > role_prefix_end:
                selected_newest_first.append(material[: available - 1].rstrip() + "…")
        break

    if not selected_newest_first:
        return ""
    chronological = list(reversed(selected_newest_first))
    return _CONTEXT_HEADER + "\n" + "\n".join(chronological)


def collect_team_context(
    session_store: object,
    source: object,
    root_message_id: object,
    max_messages: int = MAX_TEAM_CONTEXT_MESSAGES,
    max_chars: int = MAX_TEAM_CONTEXT_CHARS,
    *,
    coordinator_profile: object = "default",
    reply_to_message_id: object = None,
    reply_to_text: object = None,
    raw_reply_to_message_id: object = None,
    raw_reply_present: object = False,
    raw_reply_provenance_safe: object = True,
    raw_reply_to_text: object = None,
    raw_reply_chat_id: object = None,
    raw_reply_thread_id: object = None,
    raw_reply_author_is_bot: object = None,
    raw_external_reply_present: object = False,
    require_complete_reply_provenance: object = False,
    safety: TeamContextSafety | None = None,
) -> str:
    """Return bounded untrusted context from one exact coordinator root session.

    The lookup is read-only and exact-keyed. It never creates a session, scans
    other entries, reads a specialist profile, or loads system/tool payloads.
    Any invalid input, unexpected storage shape, or lookup failure returns an
    empty string.
    """
    if (
        type(max_messages) is not int
        or not 1 <= max_messages <= MAX_TEAM_CONTEXT_MESSAGES
        or type(max_chars) is not int
        or not 1 <= max_chars <= MAX_TEAM_CONTEXT_CHARS
    ):
        return ""
    if safety is not None and type(safety) is not TeamContextSafety:
        return ""
    if (
        type(coordinator_profile) is not str
        or _normalize_profile(coordinator_profile) != coordinator_profile
    ):
        if safety is not None:
            safety.unsafe = True
        return ""
    safe_coordinator_profile = cast(str, coordinator_profile)
    root_id = _normalize_message_id(root_message_id)
    if root_id is None:
        return ""
    if not _is_exact_shared_root_source(
        source,
        root_id,
        safe_coordinator_profile,
    ):
        if safety is not None:
            safety.unsafe = True
        return ""
    safe_source = cast(SessionSource, source)

    try:
        reply_safe, immediate_reply = _render_immediate_reply(
            safe_source,
            reply_to_message_id,
            reply_to_text,
            raw_reply_to_message_id,
            raw_reply_present=raw_reply_present,
            raw_reply_provenance_safe=raw_reply_provenance_safe,
            raw_reply_to_text=raw_reply_to_text,
            raw_reply_chat_id=raw_reply_chat_id,
            raw_reply_thread_id=raw_reply_thread_id,
            raw_reply_author_is_bot=raw_reply_author_is_bot,
            raw_external_reply_present=raw_external_reply_present,
            require_complete_reply_provenance=require_complete_reply_provenance,
        )
        if not reply_safe:
            if safety is not None:
                safety.unsafe = True
            return ""
        key_builder = getattr(session_store, "_generate_session_key", None)
        lookup = getattr(session_store, "lookup_loaded_session_by_key", None)
        if not callable(key_builder) or not callable(lookup):
            return ""
        session_key = key_builder(safe_source)
        if type(session_key) is not str or not session_key:
            return ""
        entry = lookup(session_key)
        transcript_limit = max_messages - (1 if immediate_reply else 0)
        raw_rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = []
        if entry is not None:
            if not _same_shared_root(
                safe_source,
                getattr(entry, "origin", None),
                safe_coordinator_profile,
            ):
                if safety is not None:
                    safety.unsafe = True
                return ""
            session_id = getattr(entry, "session_id", None)
            if type(session_id) is not str or not session_id:
                return ""
            if transcript_limit > 0:
                db = getattr(session_store, "_db", None)
                get_messages = getattr(db, "get_messages", None)
                if not callable(get_messages):
                    return ""
                raw_rows_result = get_messages(
                    session_id,
                    limit=transcript_limit,
                    latest=True,
                )
                if (
                    not isinstance(raw_rows_result, (list, tuple))
                    or len(raw_rows_result) > transcript_limit
                ):
                    return ""
                raw_rows = cast(
                    list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
                    raw_rows_result,
                )

        rows: list[Mapping[str, Any]] = []
        seen_ids: set[int] = set()
        for raw_row in raw_rows:
            if not isinstance(raw_row, Mapping):
                return ""
            row_id = raw_row.get("id")
            if type(row_id) is not int or row_id <= 0 or row_id in seen_ids:
                return ""
            seen_ids.add(row_id)
            rows.append(raw_row)
        rows.sort(key=lambda row: row["id"])
        return _bounded_render(rows, max_chars, immediate_reply)
    except _TeamContextRedactionError:
        if safety is not None:
            safety.unsafe = True
            safety.redaction_failed = True
        return ""
    except BaseException as exc:
        # Cancellation/system-exit cannot occur on this synchronous read helper;
        # hostile objects can nevertheless raise arbitrary BaseException from a
        # property. Fail closed without letting them escape the routing boundary.
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return ""
