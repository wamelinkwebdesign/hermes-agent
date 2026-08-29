"""Read-only, root-scoped context for Telegram team classification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any, cast
from urllib.parse import unquote_plus

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.telegram_team_routing import (
    _normalize_group_chat_id,
    _normalize_message_id,
    _normalize_profile,
)

MAX_TEAM_CONTEXT_MESSAGES = 12
MAX_TEAM_CONTEXT_CHARS = 12_000
MAX_TEAM_CLASSIFIER_INPUT_BYTES = 64 * 1_024
_CONTEXT_HEADER = "[UNTRUSTED TELEGRAM TEAM CONTEXT]"
_ALLOWED_ROLES = frozenset({"user", "assistant"})
_SENSITIVE_FIELD_COMPONENTS = frozenset({
    "apikey",
    "auth",
    "authorization",
    "credential",
    "credentials",
    "passphrase",
    "passwd",
    "password",
    "pw",
    "secret",
    "token",
})
_SENSITIVE_FIELD_COMPOUNDS = frozenset({
    "accesskey",
    "apikey",
    "authkey",
    "authtoken",
    "clientsecret",
    "privatekey",
    "secretkey",
})
_SENSITIVE_FIELD_PAIRS = frozenset({
    ("access", "key"),
    ("api", "key"),
    ("auth", "key"),
    ("auth", "token"),
    ("client", "secret"),
    ("private", "key"),
})
_STRUCTURED_FIELD_START_RE = re.compile(
    r"(?:(?P<line>^[ \t]*(?:export[ \t]+)?)|"
    r"(?P<container>(?<=[{,])[ \t]*))"
    r"(?P<key>\"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\"|"
    r"'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}'|"
    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127})"
    r"[ \t]*(?P<separator>[:=])[ \t]*",
    re.MULTILINE,
)
_STRUCTURED_FIELD_KEY_CANDIDATE_RE = re.compile(
    r"(?:(?P<line>^[ \t]*(?:export[ \t]+)?)|"
    r"(?P<container>(?<=[{,])[ \t]*))"
    r"(?P<key>[\"']?[A-Za-z0-9][A-Za-z0-9_.-]{0,127}[\"']?)"
    r"[ \t]*[:=]",
    re.MULTILINE,
)
_QUERY_FIELD_START_RE = re.compile(
    r"(?P<boundary>[?&;])(?P<key>[A-Za-z0-9_.~+%\-]{1,256})"
    r"[ \t]*=[ \t]*"
)
_STRICT_AUTHORIZATION_START_RE = re.compile(
    r"^[ \t]*(?:Proxy-)?Authorization[ \t]*[:=][ \t]*",
    re.IGNORECASE | re.MULTILINE,
)
_STRICT_BEARER_START_RE = re.compile(
    r"^[ \t]*Bearer[ \t]+",
    re.IGNORECASE | re.MULTILINE,
)
_STRICT_AWS_ACCESS_KEY_RE = re.compile(
    r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"
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


def _is_bounded_classifier_text(text: object) -> bool:
    """Accept exact strings whose complete UTF-8 representation is at most 64 KiB."""
    if type(text) is not str:
        return False
    try:
        encoded = text.encode("utf-8", errors="strict")
    except UnicodeError:
        return False
    return len(encoded) <= MAX_TEAM_CLASSIFIER_INPUT_BYTES


def _structured_key_components(key: str) -> tuple[str, ...]:
    """Return decoded, case-folded identifier components."""
    decoded = key
    for _ in range(3):
        next_value = unquote_plus(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    return tuple(
        component.casefold() for component in re.split(r"[_.-]+", decoded) if component
    )


def _is_sensitive_structured_key(key: str) -> bool:
    """Recognize sensitive components anywhere in a structured field name."""
    components = _structured_key_components(key)
    if not components:
        return False
    if any(
        component in _SENSITIVE_FIELD_COMPONENTS
        or component in _SENSITIVE_FIELD_COMPOUNDS
        for component in components
    ):
        return True
    return any(
        pair in _SENSITIVE_FIELD_PAIRS for pair in zip(components, components[1:])
    )


def _is_composite_sensitive_key(key: str) -> bool:
    components = _structured_key_components(key)
    return len(components) > 1 or any(
        component in _SENSITIVE_FIELD_COMPOUNDS for component in components
    )


def _reject_malformed_structured_field_keys(text: str) -> None:
    """Fail closed on mismatched quotes at an actual field boundary."""
    for match in _STRUCTURED_FIELD_KEY_CANDIDATE_RE.finditer(text):
        raw_key = match.group("key")
        opens_quoted = raw_key[0] in {'"', "'"}
        closes_quoted = raw_key[-1] in {'"', "'"}
        if not (opens_quoted or closes_quoted):
            continue
        if opens_quoted and closes_quoted and raw_key[0] == raw_key[-1]:
            continue
        if _is_sensitive_structured_key(raw_key.strip("\"'")):
            raise _TeamContextRedactionError


def _redact_query_fields(text: str) -> str:
    """Mask sensitive query values at exact parameter boundaries."""
    rendered: list[str] = []
    output_cursor = 0
    search_cursor = 0
    while match := _QUERY_FIELD_START_RE.search(text, search_cursor):
        search_cursor = match.end()
        if not _is_sensitive_structured_key(match.group("key")):
            continue
        value_start = match.end()
        value_end = value_start
        while value_end < len(text) and text[value_end] not in "&#;\r\n\t \"'<>?":
            if text[value_end] == "\x00":
                raise _TeamContextRedactionError
            value_end += 1
        if value_end == value_start:
            raise _TeamContextRedactionError
        rendered.append(text[output_cursor:value_start])
        rendered.append(_STRICT_REDACTION)
        output_cursor = value_end
        search_cursor = value_end
    rendered.append(text[output_cursor:])
    return "".join(rendered)


def _line_colon_value_looks_credential(
    text: str,
    value_start: int,
    field_end: int,
) -> bool:
    """Distinguish opaque values from ordinary ``label: prose`` sentences."""
    value = text[value_start:field_end].strip()
    if not value:
        raise _TeamContextRedactionError
    first_token = value.split(maxsplit=1)[0]
    return len(first_token) >= 16 or not first_token.isalpha()


def _redact_structured_fields(text: str) -> str:
    """Mask boundary-anchored env/config/JSON fields without matching prose."""
    _reject_malformed_structured_field_keys(text)
    rendered: list[str] = []
    output_cursor = 0
    search_cursor = 0
    while match := _STRUCTURED_FIELD_START_RE.search(text, search_cursor):
        search_cursor = match.end()
        raw_key = match.group("key")
        quoted_key = raw_key[0] in {'"', "'"}
        key = raw_key[1:-1] if quoted_key else raw_key
        if not _is_sensitive_structured_key(key):
            continue
        is_container = match.group("container") is not None
        if is_container and not quoted_key:
            continue

        value_start = match.end()
        if value_start >= len(text):
            raise _TeamContextRedactionError
        opening = text[value_start]
        is_quoted_value = opening in {'"', "'"}
        if is_quoted_value:
            value_end = _quoted_value_end(text, value_start, opening)
        elif is_container:
            value_end = value_start
            while value_end < len(text) and text[value_end] not in ",}\r\n":
                if text[value_end] == "\x00":
                    raise _TeamContextRedactionError
                value_end += 1
            if not text[value_start:value_end].strip():
                raise _TeamContextRedactionError
        else:
            value_end = _bounded_field_end(text, value_start)
            field_value = text[value_start:value_end]
            if not field_value.strip():
                raise _TeamContextRedactionError
            if any(
                character == "\x00"
                or (ord(character) < 0x20 and character not in "\t\r\n")
                for character in field_value
            ):
                raise _TeamContextRedactionError

        if (
            match.group("line") is not None
            and match.group("separator") == ":"
            and not quoted_key
            and not is_quoted_value
            and not _is_composite_sensitive_key(key)
            and not _line_colon_value_looks_credential(text, value_start, value_end)
        ):
            continue

        if is_quoted_value:
            rendered.append(text[output_cursor : value_start + 1])
            rendered.append(_STRICT_REDACTION)
            rendered.append(opening)
            if is_container:
                output_cursor = value_end
                search_cursor = value_end
            else:
                output_cursor = _bounded_field_end(text, value_end)
                search_cursor = output_cursor
        else:
            rendered.append(text[output_cursor:value_start])
            rendered.append(_STRICT_REDACTION)
            output_cursor = value_end
            search_cursor = value_end
    rendered.append(text[output_cursor:])
    return "".join(rendered)


def _normalize_canonical_redactions(text: str) -> str:
    without_sentinels = _CANONICAL_REDACTION_SENTINEL_RE.sub(
        _STRICT_REDACTION,
        text,
    )
    return _CANONICAL_MASK_ARTIFACT_RE.sub(_STRICT_REDACTION, without_sentinels)


def _strict_redact_credential_assignments(text: str) -> str:
    """Parse and completely mask bounded sensitive assignments and headers."""
    if not _is_bounded_classifier_text(text):
        raise _TeamContextRedactionError
    redacted = _redact_authorization_fields(text)
    redacted = _redact_bounded_fields(redacted, _STRICT_BEARER_START_RE)
    redacted = _redact_query_fields(redacted)
    redacted = _redact_structured_fields(redacted)
    redacted = _STRICT_AWS_ACCESS_KEY_RE.sub(_STRICT_REDACTION, redacted)
    return _normalize_canonical_redactions(redacted)


def redact_team_classifier_text(text: object) -> str | None:
    """Strictly redact a complete bounded value before any later truncation."""
    if not _is_bounded_classifier_text(text):
        return None
    try:
        from agent.redact import redact_sensitive_text

        strict_redacted = _strict_redact_credential_assignments(cast(str, text))
        redacted = redact_sensitive_text(
            strict_redacted,
            force=True,
            file_read=True,
            redact_url_credentials=True,
        )
        if type(redacted) is not str:
            return None
        return _normalize_canonical_redactions(redacted)
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
        or not _is_bounded_classifier_text(reply_to_text)
    ):
        return False, None
    safe_reply_text = cast(str, reply_to_text)

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
            or not _is_bounded_classifier_text(raw_reply_to_text)
            or raw_reply_to_text != reply_to_text
            or not safe_reply_text.strip()
            or not cast(str, raw_reply_to_text).strip()
            or normalized_raw_chat_id != source.chat_id
            or normalized_raw_thread_id != expected_thread_id
            or raw_reply_author_is_bot is not False
        ):
            return False, None
    elif require_complete_reply_provenance is not False:
        return False, None

    redacted = redact_team_classifier_text(safe_reply_text)
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
        material = _render_message(role, content)
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
