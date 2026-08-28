"""Bounded, root-scoped context collection for Telegram team routing."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.telegram_team_context import collect_team_context


_CHAT_ID = "-1001234567890"
_ROOT_ID = "300"
_SCOPE_ID = f"telegram-team:{_CHAT_ID}:{_ROOT_ID}"


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=_CHAT_ID,
        chat_name="Team Room",
        chat_type="group",
        user_id="123",
        user_name="Alice",
        thread_id="77",
        chat_topic="Planning",
        message_id="305",
        profile="default",
        session_scope_id=_SCOPE_ID,
    )


class _Store:
    def __init__(self, messages, *, origin: SessionSource | None = None):
        self.origin = origin or _source()
        self.entry = SimpleNamespace(
            session_id="shared-root-session", origin=self.origin
        )
        self.get_or_create_session = Mock(
            side_effect=AssertionError("context collection must not create sessions")
        )
        self.append_to_transcript = Mock(
            side_effect=AssertionError("context collection must not mutate transcripts")
        )
        self.lookup_by_session_key = Mock(return_value=self.entry)
        self._generate_session_key = Mock(return_value="exact-root-key")
        self._db = SimpleNamespace(get_messages=Mock(return_value=messages))


def test_collect_team_context_reads_newest_bound_then_restores_chronological_order():
    store = _Store([
        {"id": 3, "role": "assistant", "content": "third"},
        {"id": 2, "role": "user", "content": "second"},
    ])

    context = collect_team_context(
        store,
        _source(),
        _ROOT_ID,
        max_messages=2,
        max_chars=500,
    )

    store._db.get_messages.assert_called_once_with(
        "shared-root-session",
        limit=2,
        latest=True,
    )
    assert context.startswith("[UNTRUSTED TELEGRAM TEAM CONTEXT]")
    assert "[UNTRUSTED user] second" in context
    assert "[UNTRUSTED assistant] third" in context
    assert context.index("second") < context.index("third")
    assert len(context) <= 500
    store.get_or_create_session.assert_not_called()
    store.append_to_transcript.assert_not_called()


def test_collect_team_context_keeps_newest_material_within_character_bound():
    store = _Store([
        {"id": 12, "role": "assistant", "content": "N" * 300},
        {"id": 11, "role": "user", "content": "older should be omitted"},
    ])

    context = collect_team_context(
        store,
        _source(),
        _ROOT_ID,
        max_messages=2,
        max_chars=100,
    )

    assert context.startswith("[UNTRUSTED TELEGRAM TEAM CONTEXT]")
    assert "[UNTRUSTED assistant]" in context
    assert "older should be omitted" not in context
    assert len(context) <= 100


@pytest.mark.parametrize(
    "origin",
    [
        replace(_source(), chat_id="-1009999999999"),
        replace(_source(), thread_id="88"),
        replace(
            _source(),
            session_scope_id=f"telegram-team:{_CHAT_ID}:999",
        ),
        replace(_source(), profile="engineering"),
        replace(_source(), chat_type="dm", chat_id="123", thread_id=None),
    ],
    ids=["other-chat", "other-topic", "other-root", "specialist-private", "dm"],
)
def test_collect_team_context_rejects_private_or_cross_root_session(origin):
    store = _Store(
        [{"id": 1, "role": "user", "content": "must not leak"}],
        origin=origin,
    )

    assert collect_team_context(store, _source(), _ROOT_ID) == ""
    store._db.get_messages.assert_not_called()


def test_collect_team_context_rejects_malformed_group_provenance_before_lookup():
    hostile = replace(
        _source(),
        chat_id="-not-a-telegram-chat",
        session_scope_id="telegram-team:-not-a-telegram-chat:300",
    )
    store = _Store(
        [{"id": 1, "role": "user", "content": "must not leak"}],
        origin=hostile,
    )

    assert collect_team_context(store, hostile, _ROOT_ID) == ""
    store._db.get_messages.assert_not_called()


def test_collect_team_context_excludes_internal_prompts_tools_and_payloads():
    store = _Store([
        {"id": 5, "role": "user", "content": "safe request"},
        {"id": 6, "role": "system", "content": "internal prompt secret"},
        {"id": 7, "role": "tool", "content": "tool payload secret"},
        {
            "id": 8,
            "role": "assistant",
            "content": "assistant tool carrier",
            "tool_calls": [{"function": {"arguments": "credential"}}],
        },
        {
            "id": 9,
            "role": "user",
            "content": "internal notification secret",
            "display_kind": "internal_notification",
        },
    ])

    context = collect_team_context(store, _source(), _ROOT_ID)

    assert "safe request" in context
    assert "internal prompt" not in context
    assert "tool payload" not in context
    assert "tool carrier" not in context
    assert "internal notification" not in context


@pytest.mark.parametrize(
    ("store", "source", "root_message_id", "max_messages", "max_chars"),
    [
        (None, _source(), _ROOT_ID, 12, 12000),
        (_Store([]), object(), _ROOT_ID, 12, 12000),
        (_Store([]), _source(), "../../../private", 12, 12000),
        (_Store([]), _source(), _ROOT_ID, True, 12000),
        (_Store([]), _source(), _ROOT_ID, 0, 12000),
        (_Store([]), _source(), _ROOT_ID, 13, 12000),
        (_Store([]), _source(), _ROOT_ID, 12, float("inf")),
        (_Store([]), _source(), _ROOT_ID, 12, 12001),
    ],
    ids=[
        "missing-store",
        "hostile-source",
        "invalid-root",
        "bool-message-limit",
        "zero-message-limit",
        "oversize-message-limit",
        "non-integer-char-limit",
        "oversize-char-limit",
    ],
)
def test_collect_team_context_invalid_inputs_fail_closed(
    store,
    source,
    root_message_id,
    max_messages,
    max_chars,
):
    assert (
        collect_team_context(
            store,
            source,
            root_message_id,
            max_messages=max_messages,
            max_chars=max_chars,
        )
        == ""
    )


@pytest.mark.parametrize(
    "failure",
    ["key", "lookup", "db", "shape", "row"],
)
def test_collect_team_context_lookup_or_shape_failure_returns_empty(failure):
    store = _Store([{"id": 1, "role": "user", "content": "safe"}])
    if failure == "key":
        store._generate_session_key.side_effect = RuntimeError("key failure")
    elif failure == "lookup":
        store.lookup_by_session_key.side_effect = RuntimeError("lookup failure")
    elif failure == "db":
        store._db.get_messages.side_effect = RuntimeError("db failure")
    elif failure == "shape":
        store._db.get_messages.return_value = {"unexpected": "mapping"}
    else:
        store._db.get_messages.return_value = [
            {"id": object(), "role": "user", "content": "bad row"}
        ]

    assert collect_team_context(store, _source(), _ROOT_ID) == ""
