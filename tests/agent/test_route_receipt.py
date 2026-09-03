"""Privacy-safe configured/requested/completed route receipts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.turn_context import (
    begin_route_attempt,
    finish_route_attempt,
    reset_route_receipt,
)
from agent.turn_finalizer import build_route_receipt, sanitize_route_receipt
from cli import _emit_quiet_route_receipt
from hermes_cli.cli_agent_setup_mixin import saved_profile_default_route


def _agent(*, model="gpt-5.6-sol", provider="openai-codex"):
    return SimpleNamespace(
        model=model,
        provider=provider,
        _fallback_index=0,
        _primary_runtime={"model": model, "provider": provider},
        _profile_default_route={"model": None, "provider": None},
    )


def _settled_attempt(agent, call, *, model=None, provider=None):
    attempt = begin_route_attempt(agent, model=model, provider=provider)
    try:
        response = call()
    except BaseException:
        finish_route_attempt(agent, attempt, outcome="failed")
        raise
    finish_route_attempt(agent, attempt, outcome="completed")
    return response


def test_product_profile_default_stays_distinct_from_openai_invocation():
    product_config = {
        "default": {
            "model": "claude-opus-5-max-cli",
            "provider": "custom:claude-max-bridge",
        },
        "provider": "auto",
    }
    agent = _agent()
    agent._profile_default_route = saved_profile_default_route(product_config)

    reset_route_receipt(agent)
    _settled_attempt(agent, lambda: object())
    receipt = build_route_receipt(
        agent,
        completed=True,
        failed=False,
        terminal_reason="text_response(finish_reason=stop)",
    )

    assert receipt["profile_default_route"] == {
        "model": "claude-opus-5-max-cli",
        "provider": "custom:claude-max-bridge",
    }
    assert receipt["requested_route"] == {
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
    }
    assert receipt["completed_route"] == receipt["requested_route"]


def test_primary_failure_then_fallback_success_records_physical_attempts():
    agent = _agent()
    reset_route_receipt(agent)

    with pytest.raises(RuntimeError):
        _settled_attempt(
            agent,
            lambda: (_ for _ in ()).throw(RuntimeError("down")),
        )

    agent.model = "qwen3-coder-plus"
    agent.provider = "openrouter"
    agent._fallback_index = 1
    response = _settled_attempt(agent, lambda: "ok")
    assert response == "ok"

    receipt = build_route_receipt(
        agent,
        completed=True,
        failed=False,
        terminal_reason="text_response(finish_reason=stop)",
    )
    assert receipt["attempt_count"] == 2
    assert receipt["successful_api_calls"] == 1
    assert receipt["fallback_used"] is True
    assert receipt["attempts"] == [
        {
            "ordinal": 1,
            "model": "gpt-5.6-sol",
            "provider": "openai-codex",
            "outcome": "failed",
        },
        {
            "ordinal": 2,
            "model": "qwen3-coder-plus",
            "provider": "openrouter",
            "outcome": "completed",
        },
    ]
    assert receipt["completed_route"] == {
        "model": "qwen3-coder-plus",
        "provider": "openrouter",
    }
    assert receipt["terminal_status"] == "completed"
    assert receipt["terminal_reason"] == "text_response"


def test_transport_model_alias_is_not_classified_as_fallback():
    agent = _agent(model="gpt-5.6-sol-900k", provider="openai-codex")
    reset_route_receipt(agent)

    response = _settled_attempt(
        agent,
        lambda: "ok",
        model="gpt-5.6-sol",
        provider="openai-codex",
    )
    assert response == "ok"

    receipt = build_route_receipt(
        agent,
        completed=True,
        failed=False,
        terminal_reason="text_response(finish_reason=stop)",
    )

    assert receipt["requested_route"] == {
        "model": "gpt-5.6-sol-900k",
        "provider": "openai-codex",
    }
    assert receipt["completed_route"] == {
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
    }
    assert receipt["fallback_used"] is False
    assert "fallback_activated" not in receipt["attempts"][0]


def test_all_failed_has_no_completed_route_and_never_echoes_exception_secret():
    secret = "sk-super-secret-tripwire"
    agent = _agent()
    reset_route_receipt(agent)

    for model, provider in (
        ("gpt-5.6-sol", "openai-codex"),
        ("qwen3-coder-plus", "openrouter"),
    ):
        agent.model = model
        agent.provider = provider
        with pytest.raises(RuntimeError):
            _settled_attempt(
                agent,
                lambda: (_ for _ in ()).throw(RuntimeError(secret)),
            )

    receipt = build_route_receipt(
        agent,
        completed=False,
        failed=True,
        terminal_reason=f"local_processing_error({secret})",
    )
    assert set(receipt) == {
        "profile_default_route",
        "requested_route",
        "completed_route",
        "attempt_count",
        "attempts",
        "fallback_used",
        "successful_api_calls",
        "elapsed_ms",
        "terminal_status",
        "terminal_reason",
    }
    assert receipt["completed_route"] is None
    assert receipt["successful_api_calls"] == 0
    assert receipt["terminal_status"] == "failed"
    assert receipt["terminal_reason"] == "local_processing_error"
    assert secret not in json.dumps(receipt)
    assert "base_url" not in json.dumps(receipt)


def test_quiet_receipt_uses_stderr_without_touching_stdout(capsys):
    agent = _agent()
    reset_route_receipt(agent)
    _settled_attempt(agent, lambda: "ok")
    receipt = build_route_receipt(
        agent,
        completed=True,
        failed=False,
        terminal_reason="text_response(finish_reason=stop)",
    )

    _emit_quiet_route_receipt({"route_receipt": receipt})

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("route_receipt: {")
    assert json.loads(captured.err.removeprefix("route_receipt: ")) == receipt


def test_quiet_receipt_strips_unknown_secret_fields(capsys):
    secret = "sk-quiet-secret"
    agent = _agent()
    reset_route_receipt(agent)
    _settled_attempt(agent, lambda: "ok")
    receipt = build_route_receipt(
        agent,
        completed=True,
        failed=False,
        terminal_reason="text_response",
    )
    receipt["prompt"] = secret
    receipt["attempts"][0]["exception"] = secret

    _emit_quiet_route_receipt({"route_receipt": receipt})

    captured = capsys.readouterr()
    assert captured.out == ""
    assert secret not in captured.err
    emitted = json.loads(captured.err.removeprefix("route_receipt: "))
    assert "prompt" not in emitted
    assert "exception" not in emitted["attempts"][0]


def test_returned_but_rejected_response_is_not_successful():
    agent = _agent()
    reset_route_receipt(agent)
    attempt = begin_route_attempt(agent)

    # The provider returned an object, but downstream validation classified
    # its terminal status as cancelled/invalid.
    finish_route_attempt(agent, attempt, outcome="failed")
    receipt = build_route_receipt(
        agent,
        completed=False,
        failed=True,
        terminal_reason="invalid_response",
    )

    assert receipt["attempt_count"] == 1
    assert receipt["successful_api_calls"] == 0
    assert receipt["completed_route"] is None
    assert receipt["terminal_status"] == "failed"
    assert receipt["attempts"][0]["outcome"] == "failed"


def test_sanitizer_strips_arbitrary_nested_fields_and_recomputes_counts():
    secret = "sk-private-tripwire"
    sanitized = sanitize_route_receipt({
        "profile_default_route": {
            "model": "profile-model",
            "provider": "profile-provider",
            "api_key": secret,
        },
        "requested_route": {
            "model": "requested-model",
            "provider": "requested-provider",
            "base_url": f"https://{secret}.invalid",
        },
        "completed_route": {
            "model": "completed-model",
            "provider": "completed-provider",
            "prompt": secret,
        },
        "attempt_count": 999,
        "attempts": [{
            "ordinal": 44,
            "model": "completed-model",
            "provider": "completed-provider",
            "outcome": "completed",
            "exception": secret,
        }],
        "fallback_used": False,
        "successful_api_calls": 999,
        "elapsed_ms": 12,
        "terminal_status": "completed",
        "terminal_reason": "text_response",
        "prompt": secret,
        "arbitrary": {"credential": secret},
    })

    assert sanitized is not None
    assert set(sanitized) == {
        "profile_default_route",
        "requested_route",
        "completed_route",
        "attempt_count",
        "attempts",
        "fallback_used",
        "successful_api_calls",
        "elapsed_ms",
        "terminal_status",
        "terminal_reason",
    }
    assert sanitized["attempt_count"] == 1
    assert sanitized["successful_api_calls"] == 1
    assert sanitized["attempts"] == [{
        "ordinal": 1,
        "model": "completed-model",
        "provider": "completed-provider",
        "outcome": "completed",
    }]
    assert secret not in json.dumps(sanitized)
