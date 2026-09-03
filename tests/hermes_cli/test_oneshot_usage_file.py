"""Tests for hermes -z --usage-file (per-run JSON usage report)."""

import json

from hermes_cli.oneshot import _write_usage_file


def _result(**overrides):
    base = {
        "estimated_cost_usd": 0.1234,
        "cost_status": "estimated",
        "cost_source": "pricing-table",
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_read_tokens": 800,
        "cache_write_tokens": 0,
        "reasoning_tokens": 50,
        "total_tokens": 1250,
        "api_calls": 3,
        "model": "openai/gpt-5.5",
        "provider": "openrouter",
        "session_id": "abc123",
        "completed": True,
        "failed": False,
        "route_receipt": {
            "profile_default_route": {
                "model": "claude-opus-5-max-cli",
                "provider": "custom:claude-max-bridge",
            },
            "requested_route": {
                "model": "openai/gpt-5.5",
                "provider": "openrouter",
            },
            "completed_route": {
                "model": "openai/gpt-5.5",
                "provider": "openrouter",
            },
            "attempt_count": 1,
            "attempts": [{
                "ordinal": 1,
                "model": "openai/gpt-5.5",
                "provider": "openrouter",
                "outcome": "completed",
            }],
            "fallback_used": False,
            "successful_api_calls": 1,
            "elapsed_ms": 12,
            "terminal_status": "completed",
            "terminal_reason": "text_response",
        },
    }
    base.update(overrides)
    return base


class TestWriteUsageFile:
    def test_writes_report_with_cost_and_tokens(self, tmp_path):
        path = tmp_path / "usage.json"
        _write_usage_file(str(path), _result())
        report = json.loads(path.read_text())
        assert report["estimated_cost_usd"] == 0.1234
        assert report["input_tokens"] == 1000
        assert report["output_tokens"] == 200
        assert report["model"] == "openai/gpt-5.5"
        assert report["api_calls"] == 3
        assert report["failed"] is False
        assert report["route_receipt"] == _result()["route_receipt"]
        assert "failure" not in report

    def test_none_path_is_noop(self, tmp_path):
        # Must not raise and must not create a report file.
        _write_usage_file(None, _result())
        assert not (tmp_path / "usage.json").exists()

    def test_failure_marks_failed_and_records_message(self, tmp_path):
        path = tmp_path / "usage.json"
        _write_usage_file(str(path), {}, failure="boom")
        report = json.loads(path.read_text())
        assert report["failed"] is True
        assert report["failure"] == "boom"
        # Missing result fields serialize as null, not KeyError.
        assert report["estimated_cost_usd"] is None

    def test_route_receipt_is_allowlist_sanitized(self, tmp_path):
        secret = "sk-usage-file-secret"
        result = _result()
        result["route_receipt"] = {
            **result["route_receipt"],
            "prompt": secret,
            "attempts": [{
                **result["route_receipt"]["attempts"][0],
                "exception": secret,
            }],
        }
        path = tmp_path / "usage.json"

        _write_usage_file(str(path), result)

        report = json.loads(path.read_text())
        serialized = json.dumps(report["route_receipt"])
        assert secret not in serialized
        assert "prompt" not in report["route_receipt"]
        assert "exception" not in report["route_receipt"]["attempts"][0]
