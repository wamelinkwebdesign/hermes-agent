"""Independent review probes, not part of the frozen implementation.

Canonical runner loads real parent conftests. Synthetic HTTP/Bot boundaries
reuse candidate fixtures; no live profile, credential or transport access.
"""
import asyncio
import json
import threading
from types import SimpleNamespace

import httpx
import pytest

from tests.gateway.test_team_dispatch_runtime import (
    PROMPTS, fleet, model, ingress, message, rows, scope, settle, tick,
)


def test_codex_classifier_preserves_strict_schema():
    from agent.auxiliary_client import _CodexCompletionsAdapter
    from gateway.team_dispatch_agent import SELECTION_SCHEMA, CLASSIFY_PROMPT
    client = SimpleNamespace(base_url="https://chatgpt.com/backend-api/codex")
    adapter = _CodexCompletionsAdapter(client, "gpt-6-astra")
    kwargs, _, _ = adapter._build_responses_kwargs({
        "model": "gpt-6-astra", "messages": [
            {"role": "system", "content": CLASSIFY_PROMPT},
            {"role": "user", "content": json.dumps({"prompt": PROMPTS["engineering"]})}],
        "tools": [], "stream": False, "max_tokens": 256, "timeout": 20,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "team_owner", "strict": True, "schema": SELECTION_SCHEMA}},
    })
    assert kwargs.get("text", {}).get("format") == {
        "type": "json_schema", "name": "team_owner", "strict": True, "schema": SELECTION_SCHEMA,
    }, kwargs


@pytest.mark.asyncio
async def test_stop_before_debounce_flush_cancels_original(fleet, model):
    await tick(fleet)
    root = message(PROMPTS["engineering"])
    await ingress(fleet, root)
    # Neither text batch has flushed, like a prompt followed immediately by stop.
    await ingress(fleet, message("/stop", mid=11, reply=root))
    await settle(fleet)
    agent_calls = [c["model"] for c in model if c.get("tools")]
    assert not agent_calls, {"calls": agent_calls, "rows": rows(fleet)}
    assert not fleet["engineering"][1]._bot.sent


@pytest.mark.asyncio
async def test_stop_interrupts_active_agent_before_next_tool(fleet, model, monkeypatch, tmp_path):
    # Real normal agent and real harmless file tool. Gate only the external model.
    output = tmp_path / "must-not-write-after-stop.txt"
    model.tool = ("write_file", {"path": str(output), "content": "AFTER_STOP"})
    entered, release = threading.Event(), threading.Event()
    original = httpx.HTTPTransport.handle_request
    def gated(self, request):
        payload = json.loads(request.content)
        if payload.get("tools") and not entered.is_set():
            entered.set()
            assert release.wait(10), "test failed to release fake model boundary"
        return original(self, request)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", gated)
    await tick(fleet)
    root = message(PROMPTS["engineering"])
    await ingress(fleet, root)
    runner, ad, home = fleet["default"]
    with scope(home):
        await asyncio.gather(*tuple(ad._pending_text_batch_tasks.values()))
        await runner._team_dispatch.tick()
        await asyncio.gather(*tuple(runner._team_dispatch.tasks))
    receiver, _, receiver_home = fleet["engineering"]
    try:
        with scope(receiver_home):
            await receiver._team_dispatch.tick()
            async with asyncio.timeout(10):
                while not entered.is_set():
                    await asyncio.sleep(0.01)
            assert rows(fleet)[0]["state"] == "running"
            await ingress(fleet, message("/stop", mid=11, reply=root))
            assert rows(fleet)[0]["error"] == "cancelled"
            release.set()
            await asyncio.gather(*tuple(receiver._team_dispatch.tasks))
    finally:
        release.set()
    await settle(fleet)
    assert not output.exists(), {"effect": output.read_text(), "row": rows(fleet)[0]}


@pytest.mark.asyncio
async def test_replacement_adapter_does_not_claim_before_publication(fleet, model):
    from gateway.run_team_dispatch import install_team_dispatch
    from plugins.platforms.telegram.adapter import TelegramAdapter
    await tick(fleet)
    await ingress(fleet, message(PROMPTS["engineering"]))
    runner, ad, home = fleet["default"]
    with scope(home):
        await asyncio.gather(*tuple(ad._pending_text_batch_tasks.values()))
        await runner._team_dispatch.tick()
        await asyncio.gather(*tuple(runner._team_dispatch.tasks))
    receiver, current, receiver_home = fleet["engineering"]
    with scope(receiver_home):
        replacement = TelegramAdapter(current.config)
        # Handler wiring happens before successful connection/publication.
        install_team_dispatch(receiver, replacement)
        await receiver._team_dispatch.tick()
        await asyncio.gather(*tuple(receiver._team_dispatch.tasks))
    assert rows(fleet)[0]["state"] == "accepted", rows(fleet)[0]


@pytest.mark.asyncio
async def test_mismatched_disabled_participant_never_native_fanout(fleet, model):
    await tick(fleet)
    # A missing opt-in is also a coherent-config mismatch, not global native enablement.
    fleet["design"][1].config.extra["team_dispatch"]["enabled"] = False
    fleet["design"][1].config.extra["require_mention"] = False
    await ingress(fleet, message(PROMPTS["engineering"]))
    await settle(fleet)
    fleet["design"][1]._message_handler.assert_not_called()
