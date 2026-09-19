"""Review regressions at provider, execution and readiness boundaries."""
import asyncio
import json
import threading
from types import SimpleNamespace

import httpx
import pytest
import yaml

from tests.gateway.test_team_dispatch_runtime import (
    fleet, model, PROMPTS, ingress, message, rows, scope, settle, tick,
)


def test_codex_explicit_route_is_not_re_resolved(monkeypatch):
    from agent import auxiliary_client as aux
    def forbidden():
        raise AssertionError("must not re-resolve the configured credential")
    monkeypatch.setattr(aux, "_read_codex_access_token", forbidden)
    monkeypatch.setattr(aux, "_select_pool_entry", lambda *a: (False, None))
    client, resolved = aux.resolve_provider_client(
        "openai-codex", model="gpt-6-astra", explicit_base_url="https://codex.invalid/responses-root",
        explicit_api_key="synthetic-only", api_mode="codex_responses")
    assert resolved == "gpt-6-astra"
    assert str(client._real_client.base_url) == "https://codex.invalid/responses-root/"
    assert client._real_client.api_key == "synthetic-only"
    client._real_client.close()


@pytest.mark.asyncio
async def test_durable_cancel_checked_before_tool_without_poll_window(fleet, model, monkeypatch, tmp_path):
    # Cancellation commits at the external response boundary with no event-loop
    # yield for the monitor. Tool dispatch itself must consult the durable fence.
    output = tmp_path / "no-late-write.txt"
    model.tool = ("write_file", {"path": str(output), "content": "AFTER_STOP"})
    original = httpx.HTTPTransport.handle_request
    def response(self, request):
        result = original(self, request)
        if json.loads(request.content).get("tools"):
            row = rows(fleet)[0]
            fleet["default"][0]._team_dispatch.store.cancel(row["root_request_id"], "111")
        return result
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", response)
    await tick(fleet)
    await ingress(fleet, message(PROMPTS["engineering"]))
    await settle(fleet)
    assert not output.exists()
    assert rows(fleet)[0]["error"] == "cancelled"
    assert not fleet["engineering"][1]._bot.sent


@pytest.mark.asyncio
async def test_cancel_during_constructor_never_starts_agent(fleet, model, monkeypatch):
    from run_agent import AIAgent
    original = AIAgent.__init__
    def construct(self, *a, **kw):
        original(self, *a, **kw)
        row = rows(fleet)[0]
        fleet["default"][0]._team_dispatch.store.cancel(row["root_request_id"], "111")
    monkeypatch.setattr(AIAgent, "__init__", construct)
    await tick(fleet)
    await ingress(fleet, message(PROMPTS["engineering"]))
    await settle(fleet)
    assert not [c for c in model if c.get("tools")]
    assert rows(fleet)[0]["error"] == "cancelled"


@pytest.mark.asyncio
async def test_reconnect_pending_then_current_delivers_once(fleet, model):
    from gateway.config import Platform
    from plugins.platforms.telegram.adapter import TelegramAdapter
    await tick(fleet)
    await ingress(fleet, message(PROMPTS["engineering"]))
    runner, ad, home = fleet["default"]
    with scope(home):
        await asyncio.gather(*tuple(ad._pending_text_batch_tasks.values()))
        await runner._team_dispatch.tick()
        await asyncio.gather(*tuple(runner._team_dispatch.tasks))
    runner, prior, home = fleet["engineering"]
    with scope(home):
        replacement = TelegramAdapter(prior.config)
        runner._wire_adapter_handlers(replacement)
        for _ in range(2):
            await runner._team_dispatch.tick()
            assert rows(fleet)[0]["state"] == "accepted"
        replacement._bot = prior._bot
        replacement._running = True
        # Connected but not published is still ineligible.
        await runner._team_dispatch.tick()
        assert rows(fleet)[0]["state"] == "accepted"
        runner._publish_primary_adapter(Platform.TELEGRAM, replacement)
        await runner._team_dispatch.tick()
        await asyncio.gather(*tuple(runner._team_dispatch.tasks))
    assert rows(fleet)[0]["state"] == "delivered"
    assert len(prior._bot.sent) == 1


@pytest.mark.asyncio
async def test_partial_disk_configuration_revokes_stale_member(fleet, model):
    await tick(fleet)
    home = fleet["design"][2]
    cfg = yaml.safe_load((home / "config.yaml").read_text())
    cfg["platforms"]["telegram"].pop("team_dispatch")
    cfg["platforms"]["telegram"]["require_mention"] = False
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))
    # No Design heartbeat needed for Ace to observe the revoked configuration.
    await ingress(fleet, message(PROMPTS["engineering"]))
    await settle(fleet)
    assert not model
    assert len(fleet["default"][1]._bot.sent) == 1
    for _, ad, _ in fleet.values():
        ad._message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_stale_classifier_snapshot_cannot_reclaim_routed_work(fleet, model):
    await tick(fleet)
    await ingress(fleet, message(PROMPTS["engineering"]))
    runner, ad, home = fleet["default"]
    await asyncio.gather(*tuple(ad._pending_text_batch_tasks.values()))
    store = runner._team_dispatch.store
    stale = store.pending()[0]
    assert store.claim(stale["request_id"], expected_owner="default", expected_reason="pending")
    assert store.route(stale, "engineering", "single_domain")
    assert not store.claim(stale["request_id"], expected_owner="default", expected_reason="pending")
    assert store.claim(stale["request_id"], expected_owner="engineering", expected_reason="single_domain")
