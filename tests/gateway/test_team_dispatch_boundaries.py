"""Additional finite-turn egress and native-lane regression evidence."""
import json
from types import SimpleNamespace

import pytest
import yaml

from tests.gateway.test_team_dispatch_runtime import (
    PROMPTS, fleet, model, ingress, message, rows, settle, tick,
)


@pytest.mark.asyncio
async def test_real_background_tool_cannot_arm_a_second_public_turn(fleet, model):
    model.tool = ("terminal", {"command": "printf finite-team-proof", "background": True, "notify": True})
    await tick(fleet)
    await ingress(fleet, message(PROMPTS["engineering"]))
    await settle(fleet)
    results = [json.loads(m["content"]) for c in model for m in c.get("messages", []) if m["role"] == "tool"]
    assert results and any(r.get("notify_on_complete") is False and r.get("notify_unsupported") for r in results)
    assert rows(fleet)[0]["state"] == "delivered"
    assert sum(len(ad._bot.sent) for _, ad, _ in fleet.values()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("native", ["dm", "nonteam", "disabled_team"])
async def test_team_callback_preserves_native_admission(fleet, model, native):
    await tick(fleet)
    msg = message("@engineering_bot help", chat=222 if native == "dm" else -300 if native == "nonteam" else -100)
    if native == "dm":
        msg.chat.type = "private"
    if native == "disabled_team":
        for _, ad, home in fleet.values():
            ad.config.extra["team_dispatch"]["enabled"] = False
            cfg = yaml.safe_load((home / "config.yaml").read_text())
            cfg["platforms"]["telegram"]["team_dispatch"]["enabled"] = False
            (home / "config.yaml").write_text(yaml.safe_dump(cfg))
    msg.entities = [SimpleNamespace(type="mention", offset=0, length=len("@engineering_bot"))]
    await ingress(fleet, msg)
    await settle(fleet)
    assert not model and not rows(fleet)
    fleet["engineering"][1]._message_handler.assert_called_once()


@pytest.mark.asyncio
async def test_expired_continuity_is_reclassified(fleet, model):
    await tick(fleet)
    await ingress(fleet, message(PROMPTS["revenue"]))
    await settle(fleet)
    with fleet["default"][0]._team_dispatch.store.connect() as db:
        db.execute("UPDATE public_handoffs SET send_started_at=1")
    model.clear()
    await ingress(fleet, message(PROMPTS["design"], mid=20))
    await settle(fleet)
    assert rows(fleet)[-1]["target_profile"] == "design"
    classifiers = [c for c in model if c.get("response_format", {}).get("json_schema", {}).get("name") == "team_owner"]
    assert len(classifiers) == 1
    assert "related" not in classifiers[0]["response_format"]["json_schema"]["schema"]["properties"]


@pytest.mark.asyncio
async def test_fallback_transport_uncertainty_is_not_retried(fleet, model, monkeypatch):
    model.final = "NO_REPLY"
    bot = fleet["default"][1]._bot
    original = bot.send_message
    async def uncertain(**kwargs):
        await original(**kwargs)
        raise TimeoutError("PRIVATE_FALLBACK_FAILURE")
    monkeypatch.setattr(bot, "send_message", uncertain)
    await tick(fleet)
    msg = message(PROMPTS["engineering"])
    await ingress(fleet, msg)
    await settle(fleet)
    await ingress(fleet, msg)
    await settle(fleet)
    assert len(bot.sent) == 1
    assert json.loads(rows(fleet)[0]["fallback_receipt"])["state"] == "delivery_uncertain"
    assert "PRIVATE_FALLBACK_FAILURE" not in json.dumps(rows(fleet))
