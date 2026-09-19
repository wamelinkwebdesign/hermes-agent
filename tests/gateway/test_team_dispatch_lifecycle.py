"""Ingress authorization, existing watcher, expiry and real import boundaries."""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.gateway.test_team_dispatch_runtime import (
    PROFILES, PROMPTS, fleet, model, ingress, message, rows, scope, settle, tick,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", ["sender", "bot", "chats", "topic", "disabled", "gateway_sender"])
async def test_admission_failures_never_model_or_native_fanout(fleet, model, gate):
    from gateway.config import Platform
    await tick(fleet)
    msg = message(PROMPTS["engineering"])
    if gate == "sender":
        msg.from_user.id = 222
    if gate == "bot":
        msg.from_user.is_bot = True
    for runner, ad, home in fleet.values():
        if gate == "chats":
            ad.config.extra["allowed_chats"] = ["-999"]
        if gate == "topic":
            ad.config.extra["allowed_topics"] = [8]
        if gate == "disabled":
            ad.config.enabled = False
        if gate == "gateway_sender":
            runner.config.platforms[Platform.TELEGRAM].extra["group_allow_from"] = ["222"]
    await ingress(fleet, msg)
    await settle(fleet)
    assert not model and not rows(fleet)
    for _, ad, _ in fleet.values():
        assert not ad._bot.sent
        ad._message_handler.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["media", "command", "missing_member", "mismatched_member"])
async def test_unsupported_or_incoherent_has_one_truthful_ace_outcome(fleet, model, kind):
    await tick(fleet)
    msg = message(None if kind == "media" else "/reset" if kind == "command" else PROMPTS["engineering"])
    if kind == "missing_member":
        fleet["design"][1].config.enabled = False
        with fleet["default"][0]._team_dispatch.store.connect() as db:
            db.execute("DELETE FROM team_members WHERE profile='design'")
    if kind == "mismatched_member":
        fleet["design"][1].config.extra["team_dispatch"]["destinations"].append({"chat_id": "-300", "thread_id": None})
        with scope(fleet["design"][2]):
            await fleet["design"][0]._team_dispatch.tick()
    if kind == "media":
        for _, ad, home in fleet.values():
            with scope(home):
                await ad._handle_media_message(SimpleNamespace(message=msg, update_id=10), None)
    else:
        await ingress(fleet, msg)
    await settle(fleet)
    assert not model
    assert sum(len(ad._bot.sent) for _, ad, _ in fleet.values()) == 1
    assert "ondersteunt" in fleet["default"][1]._bot.sent[0]["text"]
    assert rows(fleet)[0]["fallback_receipt"]


@pytest.mark.asyncio
async def test_direct_specialist_control_stays_on_native_path(fleet, model):
    await tick(fleet)
    msg = message("/help@engineering_bot")
    msg.entities = [SimpleNamespace(type="bot_command", offset=0, length=len(msg.text))]
    await ingress(fleet, msg)
    await asyncio.sleep(0)
    await settle(fleet)
    assert not model and not rows(fleet)
    fleet["engineering"][1]._message_handler.assert_called_once()
    for p, (_, ad, _) in fleet.items():
        if p != "engineering":
            ad._message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_expiry_restart_and_receipt_dedup(fleet, model):
    from gateway.run_team_dispatch import install_team_dispatch
    await tick(fleet)
    first = message(PROMPTS["engineering"])
    await ingress(fleet, first)
    await settle(fleet)
    count = len(model)
    for runner, ad, home in fleet.values():
        with scope(home):
            install_team_dispatch(runner, ad)
    await ingress(fleet, first)
    await settle(fleet)
    assert len(model) == count
    assert sum(len(ad._bot.sent) for _, ad, _ in fleet.values()) == 1
    await ingress(fleet, message(PROMPTS["design"], mid=20))
    ad = fleet["default"][1]
    with scope(fleet["default"][2]):
        await asyncio.gather(*tuple(ad._pending_text_batch_tasks.values()))
    with fleet["default"][0]._team_dispatch.store.connect() as db:
        db.execute("UPDATE public_handoffs SET expires_at=1 WHERE state='accepted'")
    await settle(fleet)
    assert len(model) == count
    assert rows(fleet)[-1]["state"] == "failed_definitive"
    assert len(fleet["default"][1]._bot.sent) == 1


@pytest.mark.asyncio
async def test_actual_watcher_not_only_direct_dispatch_calls(fleet, model):
    tasks = []
    try:
        for runner, ad, home in fleet.values():
            with scope(home):
                tasks.append(asyncio.create_task(runner._handoff_watcher(interval=0.02)))
        async with asyncio.timeout(12):
            while len(fleet["default"][0]._team_dispatch.store.members()) != len(PROFILES):
                await asyncio.sleep(0.02)
            await ingress(fleet, message(PROMPTS["finance-ops"]))
            while not fleet["finance-ops"][1]._bot.sent:
                await asyncio.sleep(0.02)
        assert rows(fleet)[0]["state"] == "delivered"
        assert sum(len(ad._bot.sent) for _, ad, _ in fleet.values()) == 1
    finally:
        for runner, _, _ in fleet.values():
            runner._running = False
        await asyncio.gather(*tasks)


def test_fresh_process_imports_candidate_runtime(tmp_path):
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[2]
    code = """import json, pathlib
import gateway.run_team_dispatch as runtime
import gateway.team_dispatch_agent as agent
import plugins.platforms.telegram.adapter as adapter
from gateway.team_dispatch_policy import TeamPolicy
assert TeamPolicy.from_extra({}) is None
print(json.dumps([str(pathlib.Path(m.__file__).resolve()) for m in (runtime, agent, adapter)]))
"""
    home = tmp_path / "fresh-home"
    home.mkdir()
    result = subprocess.run([sys.executable, "-c", code], cwd=repo, text=True, capture_output=True,
        env={"PATH": os.environ["PATH"], "HOME": str(home), "HERMES_HOME": str(home),
             "HERMES_TESTING": "1", "PYTHONPATH": str(repo)}, timeout=30)
    assert result.returncode == 0, result.stderr
    paths = json.loads(result.stdout.strip().splitlines()[-1])
    assert len(paths) == 3 and all(Path(p).is_relative_to(repo) for p in paths)
