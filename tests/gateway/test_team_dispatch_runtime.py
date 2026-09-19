"""Real ingress, profile runtime/tools and transport, external services replaced."""
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import yaml

from gateway.config import Platform, load_gateway_config
from gateway.run import GatewayRunner
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from plugins.platforms.telegram.adapter import TelegramAdapter

PROFILES = ("default", "revenue", "product", "design", "engineering", "finance-ops")
PROMPTS = {
    "engineering": "Hoe testen we de rollback van deze API veilig?",
    "product": "Welke gebruikersbehoefte moeten we als eerste valideren?",
    "design": "Hoe maken we de navigatie duidelijker voor de gebruiker?",
    "revenue": "Welke verkoopaanpak past bij deze doelgroep?",
    "finance-ops": "Hoe berekenen we de operationele maandkosten?",
    "default": "Vergelijk de belangen van alle teams en adviseer mij.",
    "clarify": "Kun je dit voor mij regelen?",
}


def message(text="hello", mid=10, reply=None, chat=-100, thread=7, sender=111):
    return SimpleNamespace(message_id=mid, text=text, caption=None, entities=[], caption_entities=[],
        photo=None, video=None, voice=None, audio=None, document=None,
        chat=SimpleNamespace(id=chat, type="supergroup", title="Synthetic", is_forum=thread is not None),
        from_user=SimpleNamespace(id=sender, full_name="Test", first_name="Test", is_bot=False),
        message_thread_id=thread, is_topic_message=thread is not None, reply_to_message=reply,
        date=datetime.now(timezone.utc))


class Bot:
    def __init__(self, profile):
        self.id = 900 + PROFILES.index(profile)
        self.username = profile.replace("-", "_") + "_bot"
        self.sent = []

    async def send_message(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(message_id=self.id * 1000 + len(self.sent),
            chat=SimpleNamespace(id=int(kwargs["chat_id"])),
            message_thread_id=kwargs.get("message_thread_id"), from_user=SimpleNamespace(id=self.id))

    async def set_my_commands(self, *args, **kwargs):
        pass

    async def send_chat_action(self, **kwargs):
        pass


@contextmanager
def scope(home):
    token = set_hermes_home_override(home)
    with patch.dict(os.environ, HERMES_HOME=str(home)):
        try:
            yield
        finally:
            reset_hermes_home_override(token)


@pytest.fixture
def fleet(tmp_path, monkeypatch, request):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-model-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.invalid/v1")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "synthetic-token")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    result = {}
    for profile in PROFILES:
        home = root if profile == "default" else root / "profiles" / profile
        home.mkdir(parents=True, exist_ok=True)
        (home / "SOUL.md").write_text("You are the persistent " + profile + " specialist. Keep approvals.")
        cfg = {"model": {"default": "test-" + profile, "provider": "custom", "base_url": "http://model.invalid/v1"},
               "agent": {"max_turns": 4}, "terminal": {"cwd": str(tmp_path)},
               "approvals": {"mode": "manual"}, "fallback_model": None,
               "memory": {"memory_enabled": False, "user_profile_enabled": False},
               "display": {"streaming": False}, "background_review": {"enabled": False},
               "tools": {"telegram": {"enabled": ["file", "terminal"], "disabled": []}},
               "platforms": {"telegram": {"enabled": True, "require_mention": True,
                  "allowed_chats": [], "allowed_topics": [7, 8], "group_allow_from": ["111"],
                  "team_dispatch": {"enabled": True, "coordinator_profile": "default",
                      "specialist_profiles": list(PROFILES[1:]),
                      "destinations": [{"chat_id": "-100", "thread_id": "7"},
                                       {"chat_id": "-200", "thread_id": "8"}]}}}}
        (home / "config.yaml").write_text(yaml.safe_dump(cfg))
        with scope(home), patch.dict(os.environ):
            config = load_gateway_config()
            ad = TelegramAdapter(config.platforms[Platform.TELEGRAM])
            ad._bot = Bot(profile)
            ad._running = True  # Fake Telegram connection has completed.
            runner = GatewayRunner(config=config)
            runner.adapters = {Platform.TELEGRAM: ad}
            runner._running = True
            runner._wire_adapter_handlers(ad)
            # Native escape is observable; selected team text must never reach it.
            ad.set_message_handler(AsyncMock(return_value=None))
        result[profile] = (runner, ad, home)
    yield result
    # Synthetic runtime receipts survive an explicit --basetemp evidence run.
    calls = request.node.funcargs.get("model", [])
    receipt = {
        "test": request.node.name,
        "requests": [{k: row[k] for k in ("brief", "target_profile", "reason_code", "state", "chat_id", "thread_id", "input_ids")}
                     for row in rows(result)],
        "classifier_calls": sum(c.get("response_format", {}).get("json_schema", {}).get("name") == "team_owner" for c in calls),
        "agent_api_calls": [c["model"] for c in calls if c.get("tools")],
        "outbound": [{"profile": p, "bot_id": ad._bot.id, "target": sent["chat_id"],
                      "thread": sent.get("message_thread_id"), "text": sent["text"]}
                     for p, (_, ad, _) in result.items() for sent in ad._bot.sent],
        "external_boundaries": "synthetic HTTP model completions and Telegram Bot API",
    }
    (tmp_path / "scenario.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    for runner, ad, home in result.values():
        runner._running = False
        runner._shutdown_event.set()


@pytest.fixture
def model(monkeypatch):
    class Calls(list):
        tool = None
        final = None
        invalid_classifier = False
    calls = Calls()
    def response(request):
        payload = json.loads(request.content)
        calls.append(payload)
        text = payload["messages"][-1]["content"]
        if payload.get("response_format", {}).get("json_schema", {}).get("name") == "team_owner":
            if "related" in payload["response_format"]["json_schema"]["schema"]["properties"]:
                content = json.dumps({"related": json.loads(text)["new"] == "Werk dat verder uit."})
            else:
                selected = next((p for p, s in PROMPTS.items() if s in text), "engineering")
                decision = "clarify" if selected == "clarify" else "self" if selected == "default" else "specialist"
                content = json.dumps({"decision": decision, "specialist": selected if decision == "specialist" else None,
                                      "reason_code": "ambiguous" if decision == "clarify" else "general" if decision == "self" else "single_domain"})
        else:
            content = "Antwoord van " + payload["model"]
        if calls.invalid_classifier and payload.get("response_format", {}).get("json_schema", {}).get("name") == "team_owner":
            content = '{"decision":"specialist","specialist":"fitness-pilot","reason_code":"single_domain"}'
        tool_calls = None
        if payload.get("tools"):
            if calls.tool and not any(m["role"] == "tool" for m in payload["messages"]):
                tool_calls = [{"id": "call_local", "type": "function", "function": {
                    "name": calls.tool[0], "arguments": json.dumps(calls.tool[1])}}]
                content = None
            elif calls.final is not None:
                content = calls.final
        data = {"id": "completion-test", "object": "chat.completion", "created": 1, "model": payload["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
                             "finish_reason": "tool_calls" if tool_calls else "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        if payload.get("stream"):
            chunk = {**data, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}]}
            if tool_calls:
                chunk["choices"][0]["delta"]["tool_calls"] = [{**call, "index": i} for i, call in enumerate(tool_calls)]
            end = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}]}
            return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"},
                                  text="data: " + json.dumps(chunk) + "\n\ndata: " + json.dumps(end) + "\n\ndata: [DONE]\n\n")
        return httpx.Response(200, request=request, json=data)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, request: response(request))
    async def async_response(self, request):
        return response(request)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_response)
    return calls


async def tick(fleet):
    for runner, ad, home in fleet.values():
        with scope(home):
            if ad._pending_text_batch_tasks:
                await asyncio.gather(*tuple(ad._pending_text_batch_tasks.values()))
            await runner._team_dispatch.tick()
            if runner._team_dispatch.tasks:
                await asyncio.gather(*tuple(runner._team_dispatch.tasks))


async def ingress(fleet, msg):
    for profile in reversed(PROFILES):
        runner, ad, home = fleet[profile]
        with scope(home):
            update = SimpleNamespace(message=msg, effective_message=msg, update_id=msg.message_id)
            handler = ad._handle_command if (msg.text or "").startswith("/") else ad._handle_text_message
            await handler(update, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [*PROFILES, "clarify"])
async def test_ordinary_dutch_question_reaches_real_profile_and_own_bot(fleet, model, selected):
    assert all(hasattr(r, "_team_dispatch") for r, _, _ in fleet.values()), "team dispatcher must be wired into actual gateways"
    await tick(fleet)
    await ingress(fleet, message(PROMPTS[selected]))
    await tick(fleet)
    await tick(fleet)
    sends = [(p, ad._bot.sent) for p, (_, ad, _) in fleet.items() if ad._bot.sent]
    owner = "default" if selected == "clarify" else selected
    assert len(sends) == 1 and sends[0][0] == owner, sends
    assert sends[0][1][0]["chat_id"] == -100
    assert sends[0][1][0]["message_thread_id"] == 7
    answer = rows(fleet)[0]["answer"]
    assert "test-" + owner in answer
    assert sends[0][1][0]["text"] == fleet[owner][1].format_message(answer)
    assert len([c for c in model if c.get("response_format", {}).get("json_schema", {}).get("name") == "team_owner"]) == 1
    assert [c["model"] for c in model if c.get("tools")] == ["test-" + owner]
    for _, ad, _ in fleet.values():
        ad._message_handler.assert_not_called()


async def settle(fleet):
    for _ in range(3):
        await tick(fleet)


@pytest.mark.asyncio
async def test_construction_failure_falls_through_instead_of_silent_swallow(fleet, monkeypatch):
    """A pre-enqueue contract violation must return False (native fall-through), never True.

    Regression for the silent-ingress outage: deadline 300 > wire cap 120 made every
    TeamRequest construction raise, and the adapter's broad catch consumed the message
    with no ledger row, no fallback and no user-visible signal.
    """
    import gateway.run_team_dispatch as rtd

    runner, _, home = fleet["default"]
    msg = message(PROMPTS["engineering"], mid=40)
    monkeypatch.setattr(rtd, "TEAM_DEADLINE_SECONDS", 601)  # above MAX_REQUEST_TTL_SECONDS
    with scope(home):
        result = await runner._team_dispatch.ingress(msg)
    assert result is False, "construction rejection must fall through to native handling"
    assert rows(fleet) == [], "nothing was enqueued, so at-most-once is not at risk"


def rows(fleet):
    store = fleet["default"][0]._team_dispatch.store
    with store.connect() as db:
        return [store.decode(r) for r in db.execute("SELECT * FROM public_handoffs ORDER BY created_at")]


@pytest.mark.asyncio
async def test_interleaved_roots_human_lineage_continuity_and_wrong_agent(fleet, model):
    await tick(fleet)
    first = message(PROMPTS["engineering"])
    other = message(PROMPTS["revenue"], mid=11)
    await ingress(fleet, first)
    await ingress(fleet, other)
    await settle(fleet)
    assert [r["target_profile"] for r in rows(fleet)] == ["engineering", "revenue"]
    # Reply to a human root, not just the bot final.
    follow = message("Werk dat verder uit.", mid=12, reply=first)
    await ingress(fleet, follow)
    await settle(fleet)
    assert rows(fleet)[-1]["target_profile"] == "engineering"
    assert rows(fleet)[-1]["root_request_id"] == rows(fleet)[0]["request_id"]
    # Related untagged input may reuse only this sender/group's recent verified context.
    await ingress(fleet, message("Werk dat verder uit.", mid=13))
    await settle(fleet)
    assert rows(fleet)[-1]["root_request_id"] == rows(fleet)[0]["request_id"]
    await ingress(fleet, message(PROMPTS["product"], mid=14))
    await settle(fleet)
    assert rows(fleet)[-1]["target_profile"] == "product"
    assert rows(fleet)[-1]["root_request_id"] == rows(fleet)[-1]["request_id"]
    await ingress(fleet, message("Verkeerde agent. " + PROMPTS["design"], mid=15, reply=follow))
    await settle(fleet)
    assert rows(fleet)[-1]["target_profile"] == "design"
    assert rows(fleet)[-1]["reassigned"] is True
    await ingress(fleet, message("Verkeerde agent. " + PROMPTS["revenue"], mid=16, reply=follow))
    await settle(fleet)
    assert rows(fleet)[-1]["target_profile"] == "default"


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", PROFILES[1:])
async def test_explicit_addressee_and_bot_reply_never_classify(fleet, model, profile):
    await tick(fleet)
    ad = fleet[profile][1]
    first = message("@" + ad._bot.username + " help")
    await ingress(fleet, first)
    await settle(fleet)
    reply = message("public answer", mid=ad._bot.id * 1000 + 1)
    reply.from_user = SimpleNamespace(id=ad._bot.id, is_bot=True)
    await ingress(fleet, message("More detail @default_bot", mid=12, reply=reply))
    await settle(fleet)
    assert len(ad._bot.sent) == 2
    assert {r["target_profile"] for r in rows(fleet)} == {profile}
    assert not [c for c in model if c.get("response_format", {}).get("json_schema", {}).get("name") == "team_owner"]
    assert [c["model"] for c in model if c.get("tools")] == ["test-" + profile] * 2


@pytest.mark.asyncio
async def test_full_profile_executes_real_file_tool_without_private_history(fleet, model, tmp_path):
    public = tmp_path / "public-proof.txt"
    public.write_text("PUBLIC_TOOL_EXECUTION_CANARY")
    model.tool = ("read_file", {"path": str(public)})
    for runner, ad, home in fleet.values():
        db = runner.session_store._db
        db.create_session("private-bot-chat", "cli")
        db.append_message("private-bot-chat", "user", "PRIVATE_HISTORY_DO_NOT_TRANSFER")
    await tick(fleet)
    await ingress(fleet, message(PROMPTS["engineering"]))
    await settle(fleet)
    assert rows(fleet)[0]["state"] == "delivered"
    tool_results = [m for c in model for m in c.get("messages", []) if m["role"] == "tool"]
    assert any("PUBLIC_TOOL_EXECUTION_CANARY" in str(m["content"]) for m in tool_results), tool_results
    assert "PRIVATE_HISTORY_DO_NOT_TRANSFER" not in json.dumps(model)
    assert "persistent engineering specialist" in json.dumps(model)
    assert len(fleet["engineering"][1]._bot.sent) == 1


@pytest.mark.asyncio
async def test_actual_dangerous_tool_approval_is_not_granted(fleet, model, tmp_path):
    protected = tmp_path / "protected"
    protected.mkdir()
    (protected / "keep.txt").write_text("keep")
    model.tool = ("terminal", {"command": "rm -rf " + str(protected)})
    await tick(fleet)
    await ingress(fleet, message(PROMPTS["engineering"]))
    await settle(fleet)
    assert (protected / "keep.txt").read_text() == "keep"
    results = [str(m["content"]) for c in model for m in c.get("messages", []) if m["role"] == "tool"]
    assert any("denied" in r.lower() or "blocked" in r.lower() for r in results), results
    assert rows(fleet)[0]["state"] == "failed_definitive"
    assert len(fleet["default"][1]._bot.sent) == 1
    assert not fleet["engineering"][1]._bot.sent


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["silent", "invalid_classifier", "send_rejected", "send_uncertain", "receipt_mismatch"])
async def test_failures_fallback_only_when_no_uncertain_send(fleet, model, monkeypatch, kind):
    from telegram.error import BadRequest
    if kind == "silent":
        model.final = "NO_REPLY"
    if kind == "invalid_classifier":
        model.invalid_classifier = True
    bot = fleet["engineering"][1]._bot
    original = bot.send_message
    async def send(**kwargs):
        if kind == "send_rejected":
            raise BadRequest("synthetic definitive rejection")
        if kind == "send_uncertain":
            await original(**kwargs)
            raise TimeoutError("PRIVATE_ERROR_MUST_NOT_LEAK")
        receipt = await original(**kwargs)
        if kind == "receipt_mismatch":
            receipt.chat.id = -999
        return receipt
    monkeypatch.setattr(bot, "send_message", send)
    await tick(fleet)
    msg = message(PROMPTS["engineering"])
    await ingress(fleet, msg)
    await settle(fleet)
    before = len(model)
    await ingress(fleet, msg)
    await settle(fleet)
    assert len(model) == before
    uncertain = kind in {"send_uncertain", "receipt_mismatch"}
    assert rows(fleet)[0]["state"] == ("delivery_uncertain" if uncertain else "failed_definitive")
    assert len(fleet["default"][1]._bot.sent) == (0 if uncertain else 1)
    assert "PRIVATE_ERROR_MUST_NOT_LEAK" not in json.dumps(rows(fleet))
    if uncertain:
        await ingress(fleet, message("follow-up", mid=12, reply=msg))
        await settle(fleet)
        assert len(rows(fleet)) == 1


@pytest.mark.asyncio
async def test_cross_group_topic_privacy_and_replayed_ids(fleet, model):
    await tick(fleet)
    await ingress(fleet, message(PROMPTS["engineering"] + " GROUP_ONE_CANARY"))
    await settle(fleet)
    model.clear()
    second = message(PROMPTS["revenue"], chat=-200, thread=8)
    for _ in range(2):
        await ingress(fleet, second)
    await settle(fleet)
    assert len(rows(fleet)) == 2
    assert "GROUP_ONE_CANARY" not in json.dumps(model)
    assert fleet["revenue"][1]._bot.sent[0]["chat_id"] == -200
    assert fleet["revenue"][1]._bot.sent[0]["message_thread_id"] == 8


@pytest.mark.asyncio
async def test_real_split_batch_preserves_every_id_and_one_final(fleet, model):
    await tick(fleet)
    threshold = fleet["default"][1]._SPLIT_THRESHOLD
    first = message(PROMPTS["engineering"].ljust(threshold, "x"), mid=30)
    second = message("En controleer de tests.", mid=31)
    await ingress(fleet, first)
    await ingress(fleet, second)
    for runner, ad, home in fleet.values():
        with scope(home):
            if ad._pending_text_batch_tasks:
                await asyncio.gather(*tuple(ad._pending_text_batch_tasks.values()))
    await settle(fleet)
    assert len(rows(fleet)) == 1
    assert rows(fleet)[0]["input_ids"] == ["30", "31"]
    assert len(fleet["engineering"][1]._bot.sent) == 1
    await ingress(fleet, second)
    await settle(fleet)
    assert len(rows(fleet)) == 1
    await ingress(fleet, message("follow-up", mid=32, reply=second))
    await settle(fleet)
    assert rows(fleet)[-1]["target_profile"] == "engineering"
    assert rows(fleet)[-1]["root_request_id"] == rows(fleet)[0]["request_id"]
