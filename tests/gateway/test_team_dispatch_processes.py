"""Real separate-process profile execution against one shared dispatch ledger."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

import pytest
import yaml

from tests.gateway.test_team_dispatch_runtime import PROFILES, PROMPTS

REPO = Path(__file__).resolve().parents[2]


def synthetic_profiles(tmp_path):
    root = tmp_path / ".hermes"
    for profile in PROFILES:
        home = root if profile == "default" else root / "profiles" / profile
        home.mkdir(parents=True, exist_ok=True)
        (home / "SOUL.md").write_text("You are " + profile + ". Keep approvals and private context private.")
        config = {
            "model": {"default": "test-" + profile, "provider": "custom", "base_url": "http://model.invalid/v1", "context_length": 128000},
            "agent": {"max_turns": 4}, "approvals": {"mode": "manual"},
            "memory": {"memory_enabled": False, "user_profile_enabled": False},
            "terminal": {"cwd": str(tmp_path), "oneshot_completion_wait_seconds": 0},
            "display": {"streaming": False}, "background_review": {"enabled": False},
            "fallback_model": None,
            "tools": {"telegram": {"enabled": ["file", "terminal"]}, "cli": {"enabled": ["file", "terminal"]}},
            "platforms": {"telegram": {"enabled": True, "require_mention": True,
                "allowed_chats": [], "allowed_topics": [7, 8], "group_allow_from": ["111"],
                "team_dispatch": {"enabled": True, "coordinator_profile": "default", "specialist_profiles": list(PROFILES[1:]),
                    "destinations": [{"chat_id": "-100", "thread_id": "7"}, {"chat_id": "-200", "thread_id": "8"}]}}},
        }
        (home / "config.yaml").write_text(yaml.safe_dump(config))
    return root


def process_env(root, profile):
    home = root if profile == "default" else root / "profiles" / profile
    return {"HOME": str(root.parent), "HERMES_HOME": str(home), "PATH": os.environ["PATH"],
            "PYTHONPATH": str(REPO), "PYTHONUNBUFFERED": "1", "HERMES_TESTING": "1", "TZ": "UTC",
            # Both HOME and HERMES_HOME are synthetic here. The production-DB
            # test guard otherwise mistakes this temporary .hermes for live.
            "HERMES_STATE_DB_GUARD_BYPASS": "1",
            "OPENAI_API_KEY": "synthetic-model-key", "OPENAI_BASE_URL": "http://model.invalid/v1",
            "TELEGRAM_BOT_TOKEN": "synthetic-token", "TERMINAL_CWD": str(root.parent)}


class ProcessGateway:
    def __init__(self, root, profile, suffix=""):
        self.profile = profile
        self.log = (root.parent / (profile + suffix + ".log")).open("w")
        self.proc = subprocess.Popen(
            [sys.executable, str(REPO / "tests/gateway/_team_dispatch_process_helper.py"), profile],
            cwd=root.parent, env=process_env(root, profile), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        self.responses = queue.Queue()
        def drain():
            for line in self.proc.stdout:
                self.responses.put(line)
            self.responses.put(None)
        threading.Thread(target=drain, daemon=True).start()
        self.identity = self.read()
        assert self.identity["status"] == "ready"
        assert Path(self.identity["runtime"]).is_relative_to(REPO)

    def read(self):
        line = self.responses.get(timeout=30)
        assert line is not None, "child exited; inspect synthetic process log"
        return json.loads(line)

    def send(self, command, **kwargs):
        self.proc.stdin.write(json.dumps({"cmd": command, **kwargs}) + "\n")
        self.proc.stdin.flush()
        result = self.read()
        assert result["status"] == "ok", result
        return result

    def close(self):
        try:
            if self.proc.poll() is None:
                self.send("shutdown")
                self.proc.wait(timeout=10)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=5)
            self.log.close()
        assert self.proc.returncode == 0


def test_separate_process_dispatch_duplicates_reconnect_shutdown(tmp_path):
    root = synthetic_profiles(tmp_path)
    fleet, overlap = {}, None
    try:
        for p in PROFILES:
            fleet[p] = ProcessGateway(root, p)
        assert len({g.identity["pid"] for g in fleet.values()}) == len(PROFILES)
        assert len({g.identity["home"] for g in fleet.values()}) == len(PROFILES)
        for g in fleet.values():
            g.send("register")
        # Real ingress on every independently configured bot, twice each.
        for _ in range(2):
            for g in fleet.values():
                g.send("ingress", message={"text": PROMPTS["engineering"], "mid": 10})
        overlap = ProcessGateway(root, "default", "-overlap")
        # Two same-owner processes race the real claim/classify boundary.
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda g: g.send("tick"), [fleet["default"], overlap]))
        fleet["engineering"].send("reconnect")
        for _ in range(2):
            for g in fleet.values():
                g.send("tick")
        snapshots = {p: g.send("snapshot") for p, g in fleet.items()}
        other = overlap.send("snapshot")
        (tmp_path / "first-process-evidence.json").write_text(json.dumps({"snapshots": snapshots, "overlap": other}, indent=2))
        row = snapshots["default"]["rows"][0]
        assert row["state"] == "delivered" and row["target_profile"] == "engineering"
        assert row["receipt"]["bot_id"] == "904"
        assert sum(len(s["sent"]) for s in [*snapshots.values(), other]) == 1
        assert all(s["native"] == 0 for s in snapshots.values())
        calls = [c for s in [*snapshots.values(), other] for c in s["calls"]]
        assert sum(c.get("response_format", {}).get("json_schema", {}).get("name") == "team_owner" for c in calls) == 1
        assert [c["model"] for c in calls if c.get("tools")] == ["test-engineering"]
        assert all(s["rows"][0]["receipt"] == row["receipt"] for s in snapshots.values())
        # Actual watcher startup/exit must revoke readiness, not leave a stale member.
        fleet["design"].send("watch")
        fleet["design"].close()
        assert "design" not in fleet["default"].send("snapshot")["members"]
        fleet["design"] = ProcessGateway(root, "design", "-restart")
        fleet["design"].send("register")
        for g in fleet.values():
            g.send("ingress", message={"text": PROMPTS["design"], "mid": 20, "chat": -200, "thread": 8})
        for _ in range(3):
            for g in fleet.values():
                g.send("tick")
        final = {p: g.send("snapshot") for p, g in fleet.items()}
        assert final["design"]["rows"][-1]["state"] == "delivered"
        assert final["design"]["sent"][0]["chat_id"] == -200
        assert final["design"]["sent"][0]["message_thread_id"] == 8
        design_requests = [c for c in final["design"]["calls"] if c.get("tools")]
        assert len(design_requests) == 1
        assert PROMPTS["engineering"] not in json.dumps(design_requests)
        (tmp_path / "process-evidence.json").write_text(json.dumps({"snapshots": final, "overlap": other}, indent=2))
    finally:
        for g in [*fleet.values(), *([overlap] if overlap else [])]:
            g.close()
