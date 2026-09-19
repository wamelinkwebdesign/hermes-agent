"""Exercise the existing fixed-profile CLI consultation, not an auxiliary clone."""
import json
from pathlib import Path
import subprocess
import sys
import shlex

import pytest
import yaml

from hermes_state import SessionDB
from tests.gateway.test_team_dispatch_processes import REPO, process_env, synthetic_profiles
from tests.gateway.test_team_dispatch_runtime import fleet, model, tick, ingress, settle, rows, message, PROMPTS


def seed_private_history(home):
    db = SessionDB(db_path=home / "state.db")
    db.create_session("private-canary", "cli")
    db.set_session_title("private-canary", "Private unrelated work")
    db.append_message("private-canary", "user", "PRIVATE_HISTORY_CANARY_DO_NOT_EXPORT")
    db.append_message("private-canary", "assistant", "Private answer")
    db.close()


@pytest.mark.parametrize("mode", ["read", "approval"])
def test_existing_cli_peer_profile_tools_and_privacy(tmp_path, mode):
    root = synthetic_profiles(tmp_path)
    target = root / "profiles" / "product"
    seed_private_history(target)
    (tmp_path / "public-brief.txt").write_text("PUBLIC_ONLY_BRIEF")
    marker = tmp_path / "approval-canary"
    marker.mkdir()
    (marker / "keep.txt").write_text("must survive denied approval")
    capture = tmp_path / "peer-http.json"
    argv = [sys.executable, str(REPO / "tests/gateway/_team_dispatch_peer_helper.py"), str(capture), mode,
            "-p", "product", "chat", "--in", str(tmp_path), "-c", "Bot Chat", "--create-if-missing",
            "-Q", "--query-file", str(tmp_path / "public-brief.txt")]
    # Fixed installed-profile selector, no shell and no user text in argv.
    try:
        result = subprocess.run(argv, cwd=tmp_path, env=process_env(root, "engineering"),
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired as exc:
        (tmp_path / "peer-timeout-stderr.log").write_bytes(exc.stderr or b"")
        raise
    (tmp_path / "peer-cli-output.txt").write_text(result.stdout + "\nSTDERR\n" + result.stderr)
    assert result.returncode == 0, result.stderr + result.stdout
    receipt = json.loads(capture.read_text())
    assert Path(receipt["profile_home"]) == target
    turns = [c for c in receipt["calls"] if c.get("tools")]
    assert turns and all(c["model"] == "test-product" for c in turns)
    assert "PRIVATE_HISTORY_CANARY" not in json.dumps(receipt)
    assert "PRIVATE_HISTORY_CANARY" not in result.stdout
    assert "PUBLIC_PEER_RESULT" in result.stdout
    tools = [m for c in turns for m in c["messages"] if m["role"] == "tool"]
    assert tools, "the real CLI must execute a tool, not return an auxiliary imitation"
    if mode == "read":
        assert "PUBLIC_ONLY_BRIEF" in json.dumps(tools)
    else:
        assert (marker / "keep.txt").is_file(), "approval must not be silently granted"
        assert any(word in json.dumps(tools).lower() for word in ["denied", "blocked", "not approved"])


@pytest.mark.asyncio
async def test_public_owner_consults_real_cli_peer_and_alone_publishes(fleet, model, tmp_path):
    root = tmp_path / ".hermes"
    root.symlink_to(fleet["default"][2], target_is_directory=True)
    target = root / "profiles" / "product"
    cfg = yaml.safe_load((target / "config.yaml").read_text())
    cfg["model"]["context_length"] = 128000
    (target / "config.yaml").write_text(yaml.safe_dump(cfg))
    seed_private_history(target)
    (tmp_path / "public-brief.txt").write_text("PUBLIC_ONLY_BRIEF")
    capture = tmp_path / "public-peer-http.json"
    argv = [sys.executable, str(REPO / "tests/gateway/_team_dispatch_peer_helper.py"), str(capture), "read",
            "-p", "product", "chat", "--in", str(tmp_path), "-c", "Bot Chat", "--create-if-missing",
            "-Q", "--query-file", str(tmp_path / "public-brief.txt")]
    env = process_env(root, "engineering")
    command = shlex.join(["env", "-i", *[k + "=" + v for k, v in env.items()], *argv])
    model.tool = ("terminal", {"command": command, "timeout": 20})
    model.final = "Engineering public synthesis from the internal peer result."
    await tick(fleet)
    await ingress(fleet, message(PROMPTS["engineering"]))
    await settle(fleet)
    assert rows(fleet)[0]["state"] == "delivered"
    tool_results = [m for c in model for m in c.get("messages", []) if m["role"] == "tool"]
    assert "PUBLIC_PEER_RESULT" in json.dumps(tool_results)
    receipt = json.loads(capture.read_text())
    assert Path(receipt["profile_home"]).resolve() == target.resolve()
    assert all(c["model"] == "test-product" for c in receipt["calls"] if c.get("tools"))
    assert "PUBLIC_ONLY_BRIEF" in json.dumps([m for c in receipt["calls"] for m in c.get("messages", []) if m["role"] == "tool"])
    assert "PRIVATE_HISTORY_CANARY" not in json.dumps(model)
    assert len(fleet["engineering"][1]._bot.sent) == 1
    assert all(not ad._bot.sent for p, (_, ad, _) in fleet.items() if p != "engineering")
