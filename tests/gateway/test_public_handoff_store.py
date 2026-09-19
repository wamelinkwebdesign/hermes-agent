"""Durable public request contract and process-death fences."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import time

import pytest

from gateway.public_handoff_store import PublicHandoffStore, PublicRequest, request_identity


def request():
    rid = request_identity("-100", "7", "10")
    now = time.time()
    return PublicRequest(1, rid, "default", "engineering", "telegram", "-100", "7", "10", "111",
                         "Engineering: public task", now, now + 120, rid)


@pytest.mark.parametrize("changes", [
    {"version": 2}, {"source_profile": "revenue"}, {"target_profile": "finance-ops"},
    {"platform": "discord"}, {"chat_id": "../secret"}, {"thread_id": " 7"},
    {"sender_id": "111\n"}, {"message_id": "-10"}, {"request_id": "forged"},
    {"root_request_id": "../../private"}, {"brief": "x" * 6001}, {"brief": ""},
    {"expires_at": float("nan")}, {"created_at": float("inf")},
])
def test_invalid_host_contract_rejected_before_persistence(tmp_path, changes):
    store = PublicHandoffStore(tmp_path / "requests.sqlite")
    with pytest.raises(ValueError):
        store.enqueue(replace(request(), **changes))
    assert not store.path.exists()


@pytest.mark.parametrize("after_send_boundary", [False, True])
def test_hard_process_exit_preserves_claim_and_send_boundary(tmp_path, monkeypatch, after_send_boundary):
    store = PublicHandoffStore(tmp_path / "requests.sqlite")
    req = request()
    store.enqueue(req)
    # This is a storage crash probe, not an ingress fixture. The integration suite
    # separately enters every dispatch through authenticated Telegram ingress.
    code = """
import os, sys
from pathlib import Path
from gateway.public_handoff_store import PublicHandoffStore
s = PublicHandoffStore(Path(sys.argv[1]))
rid = sys.argv[2]
assert s.claim(rid)
assert s.bind_session(rid, rid, 'group-public-key', 'existing-group-session')
if sys.argv[3] == 'send':
    assert s.begin_send(rid, 'public answer')
os._exit(19)
"""
    child = subprocess.run([sys.executable, "-c", code, str(store.path), req.request_id,
                            "send" if after_send_boundary else "before"], capture_output=True, text=True, timeout=20)
    assert child.returncode == 19, child.stderr
    recovered = PublicHandoffStore(store.path)
    assert recovered.get(req.request_id)["state"] == "running"
    assert not recovered.claim(req.request_id)
    monkeypatch.setattr(time, "time", lambda: req.expires_at + 1)
    recovered.expire()
    row = recovered.get(req.request_id)
    assert row["state"] == ("delivery_uncertain" if after_send_boundary else "failed_definitive")
    assert not recovered.begin_send(req.request_id, "stale result")
    assert not recovered.delivered(req.request_id, {})
    assert recovered.claim_fallback(req.request_id) is (not after_send_boundary)
    assert not recovered.claim_fallback(req.request_id)


@pytest.mark.parametrize("boundary", ["timeout", "hard_exit", "delivered"])
def test_fallback_quarantine_fences_atomic_send_claims(tmp_path, boundary):
    store = PublicHandoffStore(tmp_path / "requests.sqlite")
    root = request()
    store.enqueue(root)
    store.fail(root.request_id, "no_receiver")
    pending = replace(root, request_id=request_identity("-100", "7", "11"), message_id="11",
                      source_profile="engineering", reply_to_message_id="10", reply_to_sender_id="111")
    store.enqueue(pending)
    failed = replace(pending, request_id=request_identity("-100", "7", "12"), message_id="12")
    store.enqueue(failed)
    store.fail(failed.request_id, "no_receiver")
    code = """
import os, sys
from pathlib import Path
from gateway.public_handoff_store import PublicHandoffStore
assert PublicHandoffStore(Path(sys.argv[1])).claim_fallback(sys.argv[2])
os._exit(19)
"""
    if boundary == "hard_exit":
        child = subprocess.run([sys.executable, "-c", code, str(store.path), root.request_id],
                               capture_output=True, text=True, timeout=20)
        assert child.returncode == 19, child.stderr
    else:
        assert store.claim_fallback(root.request_id)
        store.finish_fallback(root.request_id, {"state": "delivered" if boundary == "delivered" else "delivery_uncertain"})
    recovered = PublicHandoffStore(store.path)
    allowed = boundary == "delivered"
    assert recovered.conversation_uncertain(root.request_id) is (not allowed)
    assert recovered.claim(pending.request_id) is allowed
    if allowed:
        assert recovered.bind_session(pending.request_id, root.request_id, "public-key", "group")
        assert recovered.begin_send(pending.request_id, "answer")
        # Fallback cannot cross a sibling's active send/inference boundary.
        assert not recovered.claim_fallback(failed.request_id)
        recovered.delivered(pending.request_id, {"message_id": "1001"})
    assert recovered.claim_fallback(failed.request_id) is allowed


def test_request_ttl_bound_accepts_producer_deadline_and_rejects_overrun(tmp_path):
    """The wire cap must admit the team producer's deadline and reject anything past the cap.

    Regression for the silent-ingress outage: a producer deadline raised above the
    validation cap made every request fail construction, and the adapter swallowed
    the ValueError with only a type-name warning.
    """
    from gateway.public_handoff_store import MAX_REQUEST_TTL_SECONDS
    from gateway.run_team_dispatch import TEAM_DEADLINE_SECONDS

    assert 0 < TEAM_DEADLINE_SECONDS <= MAX_REQUEST_TTL_SECONDS
    rid = request_identity("-100", "7", "10")
    now = time.time()
    store = PublicHandoffStore(tmp_path / "requests.sqlite")

    at_deadline = PublicRequest(1, rid, "default", "engineering", "telegram", "-100", "7",
                                "10", "111", "task", now, now + TEAM_DEADLINE_SECONDS, rid)
    store.enqueue(at_deadline)
    assert store.get(rid)["state"] == "accepted"

    with pytest.raises(ValueError):
        PublicRequest(1, rid, "default", "engineering", "telegram", "-100", "7",
                      "10", "111", "task", now, now + MAX_REQUEST_TTL_SECONDS + 1, rid)
