import json

import pytest

from plugins.memory.supabase import (
    SupabaseMemoryProvider,
    _EMBEDDING_DIMENSIONS,
    _format_results,
    _load_config,
)


class FakeSupabaseClient:
    instances = []

    def __init__(self, *, supabase_url, service_role_key, openai_api_key, timeout=10.0):
        self.supabase_url = supabase_url
        self.service_role_key = service_role_key
        self.openai_api_key = openai_api_key
        self.timeout = timeout
        self.search_calls = []
        self.store_calls = []
        self.search_response = [
            {
                "chunk_id": "c1",
                "memory_id": "m1",
                "source_type": "wiki",
                "source_path": "wiki/wamelink/foo.md",
                "recall_tier": "fast",
                "chunk_text": "WAMELINK outreach subject details",
                "similarity": 0.88,
            }
        ]
        FakeSupabaseClient.instances.append(self)

    def search(self, query, *, user_id, source_types=None, recall_tiers=None, limit=8, min_similarity=0.25):
        self.search_calls.append({
            "query": query,
            "user_id": user_id,
            "source_types": source_types,
            "recall_tiers": recall_tiers,
            "limit": limit,
            "min_similarity": min_similarity,
        })
        return self.search_response

    def store_manual(self, content, *, user_id, category="Manual Memory", title="", source_id="", source_type="manual", recall_tier="fast", metadata=None):
        self.store_calls.append({
            "content": content,
            "user_id": user_id,
            "category": category,
            "title": title,
            "source_id": source_id,
            "source_type": source_type,
            "recall_tier": recall_tier,
            "metadata": metadata,
        })
        return {"id": "mem-1", "source_id": source_id or "hermes:1", "chunks": 1, "content_hash": "hash"}


@pytest.fixture
def provider(monkeypatch, tmp_path):
    FakeSupabaseClient.instances.clear()
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setattr("plugins.memory.supabase._SupabaseMemoryClient", FakeSupabaseClient)
    p = SupabaseMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    return p


def test_is_available_requires_supabase_and_openai_env(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("NEXT_PUBLIC_SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert SupabaseMemoryProvider().is_available() is False

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert SupabaseMemoryProvider().is_available() is True


def test_load_config_uses_file_overrides_without_logging_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPABASE_URL", "https://env.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    (tmp_path / "supabase_memory.json").write_text(json.dumps({"user_id": "dennis", "match_count": 12}), encoding="utf-8")

    cfg = _load_config(str(tmp_path))

    assert cfg["supabase_url"] == "https://env.supabase.co"
    assert cfg["user_id"] == "dennis"
    assert cfg["match_count"] == 12


def test_format_results_wraps_context_and_includes_sources():
    result = _format_results([
        {"source_path": "wiki/foo.md", "chunk_text": "Relevant memory", "similarity": 0.9}
    ], max_results=5)
    assert "<supabase-memory-context>" in result
    assert "wiki/foo.md" in result
    assert "Relevant memory" in result
    assert "supabase-memory: tiers=all" in result


def test_search_tool_routes_to_supabase_rpc_client(provider):
    raw = provider.handle_tool_call("supabase_memory_search", {
        "query": "approval queue",
        "source_types": ["wiki"],
        "limit": 5,
        "min_similarity": 0.3,
    })
    out = json.loads(raw)

    assert out["count"] == 1
    client = FakeSupabaseClient.instances[-1]
    assert client.search_calls == [{
        "query": "approval queue",
        "user_id": "dennis",
        "source_types": ["wiki"],
        "recall_tiers": ["fast"],
        "limit": 5,
        "min_similarity": 0.3,
    }]


def test_search_tool_rejects_invalid_source_type(provider):
    raw = provider.handle_tool_call("supabase_memory_search", {
        "query": "x",
        "source_types": ["nope"],
    })
    assert "error" in json.loads(raw)
    assert FakeSupabaseClient.instances == []


def test_store_tool_writes_manual_memory_scoped_to_dennis(provider):
    raw = provider.handle_tool_call("supabase_memory_store", {
        "content": "Dennis prefers concrete evidence-backed outreach critique",
        "category": "Preference",
        "source_id": "test-source",
    })
    out = json.loads(raw)

    assert out["stored"] is True
    client = FakeSupabaseClient.instances[-1]
    assert client.store_calls[0]["user_id"] == "dennis"
    assert client.store_calls[0]["category"] == "Preference"
    assert client.store_calls[0]["source_id"] == "test-source"
    assert client.store_calls[0]["source_type"] == "manual"
    assert client.store_calls[0]["recall_tier"] == "fast"


def test_queue_prefetch_returns_context_on_next_prefetch(provider):
    provider.queue_prefetch("WAMELINK outreach subject", session_id="session-1")
    result = provider.prefetch("next", session_id="session-1")

    assert "Supabase Memory Recall" in result
    assert "WAMELINK outreach subject details" in result
    assert "tiers=fast" in result
    client = FakeSupabaseClient.instances[-1]
    assert client.search_calls[0] == {
        "query": "WAMELINK outreach subject",
        "user_id": "dennis",
        "source_types": None,
        "recall_tiers": ["fast"],
        "limit": 5,
        "min_similarity": 0.45,
    }


def test_queue_prefetch_stays_silent_when_top_similarity_is_weak(provider):
    client = provider._get_client()
    client.search_response = [{
        "source_type": "wiki",
        "source_path": "wiki/tangent.md",
        "recall_tier": "fast",
        "chunk_text": "Tangential memory",
        "similarity": 0.44,
    }]

    provider.queue_prefetch("something vague", session_id="session-1")
    result = provider.prefetch("next", session_id="session-1")

    assert result == ""


def test_search_tool_accepts_explicit_recall_tiers(provider):
    raw = provider.handle_tool_call("supabase_memory_search", {
        "query": "approval internals",
        "recall_tiers": ["deep", "raw"],
    })
    out = json.loads(raw)

    assert out["count"] == 1
    client = FakeSupabaseClient.instances[-1]
    assert client.search_calls[-1] == {
        "query": "approval internals",
        "user_id": "dennis",
        "source_types": None,
        "recall_tiers": ["deep", "raw"],
        "limit": 8,
        "min_similarity": 0.25,
    }


def test_search_tool_rejects_invalid_recall_tier(provider):
    raw = provider.handle_tool_call("supabase_memory_search", {
        "query": "x",
        "recall_tiers": ["nope"],
    })
    assert "error" in json.loads(raw)
    assert FakeSupabaseClient.instances == []


def test_load_config_clamps_recall_tiers_and_auto_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPABASE_URL", "https://env.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    (tmp_path / "supabase_memory.json").write_text(json.dumps({
        "recall_tiers": ["fast", "bogus", "deep", "fast"],
        "auto_match_count": 99,
        "auto_min_similarity": -1,
        "auto_inject_min_top_similarity": 2,
        "write_tier": "raw",
    }), encoding="utf-8")

    cfg = _load_config(str(tmp_path))

    assert cfg["recall_tiers"] == ["fast", "deep"]
    assert cfg["auto_match_count"] == 20
    assert cfg["auto_min_similarity"] == 0.0
    assert cfg["auto_inject_min_top_similarity"] == 1.0
    assert cfg["write_tier"] == "raw"


def test_on_memory_write_mirrors_builtin_memory_with_stable_source_id(provider):
    provider.on_memory_write("add", "memory", "Dennis wants proactive skill discovery")
    assert provider._write_thread is not None
    provider._write_thread.join(timeout=1)

    client = FakeSupabaseClient.instances[-1]
    call = client.store_calls[0]
    assert call["category"] == "Hermes Built-in Memory"
    assert call["source_type"] == "hermes_injected"
    assert call["recall_tier"] == "fast"
    assert call["source_id"].startswith("builtin:memory:")
    assert call["metadata"]["target"] == "memory"


def test_embedding_dimension_constant_matches_second_brain_contract():
    assert _EMBEDDING_DIMENSIONS == 1536
