"""Supabase/pgvector memory plugin for Hermes.

Uses Dennis's Second Brain Supabase memory schema directly:
- memories
- memory_chunks
- match_memory_chunks(query_embedding vector(1536), ...)

Config via environment variables:
  SUPABASE_URL or NEXT_PUBLIC_SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY
  OPENAI_API_KEY
  SUPABASE_MEMORY_USER_ID (optional, default: dennis)

Optional JSON overrides live in $HERMES_HOME/supabase_memory.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIMENSIONS = 1536
_DEFAULT_USER_ID = "dennis"
_DEFAULT_MATCH_COUNT = 8
_DEFAULT_MIN_SIMILARITY = 0.25
_DEFAULT_AUTO_MATCH_COUNT = 5
_DEFAULT_AUTO_MIN_SIMILARITY = 0.45
_DEFAULT_AUTO_INJECT_MIN_TOP_SIMILARITY = 0.45
_DEFAULT_TIMEOUT = 10.0
_DEFAULT_RECALL_TIERS = ("fast",)
_DEFAULT_WRITE_TIER = "fast"
_MAX_STORE_CHARS = 50_000
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120
_SOURCE_TYPES = {
    "hermes_injected",
    "hermes_detailed",
    "wiki",
    "manual",
    "session_summary",
    "skill",
    "project",
    "prospect",
}
_VALID_RECALL_TIERS = {"fast", "deep", "raw"}
_TRIVIAL_RE = re.compile(r"^(ok|okay|thanks|thank you|got it|sure|yes|no|yep|nope|k|ty|thx|np)\.?$", re.I)
_CONTEXT_RE = re.compile(r"<supabase-memory-context>[\s\S]*?</supabase-memory-context>\s*", re.I)


def _default_config() -> dict:
    return {
        "supabase_url": os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL", ""),
        "service_role_key": os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
        "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
        "user_id": os.environ.get("SUPABASE_MEMORY_USER_ID", _DEFAULT_USER_ID),
        "match_count": _DEFAULT_MATCH_COUNT,
        "min_similarity": _DEFAULT_MIN_SIMILARITY,
        "auto_match_count": _DEFAULT_AUTO_MATCH_COUNT,
        "auto_min_similarity": _DEFAULT_AUTO_MIN_SIMILARITY,
        "auto_inject_min_top_similarity": _DEFAULT_AUTO_INJECT_MIN_TOP_SIMILARITY,
        "recall_tiers": list(_DEFAULT_RECALL_TIERS),
        "write_tier": _DEFAULT_WRITE_TIER,
        "api_timeout": _DEFAULT_TIMEOUT,
        "auto_recall": True,
        "mirror_builtin_writes": True,
    }


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _normalize_recall_tiers(value: Any, default: Sequence[str] = _DEFAULT_RECALL_TIERS) -> list[str]:
    if isinstance(value, str):
        candidates = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        candidates = [str(part).strip() for part in value]
    else:
        candidates = list(default)
    normalized: list[str] = []
    for tier in candidates:
        if tier in _VALID_RECALL_TIERS and tier not in normalized:
            normalized.append(tier)
    return normalized or list(default)


def _clamp_int(value: Any, *, default: int, low: int = 1, high: int = 50) -> int:
    try:
        return max(low, min(high, int(value)))
    except Exception:
        return default


def _clamp_float(value: Any, *, default: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return default


def _load_config(hermes_home: Optional[str] = None) -> dict:
    config = _default_config()
    if hermes_home:
        cfg_path = Path(hermes_home) / "supabase_memory.json"
        if cfg_path.exists():
            try:
                raw = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    config.update({k: v for k, v in raw.items() if v is not None and v != ""})
            except Exception:
                logger.debug("Failed to parse %s", cfg_path, exc_info=True)
    config["match_count"] = _clamp_int(config.get("match_count"), default=_DEFAULT_MATCH_COUNT, high=50)
    config["min_similarity"] = _clamp_float(config.get("min_similarity"), default=_DEFAULT_MIN_SIMILARITY)
    config["auto_match_count"] = _clamp_int(config.get("auto_match_count"), default=_DEFAULT_AUTO_MATCH_COUNT, high=20)
    config["auto_min_similarity"] = _clamp_float(config.get("auto_min_similarity"), default=_DEFAULT_AUTO_MIN_SIMILARITY)
    config["auto_inject_min_top_similarity"] = _clamp_float(
        config.get("auto_inject_min_top_similarity"),
        default=_DEFAULT_AUTO_INJECT_MIN_TOP_SIMILARITY,
    )
    config["recall_tiers"] = _normalize_recall_tiers(config.get("recall_tiers"))
    write_tier = str(config.get("write_tier") or _DEFAULT_WRITE_TIER).strip()
    config["write_tier"] = write_tier if write_tier in _VALID_RECALL_TIERS else _DEFAULT_WRITE_TIER
    config["api_timeout"] = _clamp_float(config.get("api_timeout"), default=_DEFAULT_TIMEOUT, low=1.0, high=30.0)
    config["auto_recall"] = _as_bool(config.get("auto_recall"), True)
    config["mirror_builtin_writes"] = _as_bool(config.get("mirror_builtin_writes"), True)
    config["user_id"] = str(config.get("user_id") or _DEFAULT_USER_ID)
    return config


def _save_config(values: dict, hermes_home: str) -> None:
    cfg_path = Path(hermes_home) / "supabase_memory.json"
    existing = {}
    if cfg_path.exists():
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing = raw
        except Exception:
            existing = {}
    existing.update(values)
    cfg_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_text(text: str) -> str:
    return _CONTEXT_RE.sub("", text or "").strip()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _title_from_content(content: str, explicit: str = "") -> Optional[str]:
    title = (explicit or "").strip()
    if title:
        return title[:140]
    for line in content.splitlines():
        stripped = line.strip("# \t")
        if stripped:
            return stripped[:140]
    return None


def _legacy_category(source_type: str, category: str = "") -> str:
    if category:
        return category
    if source_type == "manual":
        return "Manual Memory"
    if source_type == "wiki":
        return "Wiki"
    if source_type == "hermes_detailed":
        return "Hermes Detailed Memory"
    if source_type == "hermes_injected":
        return "Hermes Injected Memory"
    return "Semantic Memory"


def _chunk_text(content: str, *, target_words: int = 800, overlap_words: int = 80) -> list[dict]:
    words = content.split()
    if not words:
        return []
    stride = max(1, target_words - min(overlap_words, target_words - 1))
    chunks = []
    start = 0
    index = 0
    while start < len(words):
        part = " ".join(words[start:start + target_words]).strip()
        if part:
            chunks.append({
                "chunk_text": part,
                "chunk_index": index,
                "chunk_hash": _hash_text(part),
                "token_count": len(part.split()),
                "metadata": {},
            })
            index += 1
        if start + target_words >= len(words):
            break
        start += stride
    return chunks


def _format_results(
    results: list[dict],
    *,
    max_results: int,
    min_similarity: float = 0.0,
    inject_min_top_similarity: Optional[float] = None,
    recall_tiers: Optional[Sequence[str]] = None,
) -> str:
    if not results:
        return ""
    filtered = []
    top_similarity: Optional[float] = None
    for row in results:
        sim = row.get("similarity")
        numeric_sim = float(sim) if isinstance(sim, (float, int)) else None
        if numeric_sim is not None:
            top_similarity = numeric_sim if top_similarity is None else max(top_similarity, numeric_sim)
            if numeric_sim < min_similarity:
                continue
        filtered.append(row)
    if inject_min_top_similarity is not None and (top_similarity is None or top_similarity < inject_min_top_similarity):
        return ""

    lines = []
    for row in filtered[:max_results]:
        text = str(row.get("chunk_text") or "").strip()
        if not text:
            continue
        source = row.get("source_path") or row.get("source_type") or "memory"
        tier = row.get("recall_tier")
        tier_label = f"/{tier}" if isinstance(tier, str) and tier else ""
        sim = row.get("similarity")
        score = f" ({float(sim):.2f})" if isinstance(sim, (float, int)) else ""
        lines.append(f"- [{source}{tier_label}{score}] {text[:700]}")
    if not lines:
        return ""
    tiers = ",".join(recall_tiers or []) or "all"
    top = f"{top_similarity:.2f}" if top_similarity is not None else "n/a"
    diagnostics = f"<!-- supabase-memory: tiers={tiers} count={len(lines)}/{max_results} top={top} thr={min_similarity:.2f} -->"
    return "<supabase-memory-context>\n## Supabase Memory Recall\n" + "\n".join(lines + [diagnostics]) + "\n</supabase-memory-context>"


class _SupabaseMemoryClient:
    def __init__(self, *, supabase_url: str, service_role_key: str, openai_api_key: str, timeout: float = _DEFAULT_TIMEOUT):
        self.supabase_url = supabase_url.rstrip("/")
        self.service_role_key = service_role_key
        self.openai_api_key = openai_api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        return {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
            "Content-Type": "application/json",
        }

    def _request_json(self, method: str, path: str, payload: Any = None, *, prefer: str = "") -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = self._headers()
        if prefer:
            headers["Prefer"] = prefer
        req = urllib.request.Request(
            f"{self.supabase_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
            if not raw:
                return None
            return json.loads(raw)

    def _openai_embeddings(self, texts: list[str]) -> list[list[float]]:
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps({
                "model": _EMBEDDING_MODEL,
                "input": texts,
                "dimensions": _EMBEDDING_DIMENSIONS,
            }).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        vectors = [item.get("embedding") for item in data.get("data", [])]
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding response shape mismatch")
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != _EMBEDDING_DIMENSIONS:
                raise RuntimeError(f"Embedding must be {_EMBEDDING_DIMENSIONS}-dimensional")
        return vectors

    def search(
        self,
        query: str,
        *,
        user_id: str,
        source_types: Optional[list[str]] = None,
        recall_tiers: Optional[Sequence[str]] = None,
        limit: int = _DEFAULT_MATCH_COUNT,
        min_similarity: float = _DEFAULT_MIN_SIMILARITY,
    ) -> list[dict]:
        [embedding] = self._openai_embeddings([query])
        payload = {
            "query_embedding": embedding,
            "match_count": max(1, min(50, int(limit))),
            "p_user_id": user_id,
            "source_types": source_types or None,
            "min_similarity": min_similarity,
            "include_archived": False,
        }
        if recall_tiers is not None:
            payload["recall_tiers"] = list(recall_tiers) or None
        return self._request_json("POST", "/rest/v1/rpc/match_memory_chunks", payload) or []

    def _find_existing(self, *, user_id: str, source_type: str, source_id: str) -> Optional[dict]:
        qs = urllib.parse.urlencode({
            "select": "id,content_hash",
            "user_id": f"eq.{user_id}",
            "source_type": f"eq.{source_type}",
            "source_id": f"eq.{source_id}",
            "source_path": "is.null",
            "limit": "1",
        })
        rows = self._request_json("GET", f"/rest/v1/memories?{qs}") or []
        return rows[0] if rows else None

    def store_manual(
        self,
        content: str,
        *,
        user_id: str,
        category: str = "Manual Memory",
        title: str = "",
        source_id: str = "",
        source_type: str = "manual",
        recall_tier: str = _DEFAULT_WRITE_TIER,
        metadata: Optional[dict] = None,
    ) -> dict:
        content = content.strip()
        if not content:
            raise ValueError("content is required")
        if len(content) > _MAX_STORE_CHARS:
            raise ValueError("content is too long")
        if source_type not in _SOURCE_TYPES:
            raise ValueError(f"invalid source_type: {source_type}")
        if recall_tier not in _VALID_RECALL_TIERS:
            raise ValueError(f"invalid recall_tier: {recall_tier}")
        source_id = source_id or f"hermes:{uuid.uuid4()}"
        metadata = dict(metadata or {})
        metadata.setdefault("category", category)
        metadata.setdefault("writer", "hermes-supabase-memory")
        content_hash = _hash_text(content)
        title_value = _title_from_content(content, title)
        existing = self._find_existing(user_id=user_id, source_type=source_type, source_id=source_id)
        if existing:
            memory_id = existing["id"]
            self._request_json("PATCH", f"/rest/v1/memories?id=eq.{urllib.parse.quote(memory_id)}", {
                "title": title_value,
                "category": _legacy_category(source_type, category),
                "source": source_id,
                "tags": ["semantic-memory", source_type],
                "content": content,
                "source_type": source_type,
                "source_path": None,
                "source_id": source_id,
                "content_hash": content_hash,
                "metadata": metadata,
                "recall_tier": recall_tier,
                "archived_at": None,
            })
        else:
            rows = self._request_json("POST", "/rest/v1/memories?select=id", {
                "user_id": user_id,
                "title": title_value,
                "category": _legacy_category(source_type, category),
                "source": source_id,
                "tags": ["semantic-memory", source_type],
                "content": content,
                "source_type": source_type,
                "source_path": None,
                "source_id": source_id,
                "content_hash": content_hash,
                "metadata": metadata,
                "recall_tier": recall_tier,
            }, prefer="return=representation") or []
            if not rows:
                raise RuntimeError("Supabase insert returned no id")
            memory_id = rows[0]["id"]

        chunks = _chunk_text(content)
        self._request_json("DELETE", f"/rest/v1/memory_chunks?memory_id=eq.{urllib.parse.quote(memory_id)}")
        if chunks:
            vectors = self._openai_embeddings([c["chunk_text"] for c in chunks])
            rows = []
            for chunk, embedding in zip(chunks, vectors):
                rows.append({
                    "memory_id": memory_id,
                    "user_id": user_id,
                    "source_type": source_type,
                    "source_path": None,
                    "chunk_text": chunk["chunk_text"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_hash": chunk["chunk_hash"],
                    "token_count": chunk["token_count"],
                    "embedding": embedding,
                    "metadata": chunk["metadata"],
                    "recall_tier": recall_tier,
                })
            self._request_json("POST", "/rest/v1/memory_chunks", rows)
        return {"id": memory_id, "source_id": source_id, "chunks": len(chunks), "content_hash": content_hash}


SEARCH_SCHEMA = {
    "name": "supabase_memory_search",
    "description": "Search Dennis's Supabase/pgvector long-term memory by meaning. Defaults to fast-tier durable memory; opt into deep/raw tiers only for explicit deeper recall.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to recall."},
            "source_types": {"type": "array", "items": {"type": "string", "enum": sorted(_SOURCE_TYPES)}, "description": "Optional source filters."},
            "recall_tiers": {"type": "array", "items": {"type": "string", "enum": sorted(_VALID_RECALL_TIERS)}, "description": "Tier filter. Default ['fast']; use ['deep'] or ['fast','deep'] for richer recall, ['raw'] for unprocessed source chunks."},
            "limit": {"type": "integer", "description": "Max results, default 8, max 50."},
            "min_similarity": {"type": "number", "description": "0-1 similarity floor, default 0.25."},
        },
        "required": ["query"],
    },
}

STORE_SCHEMA = {
    "name": "supabase_memory_store",
    "description": "Store one explicit durable memory into Dennis's Supabase memory backend. Use only for stable facts, preferences, decisions, corrections, or explicit remember-this requests.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Durable fact or note to store."},
            "category": {"type": "string", "description": "Short category, e.g. Preference, Decision, Project."},
            "title": {"type": "string", "description": "Optional title."},
            "source_id": {"type": "string", "description": "Optional stable idempotency key for this memory."},
        },
        "required": ["content"],
    },
}


class SupabaseMemoryProvider(MemoryProvider):
    def __init__(self):
        self._config = _default_config()
        self._client: Optional[_SupabaseMemoryClient] = None
        self._client_lock = threading.Lock()
        self._prefetch_thread = None
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._write_thread = None
        self._session_id = ""
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "supabase"

    def is_available(self) -> bool:
        cfg = _load_config()
        return bool(cfg.get("supabase_url") and cfg.get("service_role_key") and cfg.get("openai_api_key"))

    def save_config(self, values, hermes_home):
        _save_config(values, hermes_home)

    def get_config_schema(self):
        return [
            {"key": "supabase_url", "description": "Supabase project URL", "secret": True, "required": True, "env_var": "SUPABASE_URL"},
            {"key": "service_role_key", "description": "Supabase service role key", "secret": True, "required": True, "env_var": "SUPABASE_SERVICE_ROLE_KEY"},
            {"key": "openai_api_key", "description": "OpenAI API key for text-embedding-3-small", "secret": True, "required": True, "env_var": "OPENAI_API_KEY"},
            {"key": "user_id", "description": "Memory user id", "default": _DEFAULT_USER_ID},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home", "")
        self._config = _load_config(hermes_home)
        self._session_id = session_id
        with self._client_lock:
            self._client = None

    def _get_client(self) -> _SupabaseMemoryClient:
        with self._client_lock:
            if self._client is None:
                self._client = _SupabaseMemoryClient(
                    supabase_url=self._config.get("supabase_url", ""),
                    service_role_key=self._config.get("service_role_key", ""),
                    openai_api_key=self._config.get("openai_api_key", ""),
                    timeout=float(self._config.get("api_timeout", _DEFAULT_TIMEOUT)),
                )
            return self._client

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning("Supabase memory circuit breaker tripped; pausing API calls")

    def system_prompt_block(self) -> str:
        tiers = ",".join(self._config.get("recall_tiers") or _DEFAULT_RECALL_TIERS)
        auto_similarity = float(self._config.get("auto_min_similarity", _DEFAULT_AUTO_MIN_SIMILARITY))
        auto_count = int(self._config.get("auto_match_count", _DEFAULT_AUTO_MATCH_COUNT))
        return (
            "# Supabase Memory\n"
            f"Active for user_id={self._config.get('user_id', _DEFAULT_USER_ID)}. "
            f"Auto-recall is conservative: tiers={tiers}, similarity >= {auto_similarity:.2f}, top {auto_count}, and silent when nothing relevant. "
            "Use supabase_memory_search for semantic recall; pass recall_tiers ['deep'] or ['raw'] only when explicit deeper search is needed. "
            "Use supabase_memory_store only for durable explicit memories."
        )

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._config.get("auto_recall", True) or self._is_breaker_open():
            return
        query = _clean_text(query)
        if not query or _TRIVIAL_RE.match(query):
            return

        def _run():
            try:
                auto_match_count = int(self._config.get("auto_match_count", _DEFAULT_AUTO_MATCH_COUNT))
                auto_min_similarity = float(self._config.get("auto_min_similarity", _DEFAULT_AUTO_MIN_SIMILARITY))
                recall_tiers = list(self._config.get("recall_tiers") or _DEFAULT_RECALL_TIERS)
                results = self._get_client().search(
                    query,
                    user_id=self._config.get("user_id", _DEFAULT_USER_ID),
                    recall_tiers=recall_tiers,
                    limit=auto_match_count,
                    min_similarity=auto_min_similarity,
                )
                formatted = _format_results(
                    results,
                    max_results=auto_match_count,
                    min_similarity=auto_min_similarity,
                    inject_min_top_similarity=float(self._config.get("auto_inject_min_top_similarity", _DEFAULT_AUTO_INJECT_MIN_TOP_SIMILARITY)),
                    recall_tiers=recall_tiers,
                )
                with self._prefetch_lock:
                    self._prefetch_result = formatted
                self._record_success()
            except Exception as exc:
                self._record_failure()
                logger.debug("Supabase memory prefetch failed: %s", exc)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="supabase-memory-prefetch")
        self._prefetch_thread.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        return result or ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, STORE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({"error": "Supabase memory temporarily unavailable after repeated failures."})
        try:
            if tool_name == "supabase_memory_search":
                query = _clean_text(str(args.get("query") or ""))
                if not query:
                    return tool_error("Missing required parameter: query")
                source_types = args.get("source_types")
                if source_types is not None:
                    if not isinstance(source_types, list) or any(s not in _SOURCE_TYPES for s in source_types):
                        return tool_error("source_types must be valid memory source types")
                recall_tiers = args.get("recall_tiers")
                if recall_tiers is not None:
                    if isinstance(recall_tiers, str):
                        recall_tier_values = [part.strip() for part in recall_tiers.split(",") if part.strip()]
                    elif isinstance(recall_tiers, list):
                        recall_tier_values = [str(part).strip() for part in recall_tiers]
                    else:
                        return tool_error("recall_tiers must be a list of fast, deep, or raw")
                    if not recall_tier_values or any(tier not in _VALID_RECALL_TIERS for tier in recall_tier_values):
                        return tool_error("recall_tiers must contain only fast, deep, or raw")
                    recall_tiers = _normalize_recall_tiers(recall_tier_values, default=())
                else:
                    recall_tiers = list(_DEFAULT_RECALL_TIERS)
                limit = _clamp_int(args.get("limit", self._config.get("match_count", _DEFAULT_MATCH_COUNT)), default=int(self._config.get("match_count", _DEFAULT_MATCH_COUNT)), high=50)
                min_similarity = _clamp_float(args.get("min_similarity", self._config.get("min_similarity", _DEFAULT_MIN_SIMILARITY)), default=float(self._config.get("min_similarity", _DEFAULT_MIN_SIMILARITY)))
                results = self._get_client().search(
                    query,
                    user_id=self._config.get("user_id", _DEFAULT_USER_ID),
                    source_types=source_types,
                    recall_tiers=recall_tiers,
                    limit=limit,
                    min_similarity=min_similarity,
                )
                self._record_success()
                return json.dumps({"results": results, "count": len(results)})

            if tool_name == "supabase_memory_store":
                content = _clean_text(str(args.get("content") or ""))
                if not content:
                    return tool_error("Missing required parameter: content")
                result = self._get_client().store_manual(
                    content,
                    user_id=self._config.get("user_id", _DEFAULT_USER_ID),
                    category=str(args.get("category") or "Manual Memory"),
                    title=str(args.get("title") or ""),
                    source_id=str(args.get("source_id") or ""),
                    recall_tier=str(self._config.get("write_tier", _DEFAULT_WRITE_TIER)),
                    metadata={"writer": "hermes-tool", "session_id": self._session_id},
                )
                self._record_success()
                return json.dumps({"stored": True, **result})
        except Exception as exc:
            self._record_failure()
            return tool_error(f"Supabase memory error: {exc}")
        return tool_error(f"Unknown tool: {tool_name}")

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[dict] = None) -> None:
        if not self._config.get("mirror_builtin_writes", True) or self._is_breaker_open():
            return
        if action not in {"add", "replace"}:
            return
        cleaned = _clean_text(content)
        if not cleaned or _TRIVIAL_RE.match(cleaned):
            return
        source_id = f"builtin:{target}:{_hash_text(cleaned)[:24]}"

        def _run():
            try:
                meta = {"writer": "hermes-built-in-memory", "target": target, "action": action}
                if isinstance(metadata, dict):
                    meta.update({k: v for k, v in metadata.items() if k not in {"api_key", "token", "secret"}})
                self._get_client().store_manual(
                    cleaned,
                    user_id=self._config.get("user_id", _DEFAULT_USER_ID),
                    category="Hermes Built-in Memory",
                    source_id=source_id,
                    source_type="hermes_injected",
                    recall_tier=str(self._config.get("write_tier", _DEFAULT_WRITE_TIER)),
                    metadata=meta,
                )
                self._record_success()
            except Exception as exc:
                self._record_failure()
                logger.warning("Supabase memory mirror write failed: %s", exc)

        if self._write_thread and self._write_thread.is_alive():
            self._write_thread.join(timeout=5.0)
        self._write_thread = threading.Thread(target=_run, daemon=True, name="supabase-memory-write")
        self._write_thread.start()

    def shutdown(self) -> None:
        for thread in (self._prefetch_thread, self._write_thread):
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
        with self._client_lock:
            self._client = None


def register(ctx) -> None:
    ctx.register_memory_provider(SupabaseMemoryProvider())
