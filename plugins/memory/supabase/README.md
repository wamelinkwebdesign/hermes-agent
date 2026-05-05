# Supabase Memory Provider

Hermes memory provider backed by the Second Brain Supabase/pgvector schema.

Required env vars:

- `SUPABASE_URL` or `NEXT_PUBLIC_SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `OPENAI_API_KEY`

Optional:

- `SUPABASE_MEMORY_USER_ID` (default `dennis`)
- `$HERMES_HOME/supabase_memory.json` for non-secret config:

```json
{
  "user_id": "dennis",
  "match_count": 8,
  "min_similarity": 0.25,
  "auto_match_count": 5,
  "auto_min_similarity": 0.45,
  "auto_inject_min_top_similarity": 0.45,
  "recall_tiers": ["fast"],
  "write_tier": "fast",
  "auto_recall": true,
  "mirror_builtin_writes": true
}
```

Behavior:

- Automatic injected recall is conservative: `recall_tiers=["fast"]`, top 5, similarity >= `0.45`, and silent when no fast-tier match clears the threshold.
- Explicit `supabase_memory_search` defaults to fast-tier recall but can opt into `recall_tiers=["deep"]`, `["raw"]`, or combinations when deeper source recall is needed.
- `supabase_memory_store` and mirrored built-in Hermes memory writes set `recall_tier="fast"` by default.

Tools:

- `supabase_memory_search`
- `supabase_memory_store`

This provider is explicit-write focused. It mirrors built-in Hermes memory writes as `hermes_injected` and offers a manual store tool; it does not auto-ingest every conversation turn.
