# Local task-accounting pilot

Standalone local-only observer plugin for the Hermes fork. It is not installed or enabled by this delivery. It adds no tools, middleware, prompts, model calls, credentials, provider calls, network telemetry, or automatic completion.

## Activation gate

Hermes must load `task-accounting` through normal standalone plugin discovery, then this separate setting must be the boolean `true`:

    plugins.entries.task-accounting.settings.enabled: true

The setting defaults to false and is checked on every callback. The dotted path above is not literal YAML. This plugin never calls `set_config`, modifies live configuration, or restarts Hermes. Disabling the setting makes later observations inert, although metadata already in the bounded queue may finish writing.

## Local storage and privacy

Storage is `$HERMES_HOME/task-accounting/attempts.sqlite3` for the active context-local profile. The directory is mode 0700 and the database is mode 0600. Symlink profile ancestry, the storage directory, database, and SQLite sidecars are refused. Existing files must be private, owned regular files with one hard link. This boundary does not defend against a malicious concurrent process running as the same OS user.

The callback projects a fixed scalar schema before enqueueing. It retains:

- SHA-256 hashes of `task_id`, session, turn, request, and auxiliary-task identities;
- exact model/provider values only from small allowlists, otherwise a hash of a bounded string;
- canonical token buckets, retry and call counters, fixed status fields, and fixed booleans;
- finite nonnegative numeric `started_at`, `ended_at`, `api_duration`, and `first_chunk_at` values;
- a local numeric `observed_at` timestamp;
- for subagent links, start/stop markers, an allowlisted child status, and bounded integer `duration_ms`.

Booleans, strings, negative values, NaN, and infinity are rejected for timing fields. Hashes are pseudonyms, not encryption. Raw prompts, responses, errors, summaries, goals, labels, URLs, tool arguments/results, and arbitrary nested fields are never stored or queued.

Callbacks use a bounded 128-entry queue and never wait for database I/O. One context-preserving daemon writer starts lazily. SQLite busy wait is 25 ms, the progress deadline is 50 ms, and page count is capped at 4096. Queue overflow, lock contention, I/O failure, and abrupt process exit can lose observations. Process-local dropped/failed counters are not durable. Reports therefore never claim exhaustive accounting.

## Request and retry semantics

Hermes reuses the same main `api_request_id` through its retry loop. The main pre/error hooks carry `retry_count`, but the terminal success hook does not. The collector stores projected hook observations separately and materializes them deterministically:

- each observed error retry remains a separate failed attempt with unknown usage;
- terminal success keeps its known usage exactly once;
- terminal success is associated with the latest unclosed pre observation only to suppress a duplicate pending row;
- the success row keeps `retry_count: null` and `attempt_precision: terminal_retry_count_unavailable`;
- duplicate and out-of-order pre/error/post delivery produce the same materialized result;
- conflicting duplicate terminal payloads are flagged and excluded from complete totals.

This association does not claim an exact success retry ordinal. `retry_count` describes Hermes hook-level retry observations only. Provider SDK-internal retries remain unknown. A failed attempt with unknown usage makes complete token totals null; its absence is never converted to zero. Known successful usage remains visible in `known_subtotals` and optional `known_priced_subtotal`.

Auxiliary retries also reuse a logical request ID; both pre and post hooks carry `retry_count`, so each `(request ID, retry_count)` pair materializes separately. Main and auxiliary streams remain distinct. Main `provider=moa` rows are retained as `aggregate_excluded`, while separately observed auxiliary physical calls may be counted. Nested child totals and `moa_references` aggregates are ignored. Streamed auxiliary calls return before final usage exists and remain unknown.

Canonical usage buckets are disjoint: `input_tokens` excludes cache, cache read/write are separate, and `output_tokens` includes the informational `reasoning_tokens` subset. Missing or malformed fields stay null. Explicit integer zero stays zero. Hermes normalization can lose original missing-field provenance, which is reported as unknown.

Main `started_at`, `ended_at`, and `api_duration` values describe the cumulative logical request across retries because Hermes shares `api_start_time`; they are not per-attempt latencies and are never summed as such. Auxiliary timings describe one physical auxiliary attempt. Reports label this distinction.

## Observed run identity and operator grouping

The emitted `task_id` is hashed before enqueue and identifies one Hermes run, not an entire user goal across multiple runs. Automatic grouping uses that hash, so concurrent runs in one session do not mix:

    python report.py inventory --home /absolute/profile/home
    python report.py observed-task --home /absolute/profile/home OBSERVED_TASK_HASH

`inventory` returns attempts and links plus `observed_tasks` summaries grouped by hashed run identity, sessions, turns, request-group count, and observed/hook times. `observed-task` selects only the exact hash. It does not automatically join another run or profile.

For a goal spanning multiple observed runs, create an explicit private manifest:

    python report.py init --manifest /private/path/tasks.json
    python report.py identity
    python report.py report --home /absolute/profile/home --manifest /private/path/tasks.json --task OPERATOR_TASK_ID

Manifest version 1 contains `tasks`. Each task has:

- `id`: generated 32-character operator grouping ID;
- `operator_asserted_completed`: boolean, initially false;
- `operator_asserted_lineage_complete`: boolean, initially false;
- `observed_tasks`: zero or more 64-character emitted-task hashes for explicit same-profile multi-run grouping;
- `members`: explicit session/turn lineage selectors with `session`, `turns`, and `parent`.

Use `observed_tasks` when run boundaries matter. A whole-session member intentionally selects all observations in that session and can include multiple runs. Multiple operator tasks may select disjoint turns from one session. Duplicate run assignment, overlapping selection, missing parents, and cycles are rejected. Observed subagent links expand selected parent members to child sessions. Cross-profile joins remain unsupported because every command reads exactly one profile home.

Subagent start and stop events merge into one link without losing `seen_start`, `seen_stop`, allowlisted `child_status`, or bounded `duration_ms`. Child goals, summaries, roles, and tool histories are not retained.

Completion is an operator assertion only:

    python report.py complete --home /absolute/profile/home --manifest /private/path/tasks.json \
      --task OPERATOR_TASK_ID --assert-completed

The CLI rejects empty or unobserved groups, any selected pending or conflicted observation, and ambiguous cross-task assignment. Final responses and session endings cannot set the assertion. `operator_asserted_lineage_complete` remains an explicit manual manifest assertion. Neither assertion proves exhaustive hook/provider coverage. `complete_task_cost` therefore remains null.

## Optional rates

Pass `--rates /private/path/rates.json` to `report` or `observed-task`. No prices or network lookups are built in. The file contains exactly `version: 1`, a bounded nonempty `revision`, `currency: "USD"`, and rate rows for provider, model, and the four disjoint billable buckets. Rates are nonnegative decimal strings per million tokens. The revision is emitted only as a hash.

`observed_estimate` is present only when every selected physical observation has all four buckets and a supplied rate. `known_priced_subtotal` is explicitly partial. Estimates are not invoices.

## Verification and rollback

Run from this directory with the Hermes source virtual environment:

    mkdir -p .test-runtime evidence
    PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH=/path/to/hermes-agent \
      /path/to/hermes-agent/venv/bin/python -m unittest discover -s tests -v

The tests use only synthetic temporary profile homes and no provider calls. Before activation, rollback is removal of this plugin directory. After a separately approved activation, disable the supported setting and unload through normal Hermes plugin controls. Retain or remove the private database only by operator choice.
