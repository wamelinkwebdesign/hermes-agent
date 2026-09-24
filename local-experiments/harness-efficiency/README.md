# Harness-efficiency experiments

This is a fork-only, opt-in experiment runner. It never installs a skill, changes a live prompt, selects a production route, or changes compaction thresholds. Do not treat it as a production context-engine plugin.

## Reproduce

Use a Python environment with the current Hermes dependencies and an authenticated Codex CLI. The program does not read authentication files. Supply an output directory outside the repository; generated skill snapshots and model transcripts can contain private material and must not be published.

```sh
python run_pilots.py prepare --skill /path/to/SKILL.md --hermes-source /path/to/hermes-agent --output /private/new-eval
python run_pilots.py run --hermes-source /path/to/hermes-agent --output /private/new-eval
python run_pilots.py evaluate --hermes-source /path/to/hermes-agent --output /private/new-eval
```

`prepare` freezes a lossless partition of the skill entrypoint, a routing prototype, synthetic cases and the actual baseline summary prompt rendered by the checked-out Hermes version. Source skill support files are not repackaged; the result is not an installable replacement skill. `run` invokes seven model calls, preserving every prompt, response, error and usage record. It includes the candidate reference-selection call in the comparison. `evaluate` scores existing results without additional model calls. Existing run directories cannot be overwritten.

Both arms use `gpt-6-astra`, high effort, read-only sandbox, ephemeral sessions and the same CLI configuration overrides. This does not establish isolation from every built-in skill scanner: the observed run loaded unrelated skill catalog context despite `--ignore-user-config`. Inspect event errors and confirm isolation before any stronger experiment. Do not disable the user's installed skills to repair the benchmark.

The summary experiment changes only the candidate body template. Both checkpoints use Hermes's existing deterministic summary prefix and lean anchor/user-message/recovery augmentation. It does not exercise the full compressor's redaction, source grounding, ghost-skill reinjection, failover or commit-fencing lifecycle. Those protections must all remain in any future integration.

## Observed exploratory outcome

- Both skill arms answered four frozen policy/operation questions correctly. The candidate found each required topic. Its core was substantially smaller, but its additional selection request made observed uncached input and output higher than baseline. This two-call design is not recommended for rollout.
- The candidate checkpoint was smaller and retained all eight scored continuity facts. The baseline response ignored JSON format, so the pair failed the frozen format gate. Neither the small sample nor this contaminated CLI setup establishes task-quality equivalence.
- There is no production savings claim, no statistical non-inferiority claim and no live candidate rollout.

## Follow-up gate

Use a clean, verified no-extra-context transport and actual same-session tool discovery rather than a standalone selector invocation. Freeze a diverse untouched task set, balanced run order and cache conditions. Include actual tool execution, auxiliary and child work, retries, output format, user corrections, stop/undo behavior and post-compaction recovery. Promote only after completed-task cost improves without observed quality loss. Leave models, reasoning effort and safety rules fixed.
