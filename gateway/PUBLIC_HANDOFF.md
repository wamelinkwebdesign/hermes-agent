# Public Telegram handoff pilot

Status: isolated local build only, default OFF. No live rollout is authorized by this document.

## Scope and selection

Two standalone gateways retain their own profiles, Bot clients and credentials. Only the default profile may initiate a root request to `engineering`. Both must explicitly opt into the same single group/topic and distinct bot IDs. Multiplex and other profiles are unsupported and do not install the pilot. No new listener, general model tool, token copying, or private Bot Chat transfer is involved.

An authenticated, admitted, top-level text starting exactly `Engineering:` selects Engineering BEFORE Ace enters ordinary gateway dispatch. `@<coordinator_username> Engineering: ...` also works. The original text is the public brief (maximum 6000 characters). This deterministic pilot selector is deliberately not a model classifier. Ordinary questions and direct specialist mentions retain the native path. Replies to a known public task or delivered specialist answer continue its bounded specialist conversation; an explicit bot mention wins. `/stop` as a native reply cancels that conversation's active requests when sent by the root task author. Untagged unrelated top-level questions remain Ace-owned.

Selected text never reaches batching, typing, normal agent callbacks, tools, commentary, streaming or audio generation. The host owns a single plain-text send after a durable fence. There is no post-hoc `NO_REPLY` filter. This pilot answers public text using Engineering's configured auxiliary inference resolver, not Engineering's private agent transcript or its full coding/tool loop. It cannot execute development tasks, browse, attach files or produce audio. Media and normal direct-mention turns are not converted into public handoff tasks.

Location and media ingress check the same persisted ownership gate before native dispatch, download, observation or typing. A non-text reply to a known public task/answer is consumed without inference or public output, because this pilot cannot process it. Send a text follow-up instead. Explicit bot addressing may choose the ordinary native path, outside this pilot's bounded-context guarantee; unrelated media remains native traffic.

Within exactly the enabled pilot group/topic, host ownership overrides Engineering's ambient triggers: group/topic free response, `require_mention: false` and wake words cannot start a second native owner. Unaddressed top-level messages belong to Ace even if Engineering receives them first. Engineering's explicit mentions and replies to its own nonpilot answers retain native handling. This does not edit user configuration or affect other groups/topics, disabled pilots, or private chats. Both gateways must still be opted in coherently; this is not coordination with an unpatched/offline specialist native gateway.

## Data and ownership

`PublicRequest` version 1 carries stable request identity, source/target profile, Telegram platform, original chat/topic/message/sender identity, bounded brief, creation/expiry times and public root/reply linkage. Identity is derived from authenticated adapter input and checked against local policy, never parsed from model prose. Request identity deduplicates Telegram message identity (chat, topic and message); edited/replayed updates do not replace an already accepted brief.

The shared spool lives only at `<default-root>/profiles/engineering/public_handoffs.sqlite`. It contains public brief/answer data and routing receipts, no tokens. Both trusted local gateway processes access it. The default gateway never reads Engineering's private session DB. Engineering reuses its existing SessionStore group binding and creates a separate `telegram` SessionDB conversation under a deterministic public request ID. Private CLI rows are neither transferred nor rebound. Existing CLI home-channel handoffs remain unchanged.

Before inference and again before send, the receiver validates its enabled/current adapter, own Bot ID, group/topic policy, adapter sender admission, gateway authorization (no adapter-delegation bypass) and an existing matching target group session. A reset group session invalidates an old public conversation. No missing group/session can fall back to a home channel or DM. Initial live group-session establishment is an external prerequisite, NOT performed by this build.

Native `allowed_chats` retains its existing semantics: an empty resolved set adds no group restriction; a nonempty set excluding the pilot chat rejects the handoff. Neither case replaces the required enabled pilot policy matching the exact approved chat/topic. Do not restrict the native list to the pilot chat, since that would change admission for unrelated groups.

The inference input consists of one fixed public-only system message, up to four delivered public exchanges for that root, and the new public brief. No SOUL, private memories, project context, plugins, tools or Bot Chat transcript are loaded by this path. Invalid/empty/oversized/tool-call responses fail before send. Prompt injection may affect answer quality but cannot change routing, provenance, authorization or invoke tools. Public task text still reaches the configured model provider; its retention policy remains an activation gate.

## Persistence and delivery contract

States: `accepted`, `running`, `delivered`, `failed_definitive`, `delivery_uncertain`.

- Atomic claims and per-root serialization prevent replay/concurrent terminal sends.
- The default request lifetime is 120 seconds; at most four public tasks per gateway run concurrently. Busy native sessions/draining gateways wait, bounded by expiry. Legacy handoff polling continues while public inference or fallback transport is slow.
- A committed `send_started_at` plus bound public/group session precedes any transport attempt. Exactly one plain-text Bot API attempt is used, with no chunking, rich-format retry or topic fallback.
- `delivered` requires the actual Bot API response to match bot, chat and topic, contain a positive outbound message ID, and carry the target profile/public session/group-session binding.
- A crash/cancel/timeout before the send boundary can become `failed_definitive`. After the boundary it becomes `delivery_uncertain`, including a mismatched receipt. Late results cannot reopen a terminal row.
- A definitive failure allows one static truthful Ace fallback after Ace's own admission recheck. Its own durable pre-send boundary forbids duplicate fallback after a crash or timeout. A fallback boundary without a receipt is uncertain, not retryable.
- Any uncertain delivery quarantines the known public conversation, including queued siblings and replies to earlier known messages. No blind resend or fallback. The spool does not offer an automatic reconciliation/resend endpoint.

Quarantine includes fallback `delivery_uncertain` receipts and fallback boundaries lacking a confirmed delivered receipt. Confirmed delivered fallbacks are not uncertain. Specialist claims and fallback claims enforce this fence atomically in SQLite, and a fallback cannot start while a sibling specialist request is running. A live fallback boundary is conservatively quarantined until its receipt is committed, so follow-ups arriving in that interval are not queued for automatic replay.

At-most-once safety favors possible loss over duplicate delivery. If a process dies immediately after the durable boundary but before an actual send, the system conservatively reports uncertainty. Telegram does not provide a transaction spanning this local SQLite write and the remote send.

## Proposed opt-in, not executed

Use the existing profile-local `hermes config set` interface after separate Dennis approval, not manual live YAML editing. The following is the schema under `platforms.telegram.public_handoff` in BOTH standalone profiles:

```yaml
public_handoff:
  enabled: false
  chat_id: '<approved negative group ID, quoted>'
  thread_id: '<approved positive topic ID, quoted, or null for no topic>'
  coordinator_bot_id: '<actual coordinator Bot identity, quoted>'
  specialist_bot_id: '<actual specialist Bot identity, quoted>'
```

Place it under `platforms.telegram`, not under `gateway`. These placeholder strings intentionally fail validation; do not use live IDs or tokens in committed examples. Missing/invalid configuration does not enable the pilot; `enabled` must be boolean `true`. Retain existing `allowed_chats`, `allowed_topics`/ignored-thread and sender/gateway allowlists. The pilot does not widen those controls. The numeric allowlist must be established from authenticated host information; runtime destination comes from the admitted message and must equal this allowlist exactly. Omit `thread_id` or use null only when the approved destination genuinely has no topic.

Prerequisites before activation: independent frozen-diff review; verify runtime versions and actual target membership/admission; choose the approved topic; establish Engineering's authorized group session; verify both distinct transport identities; approve model/provider data handling; preserve a backup of code/config and public spool. Enable both profiles coherently during an approved quiet window. A gateway restart, if needed to load code/config, and a live canary need their own approval. No such commands have been run.

## Rollback and reconciliation

Do NOT simply flip one gateway off while requests can still arrive. First arrange an approved quiet window and stop initiating new pilot requests. Let accepted/running requests settle or expire; cancel pending work where appropriate. Preserve the spool and public session evidence. Treat `delivery_uncertain` and any fallback boundary without a confirmed receipt as quarantine cases. A human must compare exact Telegram messages/IDs with the receipt before deciding what happened; never reset those rows to accepted or delete the spool as a retry mechanism.

After drain/quarantine, set `platforms.telegram.public_handoff.enabled` to false in both profiles via the normal configuration interface and restore only the approved code/config delta, with separately approved restart if required. Native explicit addressing remains the alternative. Do not resurrect prior pending rows on re-enable without reviewing them. This build adds no service and requires no dependency migration.

## Trust boundary and limitations

Trusted components are the authenticated Telegram adapter, two host gateways, and same-user local profile files. A profile is not an OS sandbox. This is NOT protection against a compromised local process with filesystem access; such a process can forge/delete the SQLite spool, read profile files or call Telegram itself. Do not expose spool mutation to models or untrusted plugins. New spool files request mode 0600; local disk encryption/backups and retention remain operator responsibilities. Unbounded historical dedup retention is intentional for this pilot; approve archival before sustained use, since deleting rows also deletes dedup and reply ownership evidence.

Live Telegram membership revocation, real Bot API failure behavior, OAuth/provider-specific inference transports, gateway reload/shutdown under production load and non-macOS runtime behavior have not been certified. A lost outbound ID after a crash cannot identify that message's later reply as a pilot receipt; unknown native specialist replies retain native routing rather than inheriting invented public context. Operators must reconcile uncertain conversations before continuing them. This is a bounded local response-routing proof, not live readiness, a general agent framework, or a security certification.

Tests: `tests/gateway/test_public_handoff_ingress.py` enters real Telegram text/command/location/media ingress and uses real YAML loading, wiring, queue, watcher, session and delivery code against temporary homes. Only inference/transport boundaries are faked; one probe traverses the real auxiliary resolver to a fake HTTP transport. Store probes also hard-exit a child interpreter across specialist and fallback send boundaries. Trigger tests use both adapter arrival orders and duplicate updates, without disabling specialist free response in the shared fixture. Run with canonical `scripts/run_tests.sh` together with the existing Telegram/handoff regressions. Recheck on any base-SHA, auth, Telegram adapter, auxiliary-provider or session-store change, and before live approval.

Historical design reference: `feat/telegram-team-dispatcher-p0` at e7b0d618b48c4cdf795fa094e3fb315b8ee6164a, authored by wamelinkwebdesign with Claude Opus 5 co-author credit. Its whitelisted host-event idea informed comparison; no code was copied, merged or cherry-picked. The stale control-socket/ownership framework was not imported. Webhook/API ingress was rejected for this pilot because it does not preserve admitted Telegram provenance or durable group ownership without additional machinery.
