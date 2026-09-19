# Telegram chief-of-staff dispatch

## Contract and deployment boundary

This default-off lane accepts ordinary unaddressed text in explicitly opted-in
Telegram groups/topics. Ace (`default`) classifies before any answering turn.
The selected existing profile runs a real `AIAgent`, retaining its configured
provider/model, persona, tool resolution and action approvals. That profile's
existing standalone gateway publishes the buffered final using its own adapter.
It is not the older `Engineering:` auxiliary-only public-handoff pilot.

Persistent specialists are exactly `revenue`, `product`, `design`, `engineering`
and `finance-ops`. No new profiles, services, models or multiplex migration are
required. Existing authorization remains authoritative. This does not activate
anything, authorize new groups, or permit external actions by the model.

## Configuration

Apply an identical configuration to all six participating gateways only after
release approval. Synthetic example, not a live destination:

```yaml
platforms:
  telegram:
    team_dispatch:
      enabled: false
      coordinator_profile: default
      specialist_profiles: [revenue, product, design, engineering, finance-ops]
      destinations:
        - chat_id: '-100'
          thread_id: '7'
        - chat_id: '-200'
          thread_id: null
```

Chat and topic identifiers must be quoted canonical strings. A null topic means
only the main/non-topic lane, not all topics. Maximum 64 explicit destinations.
Wholly disabled fleets leave native behavior unchanged. If any participant opts
in, patched participating gateways suppress unaddressed native admission for
those exact destinations even when their own opt-in is absent. This suppression
does not authorize generation. Missing, disabled or mismatched participants
allow only one truthful Ace failure outcome. Non-team and DM traffic stay native.
Enabled malformed
configuration fails closed. The same destination may not also enable legacy
`public_handoff`. Do not broaden `allowed_chats`, sender gates or topic gates.
An empty native allowed-chat list retains its native unrestricted meaning;
team dispatch still requires an exact destination match.

Each live gateway renews its profile/bot-identity/policy-fingerprint registration
in the shared local ledger only when connected and published as the current
adapter. Disconnection/disablement/shutdown revokes readiness. Wiring an unready
replacement does not claim queued work. All six distinct identities must have a registration
less than 45 seconds old with the same policy fingerprint. This is operational
coherence evidence, not Telegram membership evidence. Fresh sibling configuration
fingerprints also fence generation, rather than trusting a stale heartbeat after
a partial configuration change. This reads settings, never sibling credentials
or transcripts. Old unpatched binaries cannot enforce the suppression gate;
verify all loaded code versions before activation. Verify membership and the
ability to receive ordinary group updates separately before release. Never use
cron output destinations as group-dispatch approval.

## Runtime seams

- `GatewayAdapterLifecycleMixin._wire_adapter_handlers` installs the callback,
  including on replacement adapters. `_handoff_watcher` drains the existing host
  path; there is no second general coordination service.
- Telegram text, command, location and media ingress enforce sender admission
  before the team callback. Team admission adds gateway sender, own-bot, native
  chat, exact destination and topic checks before touching ownership state.
- Current authenticated bot replies and direct addressees are resolved before
  semantic selection. Durable human/outbound message aliases preserve root
  ownership. Several current addressees resolve to Ace, never a public debate.
- Only Ace receives genuinely unaddressed new roots. A schema-constrained,
  non-streaming, tool-free call uses Ace's configured provider/model with a
  20-second bound. Codex/Responses preserves the strict schema in `text.format`
  and preserves an explicitly resolved credential/endpoint. Invalid output fails
  closed, not to a different model. Claim CAS checks the owner/reason snapshot
  so a stale classifier consumer cannot reclaim freshly routed work.
- A recent successful specialist answer from the same sender/group/topic is
  eligible for a separate tool-free relatedness check for at most 15 minutes.
  Mere shared domain is not continuity. Unrelated messages are classified anew.
- A `Verkeerde agent` / `wrong agent` correction at the start of a known reply
  allows one reassignment. Further correction returns to Ace for clarification.
- Telegram's actual text debounce path is reused. Likely adjacent client splits
  retain all constituent IDs. Separate ordinary roots do not merge. The durable
  envelope and its full input-ID set are inserted atomically before generation.
- Each selected gateway constructs its normal profile agent with only that
  root's public history, at most four delivered request/answer pairs. There is
  no private CLI or Bot Chat transcript import and no cross-group lookup.
- Final text is buffered. No normal stream/progress/voice callbacks are wired.
  The runtime denies `send_message` through the existing tool whitelist and
  binds native finite-session `async_delivery=False` so detached terminal and
  delegation completion cannot create a second public turn. Tool schemas and
  normal capability resolution are otherwise preserved.
- Only the selected public owner's connected bot refreshes native typing during
  the actual buffered conversation, not during classification, construction or
  peer consultation in another profile. Each refresh rechecks authorization,
  current adapter/identity and durable ownership. Completion, cancellation,
  timeout, generation error or lost ownership stops refreshes. Telegram has no
  explicit stop action, so its existing bubble can linger until natural expiry.
  Native typing disablement, pauses and transport backoff remain respected;
  typing errors never create an extra message or block the final.
- Standard Markdown finals use the existing Telegram formatter. Preparation is
  local and precedes the send fence. The raw 4,000 UTF-16-unit cap is preserved;
  if conversion/escaping expands beyond Telegram's 4,096-unit bound, plain text
  is chosen before sending. This conservative payload bound also bounds rendered
  text and table expansion. Encoded link URLs of 1,999 characters or more also
  choose plain text locally, because the formatter's bounded backward scan can
  escape their closing delimiter. Ordinary shorter links retain formatting.
  There is still exactly one transport attempt, no
  chunking, and no formatted-to-plain retry after sending. The formatting path
  does not alter receipt identity or uncertain-delivery quarantine.
- Actions requiring a new interactive approval are denied rather than granted
  or allowed to outlive the finite delivery obligation. A sanitized Ace failure
  follows. Existing profile policies are not relaxed. Terminal/network tools
  are not an OS sandbox; prompt/action approvals still constrain their use.

## Supported and unsupported input

Ordinary text, direct addressed text, authenticated bot replies and human reply
lineage use the bounded team lane. Text is limited to 6,000 characters; final
text must fit one Telegram message (4,000 UTF-16 code units at validation).

Directly addressed media/control commands without an existing team-root reply
continue through the native selected bot's capable path. Unaddressed media,
location, slash commands, over-limit text and unsupported routed controls get
one clear Ace response instead of accidental native fan-out. `/stop` replying
to a known root, including a not-yet-flushed merged input alias, cancels execution
and pending final delivery for the originating sender. Native agent interrupt
checks consult the durable cancellation/deadline fence, and a bounded monitor
propagates hard cancellation to already active execution. Delayed construction
cannot start a cancelled turn. This is not a hard kill of an arbitrary system
call already executing, nor rollback of an action admitted before cancellation.
Other routed commands
are explicitly unsupported in this finite lane; this is not a replacement for
the native long-running interactive gateway session.

## Ledger, receipts and failures

`<default Hermes root>/team_dispatch.sqlite` reuses `PublicHandoffStore` CAS,
root serialization, send reservation, expiry, fallback reservation and quarantine.
It contains bounded public envelopes/answers, public identifiers and receipts,
never bot tokens or private chat transcripts. File creation uses mode 0600;
existing local-user profile homes remain the trust boundary.

The request deadline is 105 seconds. Generation reserves the last 15 seconds
for sending. Definitive generation/admission/Telegram rejection permits one
sanitized Ace fallback. Healthy gateway scheduling plus its 15-second send
bound keeps failure handling approximately within two minutes. An unavailable
Ace cannot provide a time-bound public fallback until service resumes.

A send succeeds only after Telegram's returned message ID, group, topic and bot
identity match the requested target. One narrow documented exception: when no
topic thread was requested and the send used a reply anchor, Telegram can echo
the anchor's reply-thread id on the returned message even in a non-forum group
(telegram-bot-api #798). A returned thread id equal to the requested reply
anchor is accepted as that echo and persisted as `reply_thread_echo` in the
receipt, keeping the requested `thread_id`; any other unexpected thread id
remains an identity mismatch, and configured topic threads keep strict
equality. Timeout, identity mismatch, cancellation
while sending or commit ambiguity is quarantined. No blind retry or competing
Ace fallback is allowed for uncertain delivery. Fallback sending has its own
persisted fence. A process crash after the fence also remains uncertain.
This is receipt-verified single-owner delivery, not an exactly-once Telegram
guarantee. Preserve uncertain rows for operator reconciliation.

## Verification boundary

`tests/gateway/test_team_dispatch_*.py` exercises the actual adapter ingress,
watcher, policy/store, provider client, normal profile agent/tool loop and own
adapter send receipt. External model HTTP responses and Telegram Bot API are
synthetic. Profiles and group IDs are isolated fixtures, not live profile copies.
The file-tool, terminal-tool and dangerous-command approval denial are real.
Fresh-process imports assert the candidate modules are loaded, not the installed
checkout. The tests prove routing/runtime contracts with supplied classifier
verdicts, not a real model's Dutch classification quality or real Telegram delivery.

For canonical local verification:

```sh
HERMES_TEST_FILE_RETRIES=0 scripts/run_tests.sh tests/gateway/test_team_dispatch_policy.py tests/gateway/test_team_dispatch_runtime.py tests/gateway/test_team_dispatch_lifecycle.py tests/gateway/test_team_dispatch_boundaries.py
```

## Existing peer consultation, locally exercised

This lane does not add an agent-to-agent engine. `tools.bot_mode_dm` permits
`message_agent` only in a managed canonical `Bot Chat`; a `team-<profile>-<root>`
Telegram session does not satisfy that gate. Existing installed specialist
instructions can use a bounded fixed-profile `hermes -p <profile> chat --in ~ -c
"Bot Chat" --create-if-missing -Q --query-file <file>` consultation via terminal.
Tests now exercise the actual CLI entrypoint, profile/session resolution and
normal tool loop in isolated subprocess homes, with only external model HTTP
and Telegram replaced. A fixed allowlisted target receives a public-only query
file, executes a real file read, and returns its bounded result to the selected
Engineering owner, which alone publishes. A seeded unrelated private-history
canary stays out of the public turn. A separate actual CLI dangerous-command
probe proves the target's manual approval policy is not silently granted.

This is local existing-transport evidence, not a new peer engine, live-provider
quality certification, or an OS privacy sandbox. Keep the canonical Bot Chat
gate intact. A consulted peer must not return private transcripts or independently
publish. Installed CLI availability, actual service environments, approval
configuration and bounded real-model completion remain live release checks.
Separate-process tests also exercise all six gateway homes and identities on
one ledger, overlapping claim consumers, reconnect publication and watcher
shutdown/restart. They are synthetic processes, not the live service fleet.

## Reversible release checklist, owned by Ace

1. Independently review the exact frozen patch and evidence against the ordinary
   natural-language requirement, not the legacy pilot. Recheck if the base moves.
2. Resolve the redacted eligibility inventory to current approved group/topic
   identifiers. Some canonical delivery IDs may be historical. Verify existing
   approval, all six memberships, current bot identities and update visibility.
   Session records alone are not membership or enablement approval.
3. Verify actual profile providers/models and context/data restrictions. Exercise
   the real configured classifier/provider structured-output path separately;
   synthetic custom-provider tests do not certify Codex/Alibaba live behavior.
4. Verify service definitions and running code versions. Retain the prior stale
   Engineering service/module preflight gate until the owner confirms it resolved.
   Never use signals/launchctl/alternate processes to bypass restart guards.
5. During an approved quiet window, drain native turns and preserve configuration,
   exact installed tree and ledger backups. Install only the reviewed patch.
   Configure all six participants coherently, leaving unrelated destinations and
   allowlists unchanged. Roll out no partial enabled team fleet.
6. Use owner-authorized supported restarts. Confirm code paths, six registration
   fingerprints, authorizations and explicit team opt-ins before a live canary.
7. Authorized canaries must cover each specialist, Ace self/clarify, direct reply,
   two interleaved roots, topic identity, one harmless real tool, failure handling
   and the separate peer-consultation proof. Record actual bot/message receipts.
8. Stop on duplicate ownership, privacy leak, wrong identity/topic, non-owner
   inference, classification failure, blocked approval bypass or uncertain send.
   Preserve evidence; do not resend uncertainty to manufacture success.
9. Rollback coherently: stop admission/drain all participants, disable the same
   destinations everywhere, preserve/reconcile pending and uncertain rows, restore
   backed-up code/config and use approved restarts. Reverse the exact patch only
   against the frozen tree. Never delete the ledger or replay uncertain rows.

Recheck before activation and whenever code, provider, profile policy, group
membership, credentials, native authorization or service topology changes.
