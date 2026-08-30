# Telegram Team Dispatcher

> **Audience:** Gateway operators
> **Source files:** `gateway/telegram_team_routing.py`, `gateway/telegram_team_classifier.py`, `gateway/telegram_team_context.py`, `gateway/telegram_team_delivery.py`, `gateway/run.py` (`_handle_telegram_team_ingress`), `plugins/platforms/telegram/adapter.py`
> **Related:** [Profile-Based Routing](profile-routing.md), [Session Lifecycle](session-lifecycle.md)

## Overview

Several Telegram bots — a coordinator and one or more specialists — share one group. Without
a dispatcher, each bot decides for itself whether a message is "for it", using its system
prompt. That is non-deterministic: two bots answer, or none do, or a bot inherits context
from an unrelated project.

The team dispatcher moves that decision out of the prompt and into the host. For every
relevant human message it picks **exactly one public owner**, gives that owner a bounded and
privacy-safe view of the conversation, and guarantees the group gets **exactly one terminal
outcome**.

Two independent guarantees, worth keeping separate in your head:

| Guarantee | Mechanism | Failure it prevents |
| --- | --- | --- |
| Not several owners | Deterministic routing + process-local ingress dedupe | Duplicate replies |
| Not zero outcomes | Durable obligation ledger (`state.db`) | Silent drops |

## Configuration

Team routing is declared on the **coordinator's** profile, under the Telegram platform's
`extra`:

```yaml
gateway:
  multiplex_profiles: true          # required

platforms:
  telegram:
    enabled: true                   # required
    extra:
      team_routing:
        coordinator_profile: default      # MUST equal the active profile name
        coordinator_username: Ace_Bot     # without the @
        members:                          # profile -> public bot username
          default: Ace_Bot
          engineering: Woz_Bot
          design: Virgil_Bot
        allowed_chats:                    # group chat ids, as strings
          - "-1001234567890"
```

All four keys are required and the block is validated **atomically** — any malformed field
rejects the whole thing. Constraints worth knowing before you edit:

- `coordinator_profile` must be the active profile, and must appear in `members` mapped to
  `coordinator_username`.
- Usernames must be unique across members and match Telegram's rules (5–32 chars,
  `[A-Za-z0-9_]`).
- `allowed_chats` entries must be **group** chat ids (negative integers). DMs are never team
  chats and keep the ordinary single-bot path.
- Every member profile needs a live, connected Telegram adapter whose cached bot identity
  matches the configured username. A stale or mismatched identity disables team routing at
  runtime rather than guessing.

Invalid configuration raises `MultiplexConfigError` at startup — the gateway refuses to boot
rather than silently falling back to prompt-based routing. Absent configuration (no
`team_routing` key) is not an error; the dispatcher simply stays off.

### Feature gates

| Key | Default | Effect when off |
| --- | --- | --- |
| `gateway.telegram_team_delivery` | on | No obligations recorded. Routing still works; the no-silent-drop guarantee is gone. |
| `gateway.delivery_ledger` | on | Outbound final-response redelivery after a crash is disabled. Independent of the above. |

## How the owner is chosen

Precedence, highest first. The first rule that matches decides, and the rest are not consulted.

| # | Rule | `route_reason` | Owner |
| --- | --- | --- | --- |
| 1 | Inside an owned chain, exactly one *other* member is @-mentioned | `mention_handover` | That member (thread moves) |
| 2 | The message replies into an already-owned root chain | `reply_to_root_chain` | The root's owner |
| 3 | The message replies to a team bot's message | `reply_to_team_bot` | That bot |
| 4 | Exactly one team member is @-mentioned | `single_team_mention` | That member |
| 5 | Two or more team members are @-mentioned | `multiple_team_mentions` | Coordinator |
| 6 | Nobody was addressed | `unaddressed_ingress` → classifier | See below |

Only rule 6 reaches the classifier. Explicit addressing is never re-decided by a model — that
would both cost tokens and risk disagreeing with the human. After classification the reason
becomes `semantic_specialist` (redispatched to a specialist), `semantic_self` (coordinator
answers), or `semantic_clarify` (coordinator asks).

Mentions are read from Telegram entities where present, and from raw text only when a
message carries no entities at all. Quoted `reply_to_message` text is **never** scanned — a
quote must not be able to address a bot on the human's behalf.

### Handing a thread over

Replying to an answer and naming somebody else is how a human passes a conversation on, so
rule 1 moves the whole root family to that member: the root, every constituent of the
original batch, and every alias recorded since. The new owner inherits the root scope, so the
conversation continues in the same session rather than forking, and later plain follow-ups
belong to them.

The rule is deliberately narrow. A handover needs **exactly one** named member who is not the
current owner:

| Message inside an owned chain | Result |
| --- | --- |
| `@Virgil_Bot what do you think?` | Hands over to design |
| `and also this` | Ordinary follow-up, stays put |
| `@Woz_Bot and also this` (current owner) | Emphasis, not a move — stays put |
| `@Virgil_Bot @Ace_Bot who owns this?` | Ambiguous — stays put |
| `@Woz_Bot thanks, @Virgil_Bot thoughts?` | Ambiguous — stays put |

The last two cases cannot be told apart from "hand this over" and "all of you look at this"
using mentions alone, so the thread stays where it is. A wrongly moved conversation is much
harder for a human to notice than one that did not move.

Ownership is otherwise immutable for the life of a root — that immutability is what stops a
chain drifting between bots. An explicit handover is the only sanctioned exception, it moves
the entire family atomically, and it fails closed rather than leaving a thread half-moved.

## Context given to the classifier

The classifier sees the current message plus a bounded slice of **one exact root session**,
looked up by exact session key (`telegram-team:<chat_id>:<root_id>`). There is no search, no
similarity, and no scan across sessions. A loaded session whose origin disagrees with the
current root — different root, chat, thread, scope or profile — contributes nothing and marks
the collection unsafe.

Everything rendered is labelled `[UNTRUSTED …]` so the model cannot mistake group text for
instructions, and every line passes credential redaction first. If redaction cannot make the
text safe, the message clarifies instead of being classified.

## Exactly-once terminal delivery

Central ingress records a durable obligation the moment it accepts a message, in the
`telegram_team_obligations` table of `~/.hermes/state.db`. The obligation is keyed on the
message's **batch head**, so a follow-up question inside a root family owes its own answer
and cannot be discharged by the answer to the first one.

```
                    ┌──────────────────────────── answered   (a real reply went out)
record_obligation   │
      │             ├──────────────────────────── clarified  (a clarification went out)
      ▼             │
   pending ─────────┤
      │             └──────────────────────────── abandoned  (closed WITHOUT speaking)
      │
      │  claim_fallback()          mark_fallback_delivered()
      └──────────► fallback_claimed ──────────────► fallback
                          │
                          └── release_fallback_claim() ──► pending
```

Two distinct mechanisms close the silence gap, and the split is deliberate:

- **In-process.** When a turn ends without putting anything in front of the group, the owner
  claims the fallback, sends it, then marks it delivered. Claiming *before* sending is what
  makes it exactly-once: a crash between the two cannot produce a second fallback, and a
  failed send is *released* rather than marked delivered, so a failure never becomes a silent
  success.
- **Cross-restart.** On boot, `sweep_recoverable()` claims only obligations whose **owning
  process is gone** and speaks a recovered-reply notice for them.

There is deliberately **no "alive but overdue" sweep.** A long agent turn is normal, and a
fallback posted beside a still-working agent is exactly the duplicate reply this dispatcher
exists to prevent. Obligations that outlive `STALE_AFTER_SECONDS` (6h) under a live owner
transition to `abandoned` and are logged, never answered — a fallback arriving hours later in
a group chat is worse than a visible gap in the ledger.

The ledger is a **safety net, never a gate**. Every call is best-effort: a disabled, locked or
broken ledger must never consume, delay or drop a message that routing already accepted.

## Observability

**Ledger state** — the fastest read on "did anything get dropped":

```bash
python -c "from gateway.telegram_team_delivery import debug_rows; print(debug_rows(50))"
```

Each row carries chat, root, head, owner profile, route reason, state, attempts and last
error. What to look for:

| Observation | Meaning |
| --- | --- |
| Rows stuck in `pending` with a live owner | Turns running long, or the settle hook is not firing |
| Any `abandoned` row | A message never got a public outcome. **Always worth investigating.** |
| `attempts` climbing toward 3 | Fallback sends are failing repeatedly — check the owner's transport |
| Rows in `fallback_claimed` after a boot | Claimed but not yet sent; the next boot reclaims them |

**Log lines** (all in the gateway log, `WARNING` or above):

- `Telegram team obligation <id> abandoned without a public outcome` — the silent-drop alarm.
- `Telegram team fallback could not be delivered for chat … root …` — fallback send failed;
  the debt stays open.
- `Telegram team live identity is stale or unverifiable for profile '<p>'` — routing is about
  to disable itself.
- `Telegram team ingress preparation failed` — ingress failed closed and consumed a message.

**Routing decisions** are visible per event in `event.metadata`:
`telegram_team_owner_profile`, `telegram_team_root_message_id`, `telegram_team_route_reason`.

## Operator recovery

**A message got no reply at all.** Check the ledger for that chat and root. A `pending` row
means the obligation was recorded and the turn has not settled — the process is probably still
working. An `abandoned` row means it exceeded the attempts cap or the stale cutoff; the
message is genuinely lost and needs resending by hand.

**Two bots answered.** This should be structurally impossible via routing. Capture both
message ids and the ledger rows, then check whether the second reply came from a path outside
the dispatcher (a cron job, a `send_message` tool call, a manual `/` command).

**A bot answered with the wrong project's context.** Capture the chat id, the root message id
and the session scope. Context is exact-keyed, so this indicates either a scope-id mismatch or
a session whose origin was mutated — both fail-closed paths that should have refused.

**Routing silently stopped.** The runtime disables itself when a member's cached bot identity
goes stale or stops matching the roster; look for the stale-identity warning. It re-enables on
the next successful identity refresh. Until then every message takes the legacy per-bot path.

**The ledger is misbehaving.** Turn it off with `gateway.telegram_team_delivery: false` and
restart. Routing keeps working; you lose only the no-silent-drop net.

## Rollback

Ordered least to most disruptive. All are safe with the gateway running; each needs a restart
to take effect.

1. **Disable the delivery ledger only** — `gateway.telegram_team_delivery: false`. Keeps
   deterministic routing, drops the obligation layer.
2. **Disable team routing, keep the bots** — remove or rename the
   `platforms.telegram.extra.team_routing` block. Every bot returns to its previous
   prompt-based behaviour immediately; no state migration and nothing to clean up. Root
   ownership is process-local, so it simply disappears with the process.
3. **Full revert** — `git revert` the range, or point the deployment back at the commit before
   it. The only durable artefact is the `telegram_team_obligations` table, which is inert when
   the code is gone and prunes itself.

Rollback leaves no orphaned state: ingress claims, root families and aliases are all
process-local, and obligations are bounded by an attempts cap, a 6h stale cutoff, a 7-day
retention window and a 500-row ceiling.

## Deliberate design decisions

Recorded because each looks like an omission until you know why:

- **No "alive but overdue" fallback sweep** — would race a working agent and produce the
  duplicate reply the project exists to prevent.
- **Stale obligations abandon rather than answer** — a very late fallback in a group chat is
  worse than a visible ledger gap.
- **The base adapter's error notice counts as a terminal outcome** — otherwise a failed turn
  would get a fallback stacked on top of the error the user already saw.
- **Obligations key on the batch head, not the root** — otherwise the answer to a first
  question would silently discharge a follow-up.
- **A separate ledger from `delivery_ledger.py`** — an inbound debt and an outbound send fail
  differently and must never share a sweep.
