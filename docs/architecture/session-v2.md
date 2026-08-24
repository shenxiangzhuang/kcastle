# Session v2

Status: accepted

## Decision

Session v2 uses a project-local SQLite WAL database as the only durable source of truth.
JSONL is an export format, not a runtime database. A single session engine owns the session
machine, serializes commands, commits complete domain transactions, evolves the in-memory machine
only after commit, publishes the committed transaction, and only then starts external effects.

The desktop builds one canonical `SessionDocument` from committed transactions. Conversation,
trajectory, timing, details, search, and composer statistics are selectors over that document.
Transient interaction state such as hover, selection, viewport, expanded rows, and active details
tabs never enters the journal.

Session v1 JSONL files are intentionally unsupported and are ignored by the catalog. There is no
dual-write or compatibility fallback.

## Invariants

1. Only a committed transaction is a durable fact.
2. One logical state transition is either wholly committed or wholly absent.
3. One actor owns each active session machine.
4. `SessionMachine::plan_batch` is pure and validates a complete candidate transition.
5. The live machine evolves only from the store's commit receipt; persistence errors never trigger
   an in-memory rollback.
6. A UI subscriber failure cannot change durable state or fail an already committed command.
7. External model and tool effects start only after a durable intent is committed.
8. Commands and effects carry stable typed correlation IDs; stale results cannot mutate a newer
   attempt.
9. A non-idempotent tool interrupted after dispatch is `UnknownSideEffects` and is never retried
   automatically.
10. Incremental projection of every accepted prefix equals a full replay of that prefix.
11. All desktop read models have the same `as_of_revision`.
12. Invalid, corrupt, archived, or non-v2 sessions are filtered by the catalog before a desktop
   runtime is created.

## Transaction protocol

Every append carries `session_id`, `tx_id`, `expected_revision`, and an ordered event batch. The
store performs an idempotency lookup, compares the expected revision, writes the transaction and
all events, and advances the session revision in one SQLite transaction. A repeated `tx_id` with
the same digest returns the original receipt; a different digest is corruption.

If the connection loses the commit result, the store resolves the outcome by querying `tx_id`
before accepting another append. Event observation time is distinct from transaction commit time.
Durations use monotonic time only within one clock/boot identity; wall time is used for actual-time
placement.

## Atomic domain boundaries

- `InputSubmitted` durably owns an input. `InputAttached` simultaneously removes it from the inbox,
  places the user surface item, and binds it to one run/turn/step. There is no `InputConsumed`.
- Assistant completion and its finalized tool declarations are one transition.
- Tool authorization and dispatch intent are committed before any runner task exists. The runner
  samples execution start immediately before invoking the tool and reports that observation back
  to the single owner; the owner persists start, finish, and result attachment as distinct facts.
  A crash after durable dispatch but before a durable finish is conservatively recovered as
  `UnknownSideEffects`, while an already durable success or error outcome is preserved.
- Normal completion, cancellation, failure, and crash recovery close all affected requests, tools,
  compactions, steps, turns, and runs in one terminal transaction.
- A request is built from the full canonical request snapshot that was committed immediately before
  dispatch, never from a second mutable configuration path.

## DSH desktop semantics

- Assistant duration: `ModelRequestStarted -> AssistantCompleted`.
- TTFT: `ModelRequestStarted ->` first non-empty text, reasoning, tool name, or tool arguments
  observation. Preparation and automatic compaction between `StepStarted` and the request are not
  counted twice as LLM time.
- Decode: first token to assistant completion.
- Tool total: tool call observation to tool output attachment.
- Tool execution: execution start to execution finish.
- Compaction timing and usage are retained but compaction usage never contributes to composer LLM
  totals.
- The latest assistant usage sample replaces earlier samples for the same step.
- The initial system item is semantically ordered before the first user item. An unchanged resume or
  config-only change does not create a system row; only system text or ordered tool schema changes
  do.
- Interrupted partial assistant output is visible but contributes no completed LLM timing.

## Desktop ownership and performance

Each GPUI `SessionRuntime` owns one mutable `SessionDocument` and publishes one immutable
`Arc<SessionView>`. `SessionMachine` is the sole semantic validator: the desktop only preflights the
committed event cursor before applying a complete batch, so it cannot partially project a transport
gap and does not maintain a second lifecycle state machine. Applying a committed batch produces a
small patch of changed stable IDs; persistent maps and vectors share untouched structure and
preserve stable record arcs.

Canonical messages carry a monotonic content revision. Visible GPUI rows synchronize markdown
directly by `(projection generation, message ID, revision, markdown mode)`, making unchanged rows an
O(1) check without aliasing a replacement runtime that reused the same session namespace and IDs.
The namespace scopes transient expansion and rating overlays; bounded notices bypass the
presentation store entirely, so no second presentation change journal is needed.

Timeline layout is a pure transformation with one geometry source for rendering and hit testing,
one bounded field-aware change journal for search and geometry consumers, and record indices as the
only cell identity inside a projection generation. Consecutive search-only revisions coalesce into
one range, so an off-screen geometry consumer keeps continuity during arbitrarily long streaming
text. The renderer clips to a dedicated plot rectangle. Interval lookup serves hover and selection,
and subpixel lane binning limits rendered primitives while preserving the full semantic item set.
Selection and viewport are independent typed states bound to an axis/document generation; an
in-progress drag is the only additional interaction state.

## Verification gates

- Store fault injection proves multi-event transactions are all-or-none and `tx_id` is idempotent.
- A real subprocess hard-kill test proves an interrupted SQLite transaction rolls back and releases
  its writer lock. Recovery-prefix tests cover request, tool, compaction, and terminal lifecycle
  boundaries, including parallel tools whose durable completion and ordered attachment differ.
- Property tests prove live apply equals replay and that invalid states are unreachable.
- DSH golden fixtures cover request changes, first token, usage replacement, compaction exclusion,
  parallel tool timing, and interrupted responses.
- A 100k-event storage benchmark and a 10k-item streaming benchmark check replay, incremental
  projection work, timeline interaction, primitive count, and storage footprint.
- The desktop is finally exercised as a native application for hover, range selection, dimming,
  ledger scrolling, zoom, details, composer statistics, and narrow-window clipping.
