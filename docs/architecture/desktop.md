# Desktop architecture

Status: accepted

## Decision

The desktop builds one canonical `SessionDocument` from committed transactions. Conversation,
trajectory, timing, details, search, and composer statistics are selectors over that document.
Transient interaction state such as hover, selection, viewport, expanded rows, and active details
tabs never enters the journal.

The [session protocol](session.md) owns durable facts and lifecycle validation.
[App storage](app-storage.md) owns product configuration and catalog persistence.

## Invariants

1. Incremental projection of every accepted prefix equals a full replay of that prefix.
2. All desktop read models have the same `as_of_revision`.
3. Invalid, corrupt, archived, or non-v2 sessions are filtered by the catalog before a desktop
   runtime is created.

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

- Projection tests check that incremental apply and full replay produce the same document.
- DSH golden fixtures cover request changes, first token, usage replacement, compaction exclusion,
  parallel tool timing, and interrupted responses.
- A 10k-item streaming benchmark checks incremental projection work, timeline interaction,
  and primitive count.
- The desktop is finally exercised as a native application for hover, range selection, dimming,
  ledger scrolling, zoom, details, composer statistics, and narrow-window clipping.
