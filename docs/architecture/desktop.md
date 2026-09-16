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

### Chat viewport

The native GPUI `ListState` owns variable-height row layout and scroll anchoring. On session
selection, the desktop restores `(message ID, source field, byte offset, offset within row)`
or tail-follow mode. Only visible rows plus 600 px of overscan in each direction allocate
selection/layout/presentation objects. History keeps source locators, not parsed documents.

Rows first display source text. A single background task prepares Markdown ASTs, code syntax
styles, and formula layout/SVGs. GPUI element creation, text shaping, and painting remain on the
UI thread, bounded by the working set. Completion remeasures the affected row; native list
anchoring preserves the top item's offset while heights above it change. Width changes use
native proportional remeasurement. Switching away discards presentations; only scroll anchors
and expansion/rating interactions survive. These are transient UI state, never journal facts.

Work is tagged with session/projection epoch, source-fragment revision, and theme. Publication
also requires current viewport demand and an uncancelled task. Offscreen work is cancelled;
the worker slot stays occupied until it returns, including after repeated session switches.
Cancellation is cooperative between parsing, highlighter construction, and formula operations.
An individual library call cannot be interrupted. Unchanged fragments retain their presentation
revision during streaming; native list splices retain unaffected measurements.

Sources up to 2 KiB remain intact. Larger sources split at paragraph boundaries and fenced code
splits at 24 lines/2 KiB with language context. Non-code blocks above 16 KiB use lossless plain
fragments to bound UI text layout; ordinary tables and multiline markup remain intact. Markdown
references spanning independently parsed fragments are not resolved across fragments. These
are explicit limits of the lightweight source index, not a second Markdown parser. Source indexing
still scans changed messages on the UI thread; journal loading/runtime projection retain their
existing ownership and are not database-page virtualization.

Prepared data has an estimated 8 MiB working-set budget and a 1 MiB per-fragment admission limit;
over-budget fragments remain readable plain text. Eviction drops ASTs, syntax spans, selections,
and reference-counted generated SVG leases. Current frames can briefly hold an extra lease.
Formula vectors use GPUI's themed SVG alpha mask. RaTeX's embedded raster glyphs (color Emoji)
are excluded from that mask and decoded once on the preparation worker into a transparent
color layer sharing the vector layer's viewBox and coordinates. Bounds expand to include color
strikes that exceed RaTeX's fallback ascent/descent, preserving the baseline's position relative
to both layers. The presentation owns its `RenderImage`
directly and includes the decoded BGRA bytes in the budget; it does not enter the global image
resource cache. Theme changes still tint vector symbols without recoloring Emoji.
This budget does not include canonical session data, native GPU/font resources, or the separate
Trajectory details renderer. There is no cross-session Chat presentation cache.

RaTeX font data is process-wide and outside the preparation budget. The local
[font patches](../../vendor/README.md) check primary Unicode outlines before
loading optional emoji/secondary fonts, and share default primary/secondary
font bytes. A missing outline retains the existing fallback chain; actual emoji
can still discover a large font. Font bytes are held by `Arc<FontData>`: on macOS,
canonical system font paths backed by a descriptor-verified read-only filesystem
use an owned read-only mapping. Writable fonts (including custom fonts), other platforms, and mapping
failures use owned byte snapshots. The mapping remains alive with its byte owner;
the OS may reclaim clean file pages without invalidating the cached font. This
assumes the OS system volume is not remounted writable during process execution.
The patches retain `OnceLock` initialization and do not change worker, cancellation,
or freshness transitions. Native file pages remain outside the presentation budget.

The [Chat presentation model](tla/chat-presentation/README.md) checks demand/freshness,
cancellation, and worker bounds. Integration tests exercise viewport-only allocation, progressive
preparation, scrolling eviction, anchor restoration, and streaming prefix reuse.

#### Chat performance checks

`chat_switch_and_scroll_do_not_wait_for_markdown` runs in the normal workspace test/CI
suite. A test-only channel suspends the preparation worker while real GPUI frames, repeated
presentation switches, and wheel input continue. It checks visible plain-text rows, eventual
Markdown publication after resuming, background-executor ownership, one occupied worker slot,
and cumulative preparation starts (including cancelled/evicted work) bounded well below the
1,000-message history. The existing viewport test also bounds preparation starts for one
oversized message, in addition to checking eviction.
The channel and counters are absent from production builds.

Run `just bench-chat` for a release-mode, headless GPUI timing baseline using only Rust's test
harness and `std::time::Instant`. Generated fixtures cover 1,000 rich messages, a single message
with 20,000 paragraphs, and repeated Haskell code/formula tables. Every sample changes the
presentation namespace and discards Chat presentations. Sample 0 is reported separately; the
next 20 samples report nearest-rank p50/p95 with process-wide fonts/libraries warmed.

- `first_frame_ms`: publish an already projected snapshot and synchronously draw source text,
  without advancing the test executor.
- `settle_ms`: subsequently drain the test executor, including Markdown/highlighting/math
  preparation and progressive UI re-layout for the demanded working set.
- `work`: all worker starts during that sample, not just the results still retained at the end.

These timings exclude database loading/projection and native compositor/display latency.
The deterministic executor interleaves UI/background work on its test thread; `settle_ms`
is therefore a harness completion measurement, not production worker-only CPU time. The
benchmark is ignored by ordinary tests. Compare logs from the same machine, toolchain, viewport,
and build profile; no machine-dependent timing threshold gates CI yet. Use the deterministic
test to guard scheduling/demand invariants, and native profiling for end-to-end interaction.

Initial local baseline (2026-09-15): Apple M4 Pro, macOS 26.5.2, rustc 1.97.1,
workspace release profile, 1180 × 720 headless viewport. Times are milliseconds for the
20 warmed samples; these are reference measurements, not CI limits.

| Fixture | Source bytes | First frame p50 / p95 | Settle p50 / p95 |
| --- | ---: | ---: | ---: |
| 1,000 rich messages | 271,890 | 0.721 / 0.939 | 238.236 / 258.528 |
| One 20,000-paragraph message | 620,011 | 1.774 / 2.047 | 14.491 / 15.745 |
| Haskell and formula tables | 10,371 | 0.549 / 0.656 | 122.741 / 127.222 |

### Timeline

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
