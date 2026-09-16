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
or tail-follow mode. Visible rows plus approximately 600 px of overscan in each direction
form the preparation working set. Native layout callbacks are not the source of truth for
demand: GPUI can reuse an offscreen row's measured height without rendering it again.
After layout, `ChatViewport::refresh_demand` adds geometric neighbours explicitly. Above
the scroll top GPUI exposes no item bounds, so this uses recently observed heights or a
bounded source-line estimate. Selection/layout entities stay local to this working set;
history keeps cheap source locators, not one parsed document per message.

Rows first display source text. A single background task prepares Markdown ASTs, code syntax
styles, and formula layout/SVGs. GPUI element creation, text shaping, and painting remain on the
UI thread, bounded by the working set. Completion remeasures the affected row; native list
anchoring preserves the top item's offset while heights above it change. Width changes use
native proportional remeasurement. Switching away discards GPUI presentations and cancels
their preparation demand, but retains reusable results in one cache shared by sessions in
the window. Scroll anchors and expansion/rating interactions survive separately. These are
transient UI state, never journal facts. Inactive sessions can still receive and persist
runtime events; they do not start Chat parsing, highlighting or layout work.

Work is tagged with session/projection epoch, source-fragment revision, and theme. Publication
also requires current viewport demand and an uncancelled task. Offscreen work is cancelled;
the worker slot stays occupied until it returns, including after repeated session switches.
Cancellation is cooperative between parsing, highlighter construction, and formula operations.
Completion starts the next already-demanded block directly; queue progress does not require
another native draw callback, which can be throttled for occluded windows. An individual library
call cannot be interrupted. Unchanged fragments retain their presentation
revision during streaming; native list splices retain unaffected measurements.

The initial source scan supplies cheap plain-text placeholders. The same worker then parses
real top-level Markdown blocks and publishes a range-only semantic index before preparing rich
rows. Lists (including loose/nested items), quotes and tables retain their block boundaries;
inter-block spacing is computed from adjacent AST nodes, including standalone strong lead-ins
followed by lists. Index publication checks the whole message revision as well as the normal
fragment/epoch/demand checks, splices rows, and restores the source-byte scroll anchor.
Once rich content is displayed, append-only updates within the indexing limit keep its complete
source/row/presentation snapshot on screen. The same worker indexes the new source and prepares
replacement blocks overlapping the demanded source range before publishing both together.
Unchanged blocks reuse their cached preparation; code slices share one preparation. The pending
batch retains at most another 8 MiB of newly prepared data, subject to the existing per-block
limits; admission still uses the shared cache budget and its readable-source fallback.
Continued appends do not cancel this worker: a completed snapshot may advance the display if it
is still a prefix of the latest source, then one new worker coalesces the remaining appends.
This prevents fast streams from starving presentation. Rewrites, session/lineage/theme changes
and leaving demand still reject stale work. Source locators, selection input and rich content
always refer to the same displayed snapshot. Cold or unprepared content retains the initial
plain-text path and completed semantic prefix reuse. Reindexing preserves matching row revisions.
Expanding or collapsing reasoning/tool output retains unchanged assistant indices and
presentations; overlay-only changes cannot redefine their cached source fragments.
Large paragraph-only messages use the existing scan without a global AST, but only after ruling
out container openers, indentation, setext headings and tables. Mixed documents up to 1 MiB
still require one whole-message background parse; their initial plain frame remains independent
of parsing. Larger mixed messages keep bounded, lossless plain-text fragments rather than starting
an unbounded whole-message parse. Logical code blocks above 256 KiB also stay readable source;
the limit is checked before cloning/parsing the entire code block for highlighting.

Code rows preserve the opening fence's indentation so reparsing retains the indexed code offsets.
They are slices of one logical block: they share the whole block's prepared AST/highlights,
show one header, copy the complete code, and round only the outer corners. The working-set budget
counts shared allocations once. Other non-code blocks above 16 KiB retain the lossless plain-text
fallback. Messages up to 2 KiB with reference definitions remain a single preparation unit,
preserving link/image references, including definitions inside containers. Cross-block reference
definitions in larger messages remain unsupported by independently prepared prose.
These are explicit bounded-rendering limits. Source scanning still runs on the UI thread; journal
loading/runtime projection retain their existing ownership and are not database-page virtualization.

Ordinary paragraphs are shaped as whole physical lines and wrapped at Unicode line-break
opportunities, keeping inline code together when it fits. They create one native text element
per visual line instead of per word/character. Selection stores original logical byte ranges and
projects visual line segments onto them, preserving copy and selection across reflow. Inline
formula SVGs retain the existing baseline-aware mixed-object flow. Body text is 16/26 px;
section/heading-following gaps are 24/8 px and tight/loose list-item gaps are 6/12 px.
Code containers keep the normal 16 px block separation even after headings, balanced with
following prose. Semantic partitioning excludes inter-block blank lines and preserves the
original fenced source, so those separators cannot become empty rows or extra code lines.

Prepared data and reusable semantic indices share an estimated 8 MiB cache budget across all
sessions in the window. This is not 8 MiB per session and not a process RSS limit. Cache keys
include namespace, projection lineage, message/fragment revision, source field/range, result
kind and (for prepared Markdown) theme. Whole-block code allocations are charged once across
their slices. Index vectors and cache-entry/key metadata are charged as well; AST sizing is
still estimated. Prose results above 1 MiB are not admitted; logical code may use the overall
budget. Allocation during preparation and current-frame leases can exceed the retained estimate.

Admission evicts non-demanded entries first, then nearby entries for visible work, using least
recent use within each priority. It never evicts a result still held by a visible presentation
or current frame. Equal-priority demanded entries do not evict each other and repeatedly reparse.
If admission cannot fit, the fragment stays readable source for that demand interval. GPU/layout
entities are never stored in the shared cache. Leaving the viewport releases their references;
only reusable ASTs, syntax spans, formula images/SVG leases and indices may survive.

Unused entries expire after five minutes, swept every 30 seconds even without input. After five
minutes without Chat frames, offscreen presentation references are released too; currently
visible content remains ready while the user reads. The sweep starts no preparation. Switching
away clears active protection immediately. Both the budget and expiry are initial internal
defaults, not user settings; tune them from the repeatable browsing benchmark below.
Formula vectors use GPUI's themed SVG alpha mask. RaTeX's embedded raster glyphs (color Emoji)
are excluded from that mask and decoded once on the preparation worker into a transparent
color layer sharing the vector layer's viewBox and coordinates. Bounds expand to include color
strikes that exceed RaTeX's fallback ascent/descent, preserving the baseline's position relative
to both layers. The presentation owns its `RenderImage`
directly and includes the decoded BGRA bytes in the budget; it does not enter the global image
resource cache. Theme changes still tint vector symbols without recoloring Emoji.
This budget does not include canonical session data, native GPU/font resources, or the separate
Trajectory details renderer. The shared Chat cache is window-owned, not process-global.

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
preparation, scrolling eviction, anchor restoration, and streaming prefix reuse. The streaming
publication regression suspends the real preparation worker between frames and compares native
list row bounds before/during/after three non-wrapping character appends. A completed 18-line
Rust block plus trailing paragraph previously moved 49 px and reverted to plain text on every
append; the regression requires unchanged geometry, rich intermediate frames and one worker per
append. A second test checks progress during continued appends and rejection after rewrite/switch.
These are GPUI headless layout checks, not a native display frame-time measurement.

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
with 20,000 paragraphs, repeated Haskell code/formula tables, and Chinese/English prose with
bold text and inline code. Every sample changes the
presentation namespace and explicitly clears the shared cache. Sample 0 is reported separately; the
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

Typography acceptance baseline before shared caching (2026-09-16): Apple M4 Pro, macOS 26.5.2, rustc 1.97.1,
workspace release profile, 1180 × 720 headless viewport. Times are milliseconds for the
20 warmed samples; these are reference measurements, not CI limits.

| Fixture | Source bytes | First frame p50 / p95 | Settle p50 / p95 |
| --- | ---: | ---: | ---: |
| 1,000 rich messages | 271,890 | 0.658 / 0.670 | 125.167 / 126.269 |
| One 20,000-paragraph message | 620,011 | 2.383 / 2.536 | 16.664 / 17.130 |
| Haskell and formula tables | 10,371 | 0.564 / 0.606 | 124.422 / 125.258 |
| Chinese/English prose | 7,331 | 0.440 / 0.509 | 3.437 / 3.612 |

#### Syntax highlighter reuse

Code preparation and direct settled-code rendering share a process-wide pool of at
most four idle `SyntaxHighlighter` objects (`syntax.rs`). Each task takes exclusive
ownership; the mutex covers only taking/returning objects, never parsing or style
resolution. An unavailable language object is constructed outside the lock, so UI
work never waits for a background parse. Entries match the complete registered
language configuration, including aliases; styles are resolved against the caller's
theme each time. A complete source deletion clears text and host/injection trees
before return. Host-language queries are reused; injected-language query lifetimes
remain managed by the upstream highlighter. Inputs above 256 KiB are not returned, limiting retained parser scratch
space. Unknown/plain languages do not occupy slots. The four-entry bound applies to
idle objects, while borrowed objects follow existing rendering/worker concurrency.
This source-free grammar/parser pool is separate from the 8 MiB presentation budget;
it stores neither sessions nor highlighted outputs. Cancelled work still passes the
existing freshness gate before publication.

#### Shared-cache comparison

Run `just bench-chat-cache 8`, `just bench-chat-cache 16`, and `just bench-chat-cache 32`.
The budget override exists only in this ignored test, not in production configuration.
The same fixture has three sessions of 100 messages, each containing 400 lines of Rust.
Each pass performs 39 operations: session switches and jumps to twelve positions per
session. The second pass revisits positions in reverse order. It reports preparation
starts (including indices), settle p50/p95, peak retained estimate, and retained bytes
after six minutes of simulated inactivity. As above, settle time is headless executor
drain time, not native wheel-to-display latency or a frame-time measurement.

Initial local comparison (2026-09-16, release, 1180 × 720; one run per budget):

| Budget | Revisit starts | Revisit settle p50 / p95 (ms) | Peak cache estimate (MiB) | Process peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 8 MiB | 166 | 34.278 / 36.141 | 7.84 | 62.17 |
| 16 MiB | 131 | 32.310 / 34.945 | 15.69 | 80.61 |
| 32 MiB | 98 | 17.322 / 35.734 | 31.35 | 87.62 |

All three retained approximately 0.98 MiB after idle expiry, keeping the visible code.
RSS was measured with `/usr/bin/time -l` around the already-built release **test binary**
in separate processes; it includes fixtures/fonts/native resources, excludes compilation,
and is not the desktop application's memory footprint. The production default stays at
8 MiB: larger caches help this code-heavy revisit workload, but these single-run results
do not establish a general latency/memory optimum. Native profiling and repeated runs
on representative sessions should precede a default change.

Regression tests cover warm-session first-frame reuse, geometry-driven overscan across
redraws and wheel input, idle expiry without interaction, visible retention while idle,
cross-session LRU/admission priorities, shared allocation accounting, freshness keys and
pre-parse source limits. Existing large-history and single-worker tests remain in CI.
Manual acceptance: use a release app, scroll several screens through code and back, switch
A → B → A at the saved reading positions, then repeat after more than five minutes away.
Confirm stable reading position and reduced plain-to-rich transitions on warm content;
profile native frame times and process memory separately from the headless checks.

The [2026-09-16 native profile](chat-native-profile-2026-09-16.md) records real macOS
wheel input, A → B → A restoration, stack sampling and process memory. It confirms
lower CPU cost on return scrolling in one paired run, while identifying repeated
highlighter construction and native text/layout work. Its screenshots do not establish
display-frame latency or the absence of transient highlighting flashes.

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
