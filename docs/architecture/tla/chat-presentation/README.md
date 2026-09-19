# Chat presentation working set and shared cache

Contract: [Desktop ownership and performance](../../desktop.md#chat-viewport).

`ChatPresentation.tla` models two sessions, two source blocks, three freshness
versions, visible rows within a larger preparation demand, a shared cache, one non-preemptible worker slot,
cancellation and publication. Switching sessions drops active demand but preserves
reusable entries; a return can reuse only the matching session and version.
A version abstracts projection lineage, source revision and theme. `generation` identifies the
displayed snapshot; `target` identifies the latest source. `Append` advances only the target,
preserving rich content until `FinishUpdate` atomically installs the prepared replacement.
Continued appends may leave work behind the target: a completed append snapshot still advances
the display, then the next job coalesces the remaining input. `Invalidate` covers non-append
rewrite/lineage/theme changes and still cancels work. Rust additionally keys by source field/range;
unchanged streaming fragments keep their revision.

- `CurrentOnly`: active results belong to current demand, session and displayed version.
- `DisplayedVersion`: display cannot advance beyond the latest source.
- `NoIntermediateDowngrade`: advancing the source alone cannot erase the displayed presentation.
- `BoundedWorker`: cancellation never frees a running worker slot prematurely.
- `BoundedReady`: protected presentations cannot outgrow visible demand.
- `BoundedCache`: all sessions share one capacity, including inactive results.
- `CachedReady`: eviction cannot drop a currently used result.
- `CancelledSettles`: with weak fairness of completion, cancelled work eventually exits.

Mapping: `ChatViewport::refresh_demand` adds geometry-based neighbours to native
render callbacks, independently of GPUI's cached measurements. `activate/release`
clear UI entities/demand and cancel work, while `PreparationCache` keeps reusable
results. `sync` can restore a cached semantic index and `row` can reuse prepared
Markdown. `current_result` checks worker freshness before either kind is admitted.
`FinishIndex` abstracts cold semantic range-index publication: rows are replaced and active
demand is collected again. `FinishUpdate` instead replaces the demanded index and preparation
together. Rust checks the whole-message revision for cold index publication; for an already
displayed append snapshot it also permits publication when the prepared source is still a prefix
of the latest source. A stable fragment revision alone does not identify a whole message.
The completion path drains the existing queue directly (`Finish` followed by `Start`),
without requiring another native draw or freeing the slot early on cancellation.

The model uses equal-size capacity units. `Admitted` can evict reusable entries or
refuse admission when active entries occupy the capacity; `Evict` abstracts TTL and
memory reclamation between operations. It does not specify exact LRU order or elapsed
time. Rust tests exercise byte accounting, visible/nearby priority, LRU across sessions,
shared code allocation, freshness keys, timer-driven expiry without input and visible
content protection while idle. The five-minute TTL is an initial policy parameter,
not a temporal theorem in this model.

Assumptions: a worker eventually returns; cancellation is cooperative between CPU
stages. GPUI layout, Markdown semantics, source limits, pixel anchors, session I/O,
transient allocations and native GPU/font resources are outside the model. SVG and
formula-image leases follow cached preparation lifetime and count in the Rust estimate;
process-wide font owners do not. The atomic-update transition abstracts a batch that fits the
budget; Rust additionally bounds new batch allocations and retains the existing plain fallback
when preparation or admission fails. The model does not prove geometry stability or performance
under unlimited source arrival. GPUI tests check intermediate geometry and advancement to a
completed prefix while the next append job is suspended. This finite model is not a refinement proof.

`syntax.rs` now reuses at most four idle, source-free highlighters across rendering
and preparation. Borrowing transfers exclusive ownership under a short mutex; parse,
style resolution and full-source deletion execute outside that mutex. The pool is
scratch state inside the existing CPU stage, not another worker queue or presentation
cache. It neither changes `Start`/`Finish` nor bypasses cancellation/freshness checks,
so the state model is unchanged. Rust tests cover the idle bound, source clearing,
alias reuse, theme/document independence and oversized/cancelled inputs. This model
does not quantify retained grammar/parser memory or cross-window pool contention.

`self-test` deliberately permits stale publication, unbounded caching, eviction of
live results and parallel workers; each must violate its invariant. The `flash` fault clears
ready content on append and must violate `NoIntermediateDowngrade`. A false `NoReady` invariant
confirms publication is reachable.

GPUI Kit 0.6.4 acceptance retained the shared Chat/Trajectory Markdown renderer:
TextView did not preserve list numbering and copy semantics in the migration probe.
Chat publication, demand, cancellation and cache transitions are unchanged; this
upgrade requires no model action. See the [acceptance record](../../../development/gpui-kit-0.6.4-validation.md).
