# Chat presentation working set and shared cache

Contract: [Desktop ownership and performance](../../desktop.md#chat-viewport).

`ChatPresentation.tla` models two sessions, two source blocks, three freshness
versions, visible rows within a larger preparation demand, a shared cache, one non-preemptible worker slot,
cancellation and publication. Switching sessions drops active demand but preserves
reusable entries; a return can reuse only the matching session and version.
A version abstracts projection lineage, source revision and theme. Rust additionally
keys by source field/range; unchanged streaming fragments keep their revision.

- `CurrentOnly`: active results belong to current demand, session and version.
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
`FinishIndex` abstracts semantic range-index publication: rows are replaced and
active demand is collected again. Rust also checks the whole-message revision for
index publication, since a stable fragment revision does not identify a whole message.
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
process-wide font owners do not. This finite model is not a refinement proof.

`syntax.rs` now reuses at most four idle, source-free highlighters across rendering
and preparation. Borrowing transfers exclusive ownership under a short mutex; parse,
style resolution and full-source deletion execute outside that mutex. The pool is
scratch state inside the existing CPU stage, not another worker queue or presentation
cache. It neither changes `Start`/`Finish` nor bypasses cancellation/freshness checks,
so the state model is unchanged. Rust tests cover the idle bound, source clearing,
alias reuse, theme/document independence and oversized/cancelled inputs. This model
does not quantify retained grammar/parser memory or cross-window pool contention.

`self-test` deliberately permits stale publication, unbounded caching, eviction of
live results and parallel workers; each must violate its invariant. A false `NoReady`
invariant confirms publication is reachable.
