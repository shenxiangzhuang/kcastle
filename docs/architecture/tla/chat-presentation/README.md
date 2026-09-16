# Chat presentation working set

Contract: [Desktop ownership and performance](../../desktop.md#chat-viewport).

`ChatPresentation.tla` models two source blocks, three freshness generations, a
changing viewport, one non-preemptible worker slot, cancellation, and result publication.
A generation abstracts session namespace, projection lineage, content revision, and
theme. In Rust, an unchanged source fragment keeps its revision during streaming.

- `CurrentOnly`: published results belong to current demand and generation.
- `BoundedWorker`: cancellation never frees a running worker slot prematurely.
- `BoundedReady`: retained presentations cannot outgrow demand.
- `CancelledSettles`: with weak fairness of completion, cancelled work eventually exits.

Mapping: `ChatViewport::sync/activate/release` invalidate state; list callbacks collect
`requested`; `finish_chat_frame` prunes and starts work; `current_result` guards publication.
`FinishIndex` represents semantic range-index publication: after the same freshness check,
rows are replaced and demand is collected again. Rust additionally checks the whole-message
revision before this operation, since unchanged fragment revisions alone cannot authorize an
index for an older message. Shared code-block preparations use the existing completion path.
The Rust completion path also calls the existing demand scheduler directly, so cached or
throttled native draws cannot stall queued preparation. This is `Finish` followed by `Start`,
not an additional worker slot. `Task` ownership retains the slot until completion. SVG leases follow presentation lifetime.
Decoded formula Emoji images share that lifetime and count toward the Rust byte budget;
their preparation does not add another worker or change publication/cancellation transitions.

Assumptions: a worker eventually returns; cancellation is cooperative between CPU stages.
The model abstracts GPUI layout, Markdown semantics, pixel anchors, byte budgets, session I/O,
and asset reclamation. Process-wide font owners (owned snapshots or immutable system-file
mappings) are also outside this model: font mapping changes storage, not worker demand,
cancellation, publication, or SVG lease transitions. Mapping safety assumes the verified
read-only macOS system volume remains read-only for the process lifetime.
Rust tests cover source partitioning, anchors, progressive rendering,
working-set eviction, and SVG lifetime. This finite model is not a refinement proof.

`self-test` deliberately allows stale publication, omits eviction, and admits a second worker;
each mutation must violate its invariant. A false `NoReady` invariant confirms publication is reachable.
