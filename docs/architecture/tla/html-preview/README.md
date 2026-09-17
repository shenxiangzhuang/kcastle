# Inline HTML previews

Contract: [Desktop HTML previews](../../desktop.md#inline-html-previews).

Two documents and two session namespaces model independent live document generations, source replacement,
viewport visibility, exclusive enlargement, native overlay occlusion and asynchronous height/ready notifications.
A source replacement or session change rejects older document tokens. Hiding a document retains
its runtime; mounting two documents does not evict either one. `Frame` abstracts a completed
GPUI paint, whose content masks provide each browser's visible rectangle. An enlarged document
is the only displayed native preview, independent of its original row visibility; covering app
overlays hide it too. Session switching clears enlargement without retaining the old owner.

`CurrentOnly` checks namespace/source freshness, `ClippedVisibility` checks that hidden or
covered documents cannot remain native overlays, and `HiddenStateRetained` checks that mere
visibility changes cannot destroy a running document. The sensitivity test accepts a stale
callback deliberately; the reachability test requires two simultaneously displayed previews.

Mapping: `HtmlPreviews::sync` performs namespace reset; `render` replaces changed sources;
`PreviewFrame::paint` collects placements and updates browser visibility; the event task checks
monotonically increasing document generations before applying callbacks.
The native WebView is reused for append-only source changes, while `Rewrite`/`Frame` describe the
logical document retirement and publication inside it. Native allocation cost and reuse are
implementation details outside this model; the Rust streaming regression checks retained layout
and preview state, and the host attaches document generations to IPC callbacks after each reload.

Bounds/assumptions: one source replacement per page, one namespace switch, finite delivery queue,
and atomic frame publication. Browser process isolation, CSP enforcement, WebKit/WebView2
behavior, pixel clipping, native cursor ownership/glyphs, wheel routing, height convergence, image encoding/save-dialog cancellation, performance and memory are outside
TLC; Rust tests and the documented native fixture cover those implementation boundaries.
