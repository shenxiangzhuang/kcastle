# Inline HTML previews

Contract: [Desktop HTML previews](../../desktop.md#inline-html-previews).

Two documents and two session namespaces model independent live document generations, source replacement,
viewport visibility, a right preview sidebar, native overlay occlusion and asynchronous height/ready notifications.
A source replacement or session change rejects older document tokens. Hiding a document retains
its runtime; mounting two documents does not evict either one. `Frame` abstracts a completed
GPUI paint, whose content masks provide each browser's visible rectangle. The selected document
mounts a separate instance in the sidebar independently of its original row visibility;
the original inline instance remains displayed too. Tokens include an inline/sidebar site. Covering app overlays hide both. Session switching clears sidebar selection.

`CurrentOnly` checks namespace/source freshness, `ClippedVisibility` checks that hidden or
covered documents cannot remain native overlays, `MountedDocumentsVisible` requires both sidebar
and inline mounts to be displayed, and `HiddenStateRetained` checks that mere visibility changes
cannot destroy a running inline document; closing/replacing the sidebar releases its copy. Sensitivity tests accept a stale callback and restore exclusive
enlargement deliberately. Reachability tests require two simultaneously displayed previews,
including the same document displayed inline and in the sidebar at once.
The `sidebar-source` fault keeps the sidebar on an obsolete source while its inline row is hidden;
`CurrentOnly` rejects it. A reachability check requires an updated sidebar with no inline mount.

Mapping: `HtmlPreviews::sync` performs namespace reset; `render` replaces changed sources;
`PreviewFrame::paint` collects placements and updates browser visibility; the event task checks
monotonically increasing document generations before applying callbacks.
Visibility publication includes both WKWebView and its retained macOS clip container. The latter
must stop participating in native hit testing when hidden, otherwise it can block a displayed
sibling while contributing no pixels. `displayed` abstracts the complete native attachment, not
just the browser child. The model does not execute AppKit hit testing; the desktop native
acceptance steps cover scroll-away/remount and overlay hide/show of retained containers.
The native WebView is reused for append-only source changes, while `Rewrite`/`Frame` describe the
logical document retirement and publication inside it. Native allocation cost and reuse are
implementation details outside this model; the Rust streaming regression checks retained layout
and preview state, and the host attaches document generations to IPC callbacks after each reload.
The selected sidebar has its own bounded background preparation, independent of chat visibility.
`Frame` abstracts successful preparation/publication; worker scheduling and cancellation are outside
this model. Rust checks the worker's sidebar generation, namespace, lineage and append ancestry before
publication, and dropping the sidebar cancels its owning task.

The floating Back to bottom pill is a partial native cutout, not a covering app overlay:
its preview remains mounted/displayed and retains its viewport. `Frame` abstracts publication
of that geometry; rounded masks and platform hit testing remain outside this lifecycle model.
Rust checks an overlapping button's cutout, surrounding native pixels/input, unchanged layout,
and removal after clicking. Native acceptance checks actual visibility and click-through.

Bounds/assumptions: one source replacement per page, one namespace switch, at most two queued callbacks (delivered in either order),
and atomic frame publication. Browser process isolation, CSP enforcement, WebKit/WebView2
behavior, pixel clipping, native cursor ownership/glyphs, wheel routing, height convergence, image encoding/save-dialog cancellation, performance and memory are outside
TLC; Rust tests and the documented native fixture cover those implementation boundaries.
The bounded lifecycle model distinguishes inline and sidebar tokens and rejects retired source
and namespace callbacks. Reopening an unchanged sidebar's allocation generation is abstracted;
Rust assigns a fresh monotonic generation on every sidebar creation, and tests rejection of the
old sidebar's height/action callbacks after replacement. JS tests cover native host message routing
without DOM wheel delivery and sidebar isolation. Rust checks the 480 px inline cap, simultaneous
inline/sidebar mounts and that sidebar boundary dispatch leaves the transcript unchanged.
The [wheel ownership model](../html-scroll/README.md) separately checks the intended routing
protocol, including short inline content and non-moving scroll candidates; it does not establish
native browser/GPUI refinement. The native fixture checks actual scroll routing and long-table geometry.
