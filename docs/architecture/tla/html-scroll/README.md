# HTML wheel ownership

Contract: [Desktop HTML previews](../../desktop.md#inline-html-previews).

This model checks the **intended wheel-routing protocol**, separately from the
[document lifecycle model](../html-preview/README.md). It does not prove that browser
events, CSS measurements, native hit testing or the implementation refine that protocol.
A green TLC run must not be presented as proof that the macOS scrolling bug is fixed.

## State and rules

Three ordered layers represent a nested widget, the document viewport, and the
transcript. Each has a scroll range of 0–2 and an offset within that range: zero range
represents short content, while the other states cover top, middle and bottom.
Wheel direction is ±1 and magnitude is 1–2, including a delta that overshoots an edge.
Both inline and sidebar placement are explored. Pointer location and direction can
change between every event; offsets persist, so repeated scrolling and reversal are
included rather than only isolated initial-state examples.

Outside input goes straight to the transcript. Inside input tries the widget, then
the document. A layer consumes the entire event only if its offset actually changes.
The remaining delta from an event that reaches an edge is not dispatched again;
the next event can pass to the next layer. If neither inner layer moves, a pending
callback delivers the event once to the transcript for inline previews. A sidebar stops at
its own boundary even if the transcript can scroll. When every eligible layer is at its edge,
completion without movement is valid.

The `phantom` set represents false-positive CSS/geometry hints: an element looks
scrollable but cannot actually move. Such a hint alone cannot consume an event.
This is particularly important for short content and viewport-propagated overflow.
It is a modeled risk, not a diagnosis of a particular browser incident.

## Checked properties

| Property | Requirement |
| --- | --- |
| `TypeOK`, `WithinBounds` | State is well formed and offsets stay within their ranges. |
| `NoDoubleConsumption` | One event changes at most one layer; at most one outer delivery occurs. |
| `SidebarIsolated` | Inside-sidebar input never changes the transcript or enqueues an outer delivery. |
| `OutsidePreservesInner` | Input outside the preview cannot move either inner layer. |
| `InnerFirst` | The nearest inner layer with room wins, in either placement. |
| `NoSwallowedWheel` | If any eligible layer has room, a completed event must move a layer. |
| `CorrectDirection` | Every movement follows the event direction. |
| `MatchesContract` | Final owner and offset equal an independent, priority-based specification, including clamping. |
| `EventSettles` | Every accepted event eventually completes, under weakly fair inner handling and callback delivery. |

TLC checks all reachable states/transitions in these finite bounds, including arbitrary
event sequences and their temporal cycles. It is not a mathematical proof for arbitrary
pixel ranges, arbitrary nesting depth or arbitrary concurrent browser behavior. No symmetry,
state constraint or depth limit prunes the configured state graph. Ordinary TLC deadlock
checking remains enabled. The explicit fairness probe distinguishes a functioning protocol
from an event loop that indefinitely stops servicing a pending event.

## Implementation mapping and assumptions

- `Begin`: a nonzero wheel's dominant axis, after modifier filtering and page cancellation.
  The macOS monitor filters modifiers; `document.js` dispatches its forwarded input as a
  cancelable synthetic wheel before entering the modeled default scrolling policy. The monitor
  selects the current clip for every event and consumes native delivery; `previewWheel`
  reaches the iframe through a trusted host message without requiring WebKit to deliver a native
  DOM wheel. The bootstrap skips its own synthetic event during DOM propagation and only calls
  the default scroll handler once dispatch finishes without cancellation.
  A point in the floating Back to bottom pill's native cutout is outside the preview
  (`inside = FALSE`); the macOS monitor and native mouse hit testing share that exclusion.
  Exact rounded-mask geometry and X11/Windows input-region enforcement are implementation
  assumptions, checked separately from the bounded ownership transitions.
- `TryInner`: walk DOM ancestors, attempt synchronous `scrollBy`, and confirm an offset
  change before returning. CSS overflow is only a candidate-selection hint.
- `Deliver`: the generation-checked `BrowserEvent::Wheel` path in `html_preview.rs`,
  deferred into GPUI after releasing the mutable entity borrow. Only inline events
  may reach this transition; the sidebar has no transcript handoff.
- Zero range corresponds to content fitting its viewport; the 480 px inline cap creates
  a nonzero range for taller content. Precise layout/height convergence is not modeled.

Bounds and pointer location are stable during an event. The model serializes completion
before accepting the next wheel; it assumes FIFO, exactly-once delivery from the live
document. Inline fallback targets the transcript owner instead of re-hit-testing stale
coordinates. Native hit testing before acceptance and JavaScript execution remain assumptions.
Browser/default wheel movement is cancelled, and an inner scroll is synchronous.
Page-consumed events (for example, canvas zoom via `preventDefault()`) do not enter `Begin`;
arbitrary page behavior is outside this scrolling model. JS regressions check element and
window-listener cancellation and exactly one fallback for unhandled native input.
Source replacement, session switching, hidden windows, lost IPC, event coalescing,
momentum phases, smooth scrolling, zoom, horizontal/vertical coupling and RTL offsets
are outside this model. The lifecycle model covers retired document tokens; JS tests
cover dominant-axis choice. Native tests must check cancellation, actual movement,
coordinate mapping and rapid gestures; those assumptions cannot be inferred from TLC.

## Run and sensitivity

```sh
just tla-check html-scroll
just tla-self-test html-scroll
```

Self-tests require counterexamples for consuming a false-positive hint, scrolling both
inner and outer layers, swallowing short-content input, leaking sidebar input into the transcript,
reversing direction, and duplicate callback delivery. Reachability probes require both
upward/downward short-content handoff, document-only consumption and an isolated sidebar boundary while the transcript still has room.
Removing scheduler fairness must produce a non-settling event trace. Each probe checks
TLC's exit code and the named violation, rather than accepting any tool failure.

For native acceptance, place short and long previews in a scrollable transcript. Test
inside/outside movement, both edges and immediate reversal, then repeat in the sidebar
and after an inline preview is partially clipped. Short previews must pass both directions
to the transcript immediately when inline; a short sidebar must leave the transcript unchanged. Repeat with a nested horizontal table and vertical widget.
Repeat in source view: its `pre` is the inner scroller in this abstraction. On Windows/Linux,
the host's DOM wheel listener supplies this entry because source events do not reach the iframe.
On macOS, the native monitor consumes the original event before the same host handler runs;
the DOM listener therefore does not deliver a second copy. The JS regression exercises both
entries, short-source handoff and sidebar boundaries; TLC still assumes delivery to the handler.

## Native regression (2026-09-18)

The release trace recorded repeated nonzero AppKit wheels inside the visible preview with no
corresponding DOM handler entries across multiple gestures (gaps of about 6 and 11 seconds).
The old model started after DOM delivery and therefore could not detect that failure.
The implementation now routes at the native window monitor, independently of browser gesture
latching. The JS regression explicitly sends native host messages without firing any DOM wheel.
This removes that delivery dependency; it still does not make TLC a proof of WebKit execution.

### Outside-preview regression

The native monitor must send outside wheels to the GPUI view that directly owns `ClipView`,
not to `NSWindow.contentView`. In gpui-pre-macos 0.3.3 the latter is a plain AppKit wrapper;
GPUI installs its `scrollWheel:` handler on a child view. Consuming the event after calling
that wrapper drops transcript scrolling. Headless tests do not create this native hierarchy
and therefore cannot detect the wrong Objective-C receiver. The `outside` fault probe
checks the corresponding swallowed-input violation, not the native receiver itself.

Repeatable native check: open the HTML demo, move the pointer onto paragraph text outside
all previews, scroll down and up while the transcript has room in both directions. Then
open a plain-text session and repeat (the window monitor remains installed). Both must
scroll. Return to the HTML session and repeat over short/long previews and sidebar boundaries;
inside HTML must work and the sidebar must not move the transcript.

Native verification on 2026-09-18 used the packaged release build: outside-preview paragraph
scrolling moved the transcript up and down, including crossing the short motion preview;
after switching to the KV Cache text session, scrolling up from section 9 reached section 8,
and reversing returned to section 9. The original HTML session was restored afterwards.
