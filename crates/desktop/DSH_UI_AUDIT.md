# DSH UI interaction audit

Reference session: `分析并点评项目`, inspected in the running DSH client at
`http://127.0.0.1:3080/` on 2026-08-15.

This is an interaction checklist, not a screenshot similarity claim. A row is complete only when
the GPUI control changes real application state and has been clicked in the built desktop app.

| Surface | DSH control/state observed by clicking | GPUI acceptance |
| --- | --- | --- |
| Sidebar | Collapse leaves a 64 px icon rail with reopen/new/add/search/settings | Keep the rail; the sidebar must always be recoverable, including an empty session |
| Sidebar | Workspace menu lists current workspaces and `Add workspace…` | Workspace selector opens the registered project list and switches the active project |
| Sidebar | Search expands inline; view menu selects grouping and recency ordering | Search and ordering change the visible session rows |
| Header | Preset menu shows Standard/Code/Minimal/Creator with descriptions | Omit the preset affordance because this agent has no preset capability |
| Header | Background jobs opens a status popover | Omit until the core exposes background jobs; do not render a fake count |
| Header | Session log downloads JSONL | Native save dialog copies the current JSONL session |
| Chat | User and assistant copy icons change to copied state | Copy buttons remain functional |
| Chat | Good/bad rating toggles and exposes an optional note editor | Good/bad toggles are functional for the active view; note persistence remains unavailable without a feedback port |
| Chat | Branch creates a new session at the selected assistant response | Defer until the core exposes transcript branching |
| Chat | Context, Think and tool rows expand/collapse | Persisted and live reasoning is represented as a real expandable Think row; context notices and tool rows keep distinct roles |
| Chat | Tool `Inspect` switches to Trajectory, selects the call and opens Details | Implement this exact cross-view navigation |
| Composer | `+` opens compact/export/feedback/goal/permission/plan/model | Show only working Export/Permission/Model commands |
| Composer | Permission menu offers Read Only/Workspace Write/Full access | Use honest Ask/Allow-all modes supported by this core; never label an unsandboxed shell Workspace Write |
| Composer | Model menu separates model and effort without leaving the composer | Model/effort selector changes the live `Agent` configuration and persists per model |
| Trajectory | Duration toggles equal-width vs recorded-duration timeline | Toggle normalized vs recorded timing when timing exists; disabled explanation otherwise |
| Trajectory | Turns collapse changed 87 rows to 7 summary rows | Collapse each user turn to its user row plus a real step/call summary |
| Trajectory | Calls collapse changed 87 rows to 70 rows | Collapse tool calls under assistant activity without losing the call count |
| Trajectory | Search filters after a short debounce and highlights matching blocks | Filter role, tool name, payload and result; keep the overview selection in sync |
| Trajectory | Three-lane Input/Model/Tools overview; blocks select/focus rows | Render three lanes, distribute blocks over the full sequence/timing range, and make every block select its ledger record |
| Trajectory | Selected row has blue outline and a numbered turn marker | Keep row and overview selection synchronized |
| Details | Summary shows hierarchy, status, payload, result, schema and timing preview | Implement the real selected-record values |
| Details | Payload tab shows parsed call JSON | Preserve tool arguments separately from tool output |
| Details | Result tab shows raw tool output | Preserve and render the full result independently |
| Details | Schema tab shows tool description/parameters | Show available core schema metadata or an honest unavailable state |
| Details | Timing tab shows started time, duration and source | Show captured live timing; restored records say timing unavailable |
| Settings | General, Models, Plugins and Agent presets pages | Keep functional General/Models pages only; omit plugins and presets this runtime does not provide |

Observed reference counts for the audited session: 3 turns, 32 assistant steps, 37 tool calls,
87 expanded trajectory rows, 70 rows with calls collapsed, and 7 rows with turns collapsed.

## GPUI Computer Use findings

The following failures were found by clicking the built `.app`, not by code inspection:

| Finding | Reproduction | Fix |
| --- | --- | --- |
| An old 07:11 app bundle kept reopening while the 07:56 debug binary was tested | Launch the debug binary while `target/K Castle.app` is still registered | Synchronize the current binary into the app bundle and test one bundle process only |
| Sidebar view menu rendered behind the workspace rows | Click the sliders icon beside Workspaces | Render the menu after the scroll tree, occlude the underlying rows, and close it when another surface opens |
| Composer menus were centered against the entire window | Click `+` with the 280 px sidebar visible | Offset the overlay by the active 280/64 px sidebar width |
| Empty-state composer menu opened near the window bottom | Click `+` before the first message | Anchor the menu immediately above the centered hero composer |
| Restored JSONL hid reasoning that is present in the session | Open `pwd` after the response completed | Preserve reasoning in `TranscriptItem`, stream `ReasoningDelta`, and render expandable Think/model records |
| Timeline blocks were packed at the left edge | Open Trajectory after a real `pwd` tool call | Position records over the full normalized sequence or captured live-time range |
| Settings exposed unsupported plugin/preset navigation | Open Settings and inspect its navigation | Remove unsupported pages instead of presenting unavailable or decorative controls |
| The Timing tab clipped at the 320 px details minimum | Drag the details divider to its rightmost limit | Reduce tab padding so all five tabs remain visible and clickable at minimum width |
| A bottom tool expansion hid its result below the transcript viewport | Expand the last tool row in Chat | Keep the transcript pinned to the bottom when expanding a tool row |

Validated GPUI journeys before the final pass: sidebar collapse/reopen, session search, view menu,
hero Commands/Permission/Model/Workspace menus, Ask/Allow-all persistence, a real `pwd` run with
approval, tool expansion and Inspect navigation, trajectory Turns/Calls collapse, search, details
Summary/Payload/Result/Schema/Timing tabs, and the supported General/Models Settings pages.

Final Computer Use pass completed after the 2026-08-15 refinement: composer model/effort selection
opens in place, Settings contains only General/Models, light/dark themes cover primary surfaces, the
collapsed rail clears the macOS traffic lights, the composer no longer overlaps Trajectory, the
details divider resizes through its range, all five tabs fit, and a long highlighted Markdown result
scrolls independently to its conclusion. A live long response remains pinned to its newest partial
line; upward scrolling wins over subsequent deltas and exposes a fixed Back to bottom control.

The finalized 48-item implementation and release acceptance contract is tracked in
[`DSH_UI_ALIGNMENT_48.md`](DSH_UI_ALIGNMENT_48.md). The supported final product deliberately
contains only General and Models.
