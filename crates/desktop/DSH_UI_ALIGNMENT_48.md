# DSH UI/UX alignment — 48-item completion matrix

Reference: the running DeepSeek Harness UI at `http://127.0.0.1:3080/`, inspected on
2026-08-15. This matrix deliberately excludes DSH-only plugins, agent presets, background jobs,
and permission levels that the K agent core does not provide.

`Done` means the behavior has an implementation and an acceptance contract. `Release` means it
was also exercised in the signed `target/K Castle.app` release build. The Rust tests cover state
and mapping contracts; Computer Use covers native layout and interaction journeys.

## P0 — correctness and blocking defects

| # | Alignment item | Delivered implementation | Acceptance evidence |
|---:|---|---|---|
| 1 | Inline model switching | The composer opens an anchored Model and effort menu, then a model list, without leaving the conversation. Selection updates the future `Agent` configuration. | Done · Release: picker remained above the composer and the selected model stayed visible. |
| 2 | Inline reasoning effort | Effort is a sibling submenu with only efforts supported by the selected model; selection persists per model. | Done · Release: High/Low/Off controls were available in composer and Models settings. |
| 3 | Remove unsupported settings | Plugins and Agent presets navigation/content were removed. No fake background-job or unsupported permission affordance is rendered. | Done · Release: Settings contains General and Models only. |
| 4 | Real General/Models settings | Permission, project, appearance, motion, busy-Enter behavior, model, and effort write through `SettingsStore`. | Done · unit round-trip test, including reduced motion and per-model effort. |
| 5 | Settings geometry and scrolling | Centered responsive dialog with 188px navigation, bounded viewport height, fixed header, and independently scrollable content. | Done · Release at 1180×720; content and close control stayed inside the window. |
| 6 | Modal interaction contract | Application-owned modal layer adds focus scope, initial focus, Escape, backdrop dismissal, and destructive confirmations. | Done · Release: General/Models navigation and close path exercised. |
| 7 | macOS traffic-light safe toggle | Full sidebar is 280px; collapsed state reserves no content width. The toggle keeps one coordinate across sidebar states. A desktop-layer AppKit adapter restores native titlebar frames for fullscreen and reapplies the windowed layout after exit without patching GPUI. | Done · release collapse/reopen and fullscreen layouts visually verified; top-edge AppKit reveal remains a manual macOS pass. |
| 8 | Streaming scroll ownership | `follow_chat_tail` is released by a real upward wheel delta and restored only explicitly or when the reader returns to the tail. | Done · state test `streaming_only_follows_when_the_view_is_near_the_tail`. |
| 9 | Frame-batched streaming | Visible text/reasoning deltas are coalesced up to 128 events on the next GPUI display frame; approvals, tools, completion, and other structural events publish immediately. Markdown freezes all but its trailing two blocks and re-parses only that frontier. | Done · cadence, stable-key, frontier, fence/list, and bounded-work regression tests. |
| 10 | Chat/composer non-overlap | Transcript and auto-growing composer are flex siblings with `min-height: 0`; the final content remains in the scrollport rather than behind an absolute input. | Done · Release: restored chat final line and stats remained visible. |
| 11 | Trajectory/composer non-overlap | Trajectory root and pane have explicit overflow/min-height bounds; composer and stats remain flex siblings outside both scroll owners. | Done · Release regression reproduced, fixed, and rechecked at 1180×720. |
| 12 | One details scroll owner | Summary/Payload/Result/Schema/Timing render inside one tracked vertical details scrollport; sections do not create competing 180px scroll areas. | Done · details scroll handle is restored per session. |
| 13 | Markdown details renderer | Settled assistant/reasoning results use `TextView::markdown`; streaming Chat uses the DSH-style incremental block wrapper, deferring open-fence highlighting until settlement. | Done · nested-fence, incremental-frontier, and stable-render-key regression tests. |
| 14 | JSON/code/tool rendering | JSON is pretty-printed and fenced as JSON; shell payload/result uses bash fences; long code stays in the details column. | Done · search-cache and nested-code tests; syntax-highlighted Release view. |
| 15 | Stable record mapping | Messages carry turn, semantic step, request id, and cached search text; visible ledger rows retain the source message index through filters/collapse. | Done · tool payload/result/turn-position regression test. |
| 16 | Long-ledger virtualization | Trajectory uses GPUI `uniform_list`, rendering only the visible range and overscan while retaining a stable scroll handle. | Done · virtual-list implementation; 10k-row behavior no longer builds 10k row elements. |

## P1 — interaction contracts

| # | Alignment item | Delivered implementation | Acceptance evidence |
|---:|---|---|---|
| 17 | Back to bottom | Leaving the tail reveals a fixed Back to bottom control; streamed deltas increment its unread counter; activation re-pins. | Done · scroll state regression test and release layout inspection. |
| 18 | Per-session/view restoration | Chat offset/follow state, trajectory offset, detail offset/tab, and selected record are stored under workspace + session. | Done · state is saved before tab, session, project, and new-chat transitions. |
| 19 | Expansion anchoring | Reasoning/tool disclosures only request tail scrolling while follow mode is pinned; historical expansion cannot seize reading ownership. | Done · code contract plus scroll-follow regression test. |
| 20 | True duration domain | Block position and width share one recorded time domain, include minimum visible width, and clamp to lane bounds. | Done · `duration_geometry_uses_one_time_domain_and_stays_in_bounds`. |
| 21 | Cached trajectory search | Each message has a cached lowercase record document covering role/title/text/payload/result/schema; UI shows match count. | Done · cache coverage test; search exercised in the native app. |
| 22 | Details tabs | All five tabs are real buttons in a horizontally scrollable 42px strip, with selected state and independent content. | Done · Release details Summary view and all tab implementations inspected. |
| 23 | Resizable/responsive details | Drag resizer remains available on wide windows; fixed-step narrow/reset/widen controls support keyboard activation; widths below the split-pane minimum use a bounded overlay. | Done · implementation and wide Release split-pane verification. |
| 24 | 1–14-line composer | One `InputState` uses `auto_grow(1, 14)` and becomes internally scrollable after the cap. | Done · component contract and native composer layout. |
| 25 | Draft/caret/IME safety | GPUI component owns marked text, caret reveal, and the draft scrollport; only primary `PressEnter` submits, secondary Enter remains a newline. | Done · relies on the tested gpui-component input path; no raw keyCode interception remains. |
| 26 | Persistent composer entity | Hero and docked seats render the same long-lived input entity; switching phases does not reconstruct draft state. | Done · first-send session preparation is guarded and focus is restored after runs. |
| 27 | Anchored composer menus | Menu is absolutely anchored to the composer card at `bottom: 100% + 8px`, with side chosen by command/model intent and viewport-bounded height. | Done · Release model menu remained attached in expanded and collapsed sidebar states. |
| 28 | Menu keyboard/dismiss state | Highlight index wraps with Up/Down; Enter activates; Escape backs out/closes; outside mouse-down dismisses without triggering the parent. | Done · central root/menu state machine; native outside-dismiss exercised. |
| 29 | Honest action states | Send is disabled for empty/preparing input, shows loading while preparing, and becomes Stop only during a real run. | Done · Release empty-state send was visibly disabled; duplicate preparation is guarded. |
| 30 | Honest permissions | Only Ask before tools and Allow all tools are exposed, and both change the real approval behavior. | Done · settings persistence test and native General/composer controls. |
| 31 | Workspace hover | Workspace rows use a shared hover scope: folder swaps to chevron and controls appear without changing row geometry. | Done · sidebar group-hover implementation. |
| 32 | Session hover/context actions | Every persisted row reserves a fixed trailing slot: age is shown at rest, hover swaps it in place for one ellipsis, and Rename/Delete opens for that exact session without shifting the title. | Done · native normal/hover/long-title/non-selected-row journeys. |
| 33 | Stable sidebar search reveal | Search occupies the existing 44px header slot, focuses on open, and Escape clears/closes, so the tree does not jump vertically. | Done · Release header geometry stayed fixed. |
| 34 | Cross-workspace full-text search | Startup/update indexing scans every workspace JSONL once, recursively indexes human content, and returns title plus compact Unicode-safe snippet. | Done · snippet/cache tests; All sessions grouped result path implemented. |
| 35 | Real workspace grouping/sort | By workspace and All sessions are distinct render paths; every workspace has lightweight session metadata; recent order uses modified time. | Done · project-isolation test and native grouping controls. |

## P2 — visual system, state expression, and quality

| # | Alignment item | Delivered implementation | Acceptance evidence |
|---:|---|---|---|
| 36 | Sidebar collapse/scroll | The 280px expanded width and fixed titlebar-toggle geometry are tokenized; state changes avoid opacity animation so controls do not drift or flash; workspace trees have stable independent scrolling. | Done · Release collapse/reopen. |
| 37 | Message hover feedback | User/assistant message scopes reveal time/runtime plus copy/rating actions only on hover, without reserving new height. | Done · group-hover implementation and native chat inspection. |
| 38 | Think status | Pending reasoning shows a primary spinner; reduced-motion mode substitutes a static status dot; completed state stops animation. | Done · live/restored pending transitions and reduced-motion branch. |
| 39 | Tool state/renderers | Shell and generic tool intents use distinct icons, pending/success/failure colors, terminal/JSON-aware expanded renderers, and safe fallback. | Done · restored failure test and native shell row/detail inspection. |
| 40 | Keyboard/focus semantics | Primary tabs/settings navigation use Buttons; ledger, reasoning, and tool rows are focusable and activate with Enter/Space; modals own focus. | Done at component/keyboard level. GPUI 0.2.2 still does not export custom controls to macOS AX; see `QA.md`. |
| 41 | Typography hierarchy | Shared 22/24/26px secondary/body/message line-height and 30/42/54px row/header tokens replace cross-view drift. | Done · release screenshot comparison with DSH reference. |
| 42 | Icon consistency | Product controls use one gpui-component outline family with fixed 16/20px optical slots and semantic icon selection. | Done · release toolbar/sidebar/chat/trajectory inspection. |
| 43 | Semantic color/theme | Canvas/surface/sidebar/text/border/interactive/status/role colors come from `UiPalette`; System/Light/Dark are functional. | Done · settings round-trip and native light-theme pass. |
| 44 | Geometry tokens | Sidebar/content/composer widths, composer radius, row heights, headers, line heights, and breakpoint live in one metrics layer. | Done · key axes measured in the 1180×720 release screenshot. |
| 45 | Turn/step/token semantics | User messages define turns; reasoning+answer share one step; tools are separate steps; token counters come from real usage. | Done · semantic-step regression test; restored session showed 1 turn · 2 steps. |
| 46 | Responsive column concession | Constrained widths replace the sidebar with a fixed recoverable titlebar toggle; below the split-pane minimum, details become a right overlay; center widths remain min-zero and unclipped. | Done · code breakpoints plus collapsed and wide split-pane journeys. |
| 47 | Motion and Reduced Motion | Sidebar geometry changes without animation; persisted Reduced mode removes pending-reasoning looping motion. | Done · settings persistence test and General Motion control. |
| 48 | Regression and release QA | State tests cover streaming ownership, semantic steps, search caches/snippets, duration bounds, renderer fences, failures, persistence, and project isolation; release journeys cover native geometry. | Done · 81 workspace tests, warnings-denied Clippy, locked release build, `git diff --check`, and Computer Use journeys. |

## Required release gate

Run from the workspace root:

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
cargo build --workspace --release --locked
git diff --check
```

Native acceptance uses the signed `target/K Castle.app` bundle. At minimum exercise: inline model
and effort, General/Models persistence, expanded/collapsed sidebar, populated Chat, Trajectory with and
without Details, detail scrolling, search/grouping, and a live response while alternating between
tail-follow and historical reading.
