# GPUI Kit 0.6.4 — full macOS regression, 2026-09-19

## Result and scope

The exercised native paths show no new regression between `3083dbd` (before the
upgrade, GPUI Kit 0.6.0) and `b4435cb833c2421e7ecbf38e39f195f25da37988` (0.6.4,
framework menus/dialogs, shared Chat/Trajectory Markdown renderer).

This extends the [initial acceptance report](gpui-kit-0.6.4-validation.md). It is a
macOS regression result, not a claim that the product has no defects: the known
observations below remain. Windows/Linux, other display scales, and every possible
transient frame were not validated here. No production code was changed in this run.

## Environment and reproducibility

- macOS 26.6.2 (25G83), arm64; native GPUI windows at 1180 × 720 and 720 × 720.
- Separate signed local app bundles, data directories and SQLite databases for the
  current and baseline builds. Real user sessions/settings were not touched.
- Current executable saved before compiling the baseline with the shared Cargo target
  directory. The baseline was built in a detached worktree at `3083dbd` with `--locked`.
- Synthetic sessions were created through `kcastle_agent::Agent` and the public
  Session APIs, persisted as real journals, then reopened with `validate_events`.
  History A: 120 exchanges / 1440 events; History B: 100 exchanges / 1200 events;
  Visual details: one exchange / 12 events. Baseline and current started from copies
  of these same journals.
- A local Responses SSE server at `127.0.0.1:18464` served deterministic Markdown and
  HTML fixtures. A dummy key was used; no external model service was called.
- Live `STREAM` responses emitted 18-character deltas with pauses after the heading,
  numbered list, 45-line code block, formula table/display math, and final sentinel.
  File gates allowed observation before/after each phase without racing a real model.

Local evidence is retained in `target/gpui-kit-full-validation/` (ignored): executable
copies, build/check logs, screenshots, `server.py`, and `native_validation_seed.rs`.
Runtime data and the two app bundles are under `/tmp/kcastle-full-validation/`.
The seed example was temporary and is not part of the production source tree.
The server imports the checked-in Markdown/HTML fixtures. To repeat live phases,
send `STREAM <label>` to the local provider and create
`/tmp/kcastle-full-validation/STREAM-<label>.continue-N` for stages 0 through 4.

Saved executable SHA-256 values, before app-bundle signing:

```text
b4435cb  783414fa8de51f72223b2e5e7fead4a4fab0dc2757fc9569b2d575435e35b368
3083dbd  8f09220a6a1bd724f574b9f8443e9dff3f1251990c21bcd0adaaf4c312a67dfc
```

## Native visual and interaction matrix

| Area | Actual operation and observed result | Evidence |
| --- | --- | --- |
| Real Chat | Loaded saved fixture; checked CJK/English, styled text, emoji/math, 98–100 list, checkboxes, table, code and final sentinel | `current-chat-top.png`, `current-restart-prepared.png` |
| Real Trajectory | Selected persisted assistant event; Summary → Preview → Raw; resized details from 342px to 559px; scrolled through final sentinel | `current-trajectory-preview-{top,bottom}.png` |
| Trajectory baseline | Repeated the same event, panel width and scroll on the pre-upgrade executable; visual layout/content match | `baseline-trajectory-preview-{top,bottom}.png` |
| Actual selection/copy | Drag-selected numbered list in the dark, narrow Trajectory Preview; Cmd-C then Cmd-V into composer produced `98. first\n99. second\n100. third` | `current-trajectory-list-copy.png` |
| Long history | Opened A at turn 119; scrolled to turns 106/107. Opened B at turn 99; scrolled to 93/94. A → B → A restored A's content and vertical position | `current-history-{a-anchor,b-anchor,a-restored}.png` |
| Horizontal code wheel | Native horizontal wheel visibly shifted a long-code segment in both executables; this closes the previous lack of any observable native offset, subject to the limitation below | `current-history-horizontal-{before,after,end}.png`, `baseline-history-end.png` |
| Live Markdown | Heading/list/formula/code rendered while response was still Running; real Trajectory Preview showed Pending, then Completed | `current-stream-trajectory.png`, `current-stream-complete.png` |
| Reading while streaming | Scrolled away before final table/math/text deltas; earlier text stayed at the same position and “8 new” appeared. Returned to tail and saw the complete final content | `current-stream-detached-{before,after}.png`, `current-stream-complete.png` |
| Switching while running | Switched to History B and back during a paused response; background stream completed, original response remained intact, Running/stop controls settled | `current-stream-complete.png`; unread-counter caveat below |
| Cancellation/retry | Stopped a second response after its first phase. Stop immediately became Send; a third prompt completed. Trajectory retained the partial cancelled response, marked Failed | `current-stream-cancelled.png`, `current-dark-narrow-cancel-trajectory.png` |
| Model/effort menus | Mouse selected Fixture B; typed into composer immediately after dismissal. Keyboard selected Fixture A and High effort. Submenus opened to the left near the right edge, and Left entered them | `current-dark-narrow-menu.png`; selected values in `current-restart-prepared.png` |
| Dialogs | Settings retained AX Dialog role; Escape closed it. Rename focused its input and Enter saved `Validated stream`. Archive → delete-confirmation → Cancel returned to archive settings; Restore brought the synthetic session back | `current-rename-confirmed.png`, `current-delete-dialog-cancel-only.png` |
| Theme/resize | Light and dark views; minimum width auto-collapsed sidebar. Settings General/Models and menu contents remained reachable; resizing back restored wide layout | `current-dark-narrow-settings.png`, `current-dark-narrow-menu.png`, prior report's light/narrow captures |
| Native HTML overlays | Loaded real HTML response, moved speed 30 → 54. Command menu and Settings hid WKWebViews; closing them restored speed 54 and the independent probability 65% | `current-html-{menu-hidden,dialog-hidden,state-restored}.png` |
| Restart | Terminated/relaunched the isolated process, reopened renamed session. All three turns, partial cancellation, final sentinel, dark appearance and session High effort persisted. Background preparation replaced the initial raw-text fallback with formatted Markdown | `current-restart-{restored,prepared}.png` |
| Release smoke | Launched the optimized build against the same isolated data. Chat final content, real Trajectory Preview and framework model menu rendered correctly | `release-{chat,trajectory,menu}-smoke.png` |

The native screenshots were inspected during the run. This was not a pixel-diff test;
timestamps, pointer position and other incidental window state differ. Automated
selection/layout tests supplement the actual native interactions.

## Existing issues, distinguished from upgrade regressions

### Unread counter follows another session that is already away from the tail

Reproduced in both builds. Current showed “8 new” on History B; baseline showed
“9 new” after the equivalent stream. The precise count depends on delta batching.

Minimal reproduction:

1. Scroll B away from the bottom, then switch to A.
2. Start a streamed response in A, scroll away, and let more deltas arrive.
3. Switch back to B while B retains its earlier reading position.
4. B incorrectly displays A's unread-update counter.

Evidence: `baseline-unread-leak.png` plus the current stream observations. The shared
reducer's `RestoreSessionView` only clears `unread_stream_updates` when the restored
session follows the tail. This code was not changed by the framework upgrade. The
session text and scroll anchor remain correct; the counter is misleading. Fixing its
ownership is separate from this dependency acceptance, and must include a reducer
regression test and the applicable architecture/model review.

### Long-code rendering has existing chunk/overflow limitations

The 40-line wide-code fixture shows breaks inside some logical source lines at chunk
boundaries in both builds (for example around lines 24 and 36). Horizontal wheel
movement affects the hit chunk rather than the entire original code block. The native
run demonstrated a visible offset but did not establish that every far-right `END`
marker is reachable. This is not signed off as perfect long-code UX.

Evidence: `baseline-history-end.png` and
`current-history-horizontal-{before,after,end}.png`. The shared code-range rendering
path predates this upgrade. Keep this limitation visible rather than interpreting the
single-block headless horizontal-scroll test as coverage of all virtualized chunks.

## Automated checks repeated on the current source

- `cargo test --workspace --locked`: **412 passed, 5 ignored, 0 failed**.
  Ignored cases include the deliberately failing future TextView acceptance probe;
  TextView is not used for Chat/Trajectory Markdown.
- `cargo test --locked -p ratex-font-loader -p ratex-unicode-font`: **9 passed**.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: passed.
- `cargo fmt --all --check`: passed.
- `just tla-check` and `just tla-self-test`: passed.
- `cargo build --workspace --release --locked`: passed; native smoke checks above.
- `git diff --check`: passed.

Cargo debug/test runs used `CARGO_PROFILE_DEV_DEBUG=0`,
`CARGO_PROFILE_TEST_DEBUG=0`, and `CARGO_INCREMENTAL=0`. These control build artifacts;
they do not disable test assertions. No dependency lockfile or application behavior was
modified for the validation.
