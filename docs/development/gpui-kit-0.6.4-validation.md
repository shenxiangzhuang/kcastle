# GPUI Kit 0.6.4 acceptance — 2026-09-19

Keep the 0.6.4 upgrade and framework menus/dialogs; retain shared Chat/Trajectory Markdown because TextView changes ordered-list numbering and drops list/task markers when copying.

Comparing `b4435cb` with pre-upgrade `3083dbd` on macOS, native visual checks covered Chat/Trajectory, long-history switching, streaming/cancellation, copying, menus/dialogs, themes, narrow windows, HTML previews and restart, with no new regression observed.

All 421 tests passed (5 additional tests intentionally ignored), along with Clippy, formatting, TLA+ checks and the release build.

Two existing issues remain: unread counters can carry across sessions, and long-code chunks have wrapping/horizontal-scroll limitations; Windows/Linux were not validated.

Local screenshots and logs remain in `target/gpui-kit-0.6.4-validation/` and `target/gpui-kit-full-validation/`.
