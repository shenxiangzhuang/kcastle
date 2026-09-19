# GPUI Kit 0.6.4 acceptance — 2026-09-19

## Decision

Keep the 0.6.4 dependency upgrade and framework PopupMenu/Popover/Dialog integration.
Do **not** ship the TextView migration. Restore Trajectory to the same Markdown renderer
as Chat, removing the parallel TextView/math-plugin/delimiter-adapter path introduced
in `fe96e4f`. A framework replacement is useful only when the old path can actually
be removed without losing behavior.

This is a rejection of the tested drop-in migration, not proof that integration is
impossible. Layout/selection can still be separated from Chat's bounded preparation
pipeline, but TextView needs semantic parity before that work is worth undertaking.

## Comparison and blockers

The same synthetic [Markdown fixture](../../crates/desktop/tests/fixtures/markdown-acceptance.md)
was rendered in the real macOS app at 1180 × 720. The baseline uses the shared renderer;
a temporary candidate replaced prepared Markdown bodies with `TextView::markdown` and
the pilot's inline/display RaTeX plugins. Native HTML remained on its existing path.
No model requests, real credentials or user sessions were used.

| Check | Shared renderer | TextView candidate | Decision |
| --- | --- | --- | --- |
| Ordered list starting at 98 | 98, 99, 100 | 1, 2, 3 | Blocker: displayed content changes |
| Plain-text list/task copy | Keeps numbers and checked/unchecked markers | `first\nsecond\ndone\npending\n` | Blocker: copied meaning changes |
| Code language and copy action | Present | Absent by default | Adaptable through `code_block_actions`, not an upstream blocker |
| Code geometry | Existing six-line fixture is 217px including gap/header/padding | 196px | Existing layout regression test fails; requires explicit styling/actions |
| Inline/display math, Chinese, emoji, formula table | Visible | Visible in captured wide/light screens | This alone does not establish parity |

In upstream v0.6.4, `format/markdown.rs` converts `Node::List` without retaining
`list.start`; `node.rs` renders prefixes from the item index. Its plain-text copy
walks list-item children without emitting numbers or task markers. Switching to
`SelectionFormat::Source` changes copying to Markdown source, which also changes
bold/link/code copying and does not fix the displayed list numbers.

Sources: [Markdown conversion](https://github.com/longbridge/gpui-kit/blob/v0.6.4/crates/base/src/text/format/markdown.rs),
[node rendering/copy](https://github.com/longbridge/gpui-kit/blob/v0.6.4/crates/base/src/text/node.rs),
[TextView public customization](https://github.com/longbridge/gpui-kit/blob/v0.6.4/crates/base/src/text/text_view.rs).

The candidate was a capability probe, not a complete Chat adapter: it did not consume
`code_visible()` ranges for split long-code rows. Rendering the full source in each
such row would duplicate content; reparsing sliced source would lose multiline syntax
context. Those need a separate integration design if the semantic blockers are fixed.
No performance improvement is claimed, and the existing worker/cache/selection
lifecycles remain unchanged.

## Runnable checks

The production regression test selects the rendered content using the real selection
layer and verifies list numbers and task states:

```sh
cargo test -p kcastle-desktop ordered_and_task_lists_preserve_markers_when_copied
```

The deliberately ignored **future-upgrade acceptance probe** is expected to fail on
0.6.4. It stays out of normal CI because TextView is not used for this content:

```sh
cargo test -p kcastle-desktop framework_markdown_list_copy_acceptance -- --ignored --nocapture
```

Before retrying the migration, make that probe pass and repeat the native list-number
comparison. Do not change it to assert the current loss of information.

The original math smoke test checked that formula elements existed and could be copied;
it did not compare list semantics, code actions or visual layout with the shared renderer.
That gap allowed the Trajectory pilot through prematurely. The checked-in fixture and
copy acceptance probe now make these differences repeatable.

## Initial application checks

The checks below were the evidence available when `b4435cb` was committed, not a
complete native acceptance run. The subsequent [full macOS regression report](gpui-kit-0.6.4-full-validation.md)
adds real session journals, Trajectory, long-history switching, live streaming,
cancellation, restart, and an executable pre-upgrade comparison.

- Workspace tests: 412 passed, 5 ignored (four existing benchmarks/tests plus the
  explicitly failing upstream acceptance probe); zero unexpected failures.
- Vendored font tests: 9 passed.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`.
- `cargo fmt --all --check`, `git diff --check`.
- `just tla-check` and `just tla-self-test`: passed. No modeled lifecycle changed;
  architecture/model scope notes were updated to reflect the withdrawn pilot.
- Native shared-renderer checks: light/dark, 1180px and minimum 720px window widths,
  Chinese/English and styled text, inline/display/emoji math, tables, numbered/task
  lists, code highlighting, bottom sentinel and vertical wheel scrolling.
  A headless horizontal-wheel test verifies long code moves; the native automation
  attempt did not show an observable horizontal offset, so native horizontal-wheel
  behavior is not signed off by these screenshots.
- Native interaction checks: text selection and actual paste into the composer preserve
  formulas; the code-copy button copies the entire source; keyboard navigation enters
  the permission submenu, confirming the existing option closes it and restores composer
  focus; Settings has the Dialog accessibility role, accepts Tab and closes with Escape.
- Native HTML overlay check: changed a slider from 30 to 52, opened the command menu,
  then opened Settings from it. Both overlays hid the native previews; closing Settings
  restored the previews with the slider still at 52.

The narrow-window check found an 800px Settings panel overflowing the 720px window and
hiding Done. The controls test now resizes while the dialog is open and asserts its
bounds stay inside the viewport. It failed with `800px × 570px` before adding
`max_w_full()` to the Settings content itself. Constraining only the outer popup did
not constrain this fixed-width child.

Local screenshots, the discarded prototype patch and failure/success logs are saved
under `target/gpui-kit-0.6.4-validation/` (ignored artifacts). Key comparisons:
`baseline-top.png` / `candidate-top.png` show the numbering difference;
`baseline-bottom.png` / `candidate-bottom.png` show code actions and layout;
`final-light-top.png`, `final-light-bottom.png`, `final-dark-narrow.png`,
`final-selection.png`, `final-copy.png`, `final-menu.png`, and dialog captures record
the retained implementation. The fixture can be reopened in a debug build with:

```sh
KCASTLE_DATA_DIR=/tmp/kcastle-markdown-acceptance \
KCASTLE_PREVIEW_MARKDOWN="$PWD/crates/desktop/tests/fixtures/markdown-acceptance.md" \
  cargo run -p kcastle-desktop --locked
```

These checks found no remaining regression in the exercised final paths. They do not
establish pixel equivalence on every OS or rule out all transient frames. In particular,
the candidate failed acceptance; it must not be described as a completed Markdown migration.
