# kcastle desktop

`kcastle-desktop` is the native GPUI interface for `kcastle-agent`. It provides project-scoped
session history, streaming chat, approvals, trajectory inspection, responsive layout, and native
macOS window behavior.

## Run on macOS

Set at least one provider key, then launch the bundled application:

```sh
export OPENAI_API_KEY=...
# or: export DEEPSEEK_API_KEY=...

just macos-run
```

Use `just macos-run-debug` for a debug build. These recipes create and sign
`target/kcastle.app` before launching it. A bare `cargo run -p kcastle-desktop` starts an
unbundled executable and does not reproduce all AppKit behavior, especially fullscreen titlebar
reveal.

Desktop preferences are stored in `~/.kcastle/settings.json`. The Models tab follows the DSH
provider-card flow: it shows credential status, expands one provider editor at a time, and supports
the OpenAI and DeepSeek providers with multiple editable models per provider. API keys saved there are never rendered
back into the form, and the settings file is written with user-only permissions on Unix. Projects
and their isolated JSONL session histories are stored below `~/.kcastle/projects`.

In the composer, press `Enter` to send and `Shift+Enter` to insert a newline.

## Architecture

The desktop crate keeps state transitions, calculations, and platform effects separate:

```text
Input / AgentEvent / Measurement
              -> Action
              -> reduce(AppState, Action)
              -> AppState + Effect[]
              -> GPUI effect runner and views
```

Key constraints:

- `domain`, `layout`, and `application` are pure layers and do not depend on GPUI.
- Views send actions instead of mutating domain state directly.
- Session, run, and layout operations use typed identities so stale async results are ignored.
- An idle `Agent` is owned by `DesktopApp`; an active run owns it until `finish` returns it.
- Streaming deltas are coalesced at the display-frame boundary, while structural events flush
  immediately.
- Chat position is restored with semantic message anchors rather than raw pixel snapshots.
- Session metadata and modification times are cached outside the render path and refreshed after
  project or session mutations.
- GPUI lifecycle calls stay in `platform/gpui`; AppKit titlebar integration stays in
  `platform/native_titlebar.rs`.

`architecture_tests.rs` enforces the pure-layer, draw-phase, and render-time filesystem boundaries.

## Verification

Run the complete local gate from the workspace root:

```sh
just qa
```

For native UI changes, also exercise the signed release bundle manually:

- Collapse and expand the sidebar and confirm the titlebar controls do not move vertically.
- Enter fullscreen, move the pointer to the top edge, and confirm all traffic lights reappear;
  exit fullscreen and confirm their windowed position is restored.
- Stream a response, scroll away from the tail, and confirm new output does not seize the scroll
  position; then use Back to bottom.
- Create, open, rename, and delete sessions across projects, including actions attempted while a
  session operation is pending.
- Resize through compact, regular, split, and overlay layouts and confirm the composer never
  covers chat or trajectory content.
