# Kcastle desktop

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
`target/Kcastle.app` before launching it. A bare `cargo run -p kcastle-desktop` starts an
unbundled executable and does not reproduce all AppKit behavior, especially fullscreen titlebar
reveal.

Desktop preferences are stored in `~/.kcastle/settings.json`. The Models tab follows the DSH
provider-card flow: configured providers stay editable, while Add provider lets you choose OpenAI
or DeepSeek before entering its API key and editing its model catalog. The composer only lists
models from providers with configured credentials. API keys saved there are never rendered
back into the form, and the settings file is written with user-only permissions on Unix. Global
model and permission choices are defaults for new sessions. Each session persists its own runtime
configuration and append-only history below `~/.kcastle/sessions`; the built-in Default project
uses `~/.kcastle/sessions/default`.

In the composer, press `Enter` to send and `Shift+Enter` to insert a newline.
Assistant Markdown renders inline formulas delimited by `$...$` or `\(...\)` and display formulas
delimited by `$$...$$` or `\[...\]`.

DMG, AppImage, and Setup EXE builds check `updates.kcastle.mathewshen.me` hourly and download a
newer release in the background. After the complete package passes its checksum, a Restart button
appears next to Settings. Restart is blocked while any session is active; otherwise the updater
waits for the old process to exit, installs the downloaded package, and launches the new version.
DEB, source, and unbundled development builds do not auto-update.

## Architecture

The desktop crate keeps view transitions, calculations, and session runtimes separate:

```text
App -> ProjectStore -> Project -> SessionRuntime -> Agent
                              \-> SessionRuntime -> Agent
```

Key constraints:

- `domain`, `layout`, and `application` are pure layers and do not depend on GPUI.
- Views send actions instead of mutating domain state directly.
- `SessionId` is the stable identity and each `SessionRuntime` owns exactly one Agent, control
  channel, event stream, approval state, queue, configuration, and writer lease. Agent events do
  not pass through an app-global bus or the currently selected session.
- Projects are directory namespaces, not scheduling boundaries. Sessions in the same or different
  projects run concurrently and switching the selected runtime only changes the visible snapshot.
- Visible run, stream, tool-result, and queued-input transitions are appended before their UI
  events. A per-session writer lease rejects a second writer while still allowing read-only browse.
- A collapsed live Think row shows the latest non-blank reasoning line and follows its horizontal
  tail at most once every three display frames; expanding it exposes the complete reasoning, and
  settlement restores the stable first line immediately.
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
- Start at least two sessions in one project and two in different projects; switch among them while
  they stream, approve, queue, stop, and finish independently.
- Confirm a background session uses a spinning ring while running, changes to a blue unread dot
  after it finishes, and clears the dot when selected; the relative time label must remain visible.
- Open a running session read-only, verify a second writer is rejected, and confirm a damaged log
  keeps valid catalog entries visible and reports its recovery state.
- Open a row's ellipsis menu without selecting it and confirm rename/delete targets that row.
- Resize through compact, regular, split, and overlay layouts and confirm the composer never
  covers chat or trajectory content.
