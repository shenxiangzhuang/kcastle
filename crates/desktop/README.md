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
model and permission choices are defaults for new sessions. Each project persists session metadata
and append-only transactions in a SQLite WAL database in its configured sessions directory; the
built-in Default project uses `~/.kcastle/sessions/default/sessions.sqlite3`. JSONL is export-only.

Development and isolated acceptance runs can set `KCASTLE_DATA_DIR` to put settings, the project
registry, and every project session store under a separate data root. An absolute path is
recommended; a relative path is resolved from the process working directory. The variable must not
be empty. Normal launches leave it unset and continue to use `~/.kcastle`.

In the composer, press `Enter` to send and `Shift+Enter` to insert a newline.
Assistant Markdown renders inline formulas delimited by `$...$` or `\(...\)` and display formulas
delimited by `$$...$$` or `\[...\]`.

The desktop trajectory surface is projected directly from session events rather than chat rows. It
has fixed Input, Model, and Tools swimlanes. Duration switches equal-width operations to recorded
durations with idle time compressed, matching the DSH desktop surface; complete wall-clock timing
remains an internal projection. Dragging across the overview focuses the interval, dims operations
outside it, and scrolls the ledger to the focused records. Scroll to zoom around the pointer,
right-drag to pan, double-click or press Escape to clear the focus. Tool
bars distinguish the full call lifecycle from nested execution time; assistant bars distinguish
TTFT from generation. Hovering a bar restores its full color, outlines it, and reveals the DSH
timing tooltip after a short delay; hovering empty lane space shows a vertical time cursor. The
Timing tab uses the same recorded timestamps and can toggle Started
between local and Unix time. The composer footer summarizes whole-session turn/step counts, LLM and
tool wall time, average TTFT, decode throughput, cache hit rate, and token usage. Session catalog
entries that fail validation are ignored by the desktop UI. Request boundary markers open an
independent Request inspector whose Options, Usage, and Timing tabs use only canonical recorded
data; System Prompt, Tools, and Tool Schema views share the same immutable request snapshot.
Trajectory search, folds, selection, details history, focus, viewport, tail-follow state, and scroll
position are retained independently for each session; Duration mode is a persisted desktop
preference shared by sessions. Timeline statistics update
incrementally, while geometry is cached by session, event revision, and mode and then cheaply
reprojected for the current viewport; Duration coordinates use a merged busy-time index rather
than rescanning every interval for every item.

DMG, AppImage, and Setup EXE builds check `updates.kcastle.mathewshen.me` hourly and download a
newer release in the background. After the complete package passes its checksum, a compact update
button appears beside Settings. Restart is blocked while any session is active; otherwise the
updater waits for the old process to exit, installs the downloaded package, and launches the new
version. DEB, source, and unbundled development builds do not auto-update.

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
  channel, committed-event stream, approval state, configuration, and canonical
  `SessionDocument`. Agent events do not pass through an app-global bus or the currently selected
  session.
- Projects are directory namespaces, not scheduling boundaries. Sessions in the same or different
  projects run concurrently and switching the selected runtime only changes the visible snapshot.
- Durable chat, trajectory, timing, details, search, and composer data are selectors over the same
  committed document revision. Transient controls never create durable chat rows.
- SQLite expected-revision compare-and-swap serializes writers. Every model/tool intent commits
  before the external effect and the UI sees only the resulting commit receipt.
- A collapsed live Think row shows the latest non-blank reasoning line and follows its horizontal
  tail at most once every three display frames; expanding it exposes the complete reasoning, and
  settlement restores the stable first line immediately.
- Chat position is restored with semantic message anchors rather than raw pixel snapshots.
- Session metadata and committed update times are cached outside the render path and refreshed
  after project or session mutations; opaque session locators are never treated as data files.
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
- Confirm unsupported or damaged sessions are silently omitted while valid catalog entries remain
  visible.
- Hover a session row and confirm rename appears before archive without selecting the row; restore
  or permanently delete archived sessions from the matching project in Settings.
- Resize through compact, regular, split, and overlay layouts and confirm the composer never
  covers chat or trajectory content.
