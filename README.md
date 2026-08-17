# kcastle

[![Crates.io](https://img.shields.io/crates/v/kcastle.svg)](https://crates.io/crates/kcastle)

A minimal native agent harness.

## Install

Install the published binary with Cargo:

```bash
cargo install kcastle --locked
```

From a source checkout:

```bash
cargo install --path crates/tui --locked
```

Run the GPUI desktop app from a source checkout:

```bash
just macos-run
```

This recipe builds the release binary, places it in `target/kcastle.app`, applies the tracked
`Info.plist`, signs the bundle, and launches a fresh instance. Use `just macos-run-debug` for a
debug build. A direct `cargo run -p kcastle-desktop` launches an unbundled executable and does not
reproduce macOS AppKit behavior such as fullscreen titlebar reveal.

The desktop app uses a strict App → Projects → Sessions hierarchy. Every project is bound to one
folder and owns an isolated session namespace below `~/.kcastle/sessions`; when no folder is
selected, the built-in Default project uses `~/.kcastle/sessions/default` as both its project and
session directory. Any number of sessions may run concurrently within one project or across
projects, and switching or creating sessions never stops background work. Removing a project from
the sidebar never deletes its folder or saved history. Sessions can be created, reopened, renamed,
and deleted from the sidebar. Provider credentials can come from environment variables or the
desktop Models settings page. Global settings are new-session defaults; each session persists its
own model, reasoning effort, and tool permission in its JSONL log. The desktop shell includes workspace-grouped session browsing,
session search and ordering, a searchable trajectory view, expandable tool output, message copy
actions, and native session-log export. The desktop Models settings page supports the OpenAI and
DeepSeek providers. Each provider owns an editable model catalog shared with the TUI defaults;
saved credentials, endpoint overrides, and model metadata are written to the user-only
`~/.kcastle/settings.json` file.

Desktop architecture, native launch behavior, and manual verification are documented in
[the desktop README](crates/desktop/README.md).

## Get started

Set one provider key and start kcastle:

```bash
export OPENAI_API_KEY=...
# or: export DEEPSEEK_API_KEY=...

kcastle
```

## Develop

```bash
just --list
just qa
```

The `Justfile` also exposes focused recipes including `fmt`, `check`, `clippy`, `test`,
`test-agent`, `build`, `tui`, `macos-app`, and their debug/run variants.

The pure Desktop core also has scheduled mutation coverage. Run a focused local pass with
`cargo mutants --package kcastle-desktop --re '<pattern>'` after installing `cargo-mutants`.

## License

[Apache-2.0](LICENSE)

## Acknowledgements

Inspired by [pi](https://github.com/badlogic/pi-mono) and
[DeepSeek Harness](https://github.com/deepseek-ai/DeepSeek-Harness).
