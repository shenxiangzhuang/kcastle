# K in Castle

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
cargo run -p kcastle-desktop
```

The desktop app treats folders as projects. Each project keeps an isolated session history under
`~/.kcastle/projects`, while removing a project from the sidebar never deletes its folder or saved
history. Sessions can be created, reopened, renamed, and deleted from the sidebar. Provider
credentials stay in environment variables; desktop preferences such as reasoning effort are saved
in `~/.kcastle/settings.json`. The desktop shell includes workspace-grouped session browsing,
session search and ordering, a searchable trajectory view, expandable tool output, message copy
actions, and native session-log export.

## Get started

Set one provider key and start K:

```bash
export OPENAI_API_KEY=...
# or: export DEEPSEEK_API_KEY=...

kcastle
```

## Develop

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
cargo build --workspace --release --locked
```

## License

[Apache-2.0](LICENSE)

## Acknowledgements

Inspired by [pi](https://github.com/badlogic/pi-mono).
