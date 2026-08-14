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
