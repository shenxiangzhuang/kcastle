# Kcastle

[![Crates.io](https://img.shields.io/crates/v/kcastle.svg)](https://crates.io/crates/kcastle)

A native agent harness with desktop and terminal clients.

## Install

- Desktop: download the DMG (macOS), EXE (Windows), or DEB (Debian/Ubuntu) from [GitHub Releases](https://github.com/shenxiangzhuang/kcastle/releases).
- Terminal: `cargo install kcastle --locked`.

## Run

Set `OPENAI_API_KEY` or `DEEPSEEK_API_KEY`, then launch the downloaded desktop app or run:

```bash
kcastle
```

From a source checkout:

```bash
just macos-run                         # macOS app bundle
cargo run -p kcastle-desktop --release # Linux or Windows desktop app
```

Desktop details live in [crates/desktop/README.md](crates/desktop/README.md).

## Develop

```bash
just qa
```

## License

[Apache-2.0](LICENSE)

## Acknowledgements

Inspired by [pi](https://github.com/badlogic/pi-mono) and
[DeepSeek Harness](https://github.com/deepseek-ai/DeepSeek-Harness).
