# Kcastle

[![Crates.io](https://img.shields.io/crates/v/kcastle-agent.svg)](https://crates.io/crates/kcastle-agent)

A native agent harness with a GPUI desktop application.

## Install

Download the DMG (macOS), Setup EXE (Windows), or AppImage/DEB (Linux) from
[GitHub Releases](https://github.com/shenxiangzhuang/kcastle/releases).

## Run

Launch the downloaded desktop app, then configure OpenAI or DeepSeek in **Settings → Models**.
From a source checkout:

```bash
just macos-run                         # macOS app bundle
cargo run -p kcastle-desktop --release # Linux or Windows desktop app
```

Desktop details live in [crates/desktop/README.md](crates/desktop/README.md).
Current prerelease desktop installers are unsigned and may trigger operating-system warnings.

## Develop

Source builds require Rust 1.97 or newer.

```bash
just qa
```

Project architecture and development workflows live in [docs/README.md](docs/README.md).

## License

[Apache-2.0](LICENSE)

## Acknowledgements

Inspired by [pi](https://github.com/badlogic/pi-mono) and
[DeepSeek Harness](https://github.com/deepseek-ai/DeepSeek-Harness).
