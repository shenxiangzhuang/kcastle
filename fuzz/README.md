# Fuzz tests

```sh
cargo install cargo-fuzz --locked
rustup toolchain install nightly --profile minimal
cargo +nightly fuzz run --release streaming_markdown
cargo +nightly fuzz run --release session_events
```
