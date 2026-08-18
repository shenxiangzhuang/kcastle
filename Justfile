set shell := ["zsh", "-cu"]

app_bundle := "target/Kcastle.app"

# List available recipes.
default:
    @just --list

# Format all workspace crates.
fmt:
    cargo fmt --all

# Check Rust formatting without changing files.
fmt-check:
    cargo fmt --all --check

# Type-check the entire workspace using the lockfile.
check:
    cargo check --workspace --locked

# Run Clippy with warnings denied.
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Run all workspace tests.
test:
    cargo test --workspace --locked

# Run the focused agent test suite.
test-agent:
    cargo test -p kcastle-agent

# Build optimized workspace binaries.
build:
    cargo build --workspace --release --locked

# Run the complete local release gate.
qa: fmt-check clippy test build
    git diff --check

# Run the terminal client; pass extra arguments after `--`.
tui *args:
    cargo run -p kcastle -- {{args}}

# Build and package the release desktop binary as a signed macOS app.
macos-app:
    cargo build -p kcastle-desktop --release --locked
    just _package-macos "target/release/kcastle-desktop"

# Build and package the debug desktop binary as a signed macOS app.
macos-app-debug:
    cargo build -p kcastle-desktop --locked
    just _package-macos "target/debug/kcastle-desktop"

# Package and launch a fresh release app instance.
macos-run: macos-app
    open -n '{{app_bundle}}'

# Package and launch a fresh debug app instance.
macos-run-debug: macos-app-debug
    open -n '{{app_bundle}}'

_package-macos binary:
    scripts/package-macos-app '{{binary}}'
