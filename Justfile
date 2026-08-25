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

# Build and package the release desktop binary as a signed macOS app.
macos-app:
    scripts/package-macos-app release

# Build and package the debug desktop binary as a signed macOS app.
macos-app-debug:
    scripts/package-macos-app debug

# Package and launch a fresh release app instance.
macos-run: macos-app
    open -n '{{app_bundle}}'

# Package and launch a fresh debug app instance.
macos-run-debug: macos-app-debug
    open -n '{{app_bundle}}'
