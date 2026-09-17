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
    cargo test --locked -p ratex-font-loader -p ratex-unicode-font

# Run the focused agent test suite.
test-agent:
    cargo test -p kcastle-agent

# Measure Chat source publication/first draw and progressive preparation separately.
bench-chat:
    cargo test -p kcastle-desktop --release --locked conversation::performance::chat_presentation_benchmark -- --ignored --exact --nocapture --test-threads=1

# Compare a shared cache budget with the same three-session code browsing workload.
bench-chat-cache mib="8":
    KCASTLE_CHAT_CACHE_MIB={{quote(mib)}} cargo test -p kcastle-desktop --release --locked conversation::performance::chat_cache_benchmark -- --ignored --exact --nocapture --test-threads=1

# Check all TLA+ models or one named model (requires Java 11+).
tla-check model="all":
    docs/architecture/tla/check check {{quote(model)}}

# Verify model sensitivity and reachability with expected counterexamples.
tla-self-test model="all":
    docs/architecture/tla/check self-test {{quote(model)}}

# Build optimized workspace binaries.
build:
    cargo build --workspace --release --locked

# Run fast local checks before pushing.
pre-push: fmt-check clippy test
    git diff --check

# Run the complete local release gate.
qa: pre-push build

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
