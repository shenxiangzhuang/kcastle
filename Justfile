set shell := ["zsh", "-cu"]

app_bundle := "target/kcastle.app"
app_plist := "crates/desktop/resources/Info.plist"
app_icon := "crates/desktop/assets/app-icon.svg"

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
    #!/usr/bin/env zsh
    set -euo pipefail
    app='{{app_bundle}}'
    plist='{{app_plist}}'
    icon='{{app_icon}}'
    icon_work=$(mktemp -d /tmp/kcastle-app-icon.XXXXXX)
    trap 'rm -rf "$icon_work"' EXIT
    iconset="$icon_work/AppIcon.iconset"
    mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources" "$iconset"

    qlmanage -t -s 1024 -o "$icon_work" "$icon" >/dev/null 2>&1
    rendered="$icon_work/app-icon.svg.png"
    [[ -f "$rendered" ]]
    for spec in \
        '16 icon_16x16.png' \
        '32 icon_16x16@2x.png' \
        '32 icon_32x32.png' \
        '64 icon_32x32@2x.png' \
        '128 icon_128x128.png' \
        '256 icon_128x128@2x.png' \
        '256 icon_256x256.png' \
        '512 icon_256x256@2x.png' \
        '512 icon_512x512.png' \
        '1024 icon_512x512@2x.png'; do
        size=${spec%% *}
        name=${spec#* }
        sips -z "$size" "$size" "$rendered" --out "$iconset/$name" >/dev/null
    done
    iconutil -c icns "$iconset" -o "$app/Contents/Resources/AppIcon.icns"

    cp "$plist" "$app/Contents/Info.plist"
    package_id=$(cargo pkgid -p kcastle-desktop)
    version=${package_id##*@}
    /usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $version" "$app/Contents/Info.plist"
    install -m 755 '{{binary}}' "$app/Contents/MacOS/kcastle"
    codesign --force --deep --sign - "$app"
    codesign --verify --deep --strict "$app"
    echo "Packaged $app ($version)"
