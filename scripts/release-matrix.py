#!/usr/bin/env python3
"""Release targets, including the one-time Universal migration; --self-test checks retirement."""

import json
import sys

# Keep this tag fixed after publication; old clients must always reach this bridge.
MACOS_BRIDGE_TAG = "v0.2.0-alpha.28"

TARGETS = [
    {
        "os": "ubuntu-latest",
        "target": "x86_64-unknown-linux-gnu",
        "targets": "x86_64-unknown-linux-gnu",
        "velopack_runtime": "linux-x64",
        "update_target": "linux-x64",
        "pack_dir": "target/velopack-input",
        "main_exe": "kcastle-desktop",
        "primary_asset": "kcastle-desktop-linux-amd64.AppImage",
        "secondary_asset": "kcastle-desktop-linux-amd64.deb"
    },
    {
        "os": "ubuntu-24.04-arm",
        "target": "aarch64-unknown-linux-gnu",
        "targets": "aarch64-unknown-linux-gnu",
        "velopack_runtime": "linux-arm64",
        "update_target": "linux-arm64",
        "pack_dir": "target/velopack-input",
        "main_exe": "kcastle-desktop",
        "primary_asset": "kcastle-desktop-linux-arm64.AppImage",
        "secondary_asset": "kcastle-desktop-linux-arm64.deb"
    },
    {
        "os": "macos-14",
        "target": "aarch64-apple-darwin",
        "targets": "aarch64-apple-darwin",
        "update_target": "osx-arm64",
        "pack_dir": "target/Kcastle.app",
        "main_exe": "kcastle",
        "primary_asset": "kcastle-desktop-macos-arm64.dmg",
        "secondary_asset": "",
        "velopack_runtime": "osx-arm64",
        "macos_arches": "arm64"
    },
    {
        "os": "macos-14",
        "target": "x86_64-apple-darwin",
        "targets": "x86_64-apple-darwin",
        "update_target": "osx-x64",
        "pack_dir": "target/Kcastle.app",
        "main_exe": "kcastle",
        "primary_asset": "kcastle-desktop-macos-x64.dmg",
        "secondary_asset": "",
        "velopack_runtime": "osx-x64",
        "macos_arches": "x86_64"
    },
    {
        "os": "macos-14",
        "target": "universal-apple-darwin",
        "targets": "aarch64-apple-darwin,x86_64-apple-darwin",
        "update_target": "osx-universal",
        "pack_dir": "target/Kcastle.app",
        "main_exe": "kcastle",
        "primary_asset": "kcastle-desktop-macos-universal.dmg",
        "secondary_asset": "",
        # Velopack's default supports Universal; an explicit "osx" RID is rejected.
        "velopack_runtime": "",
        "macos_arches": "x86_64 arm64"
    },
    {
        "os": "windows-latest",
        "target": "x86_64-pc-windows-msvc",
        "targets": "x86_64-pc-windows-msvc",
        "velopack_runtime": "win-x64",
        "update_target": "win-x64",
        "pack_dir": "target/velopack-input",
        "main_exe": "kcastle-desktop.exe",
        "primary_asset": "kcastle-desktop-windows-x86_64-setup.exe",
        "secondary_asset": ""
    }
]


def release_matrix(tag):
    return {"include": [
        target for target in TARGETS
        if target["update_target"] != "osx-universal" or tag == MACOS_BRIDGE_TAG
    ]}


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        for tag in ("v0.2.0-alpha.27", MACOS_BRIDGE_TAG, "v0.2.0-alpha.29", "v0.2.0-beta.1", "v0.2.0"):
            targets = release_matrix(tag)["include"]
            feeds = {target["update_target"] for target in targets}
            assert {"linux-x64", "linux-arm64", "win-x64", "osx-arm64", "osx-x64"} <= feeds
            assert ("osx-universal" in feeds) == (tag == MACOS_BRIDGE_TAG)
            assert len(feeds) == len(targets)
            assert len({target["primary_asset"] for target in targets}) == len(targets)
            for target in targets:
                expected_runtime = "" if target["update_target"] == "osx-universal" else target["update_target"]
                assert target["velopack_runtime"] == expected_runtime
        print("Release matrix checks passed.")
    else:
        print(json.dumps(release_matrix(sys.argv[1])))
