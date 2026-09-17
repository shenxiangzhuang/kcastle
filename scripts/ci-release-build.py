#!/usr/bin/env python3
"""Select release builds; run --self-test to check the PR policy."""

import os
from pathlib import PurePosixPath
import subprocess
import sys


def required(event, branch, paths):
    if event != "pull_request" or branch.startswith("release/"):
        return True
    return any(
        PurePosixPath(path).name
        in {
            "Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain", "rust-toolchain.toml",
            "justfile", "Justfile",
        }
        or path.startswith((".cargo/", ".github/", "scripts/", "vendor/"))
        for path in paths
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        assert required("push", "", [])
        assert required("pull_request", "release/0.2.0", [])
        assert not required("pull_request", "feature/ui", [])
        assert not required("pull_request", "feature/ui", ["crates/desktop/src/main.rs", "README.md"])
        for path in (
            "Cargo.toml", "Cargo.lock", "crates/agent/Cargo.toml", "fuzz/Cargo.lock",
            "crates/desktop/build.rs", "rust-toolchain", "rust-toolchain.toml", "justfile", "Justfile",
            ".cargo/config.toml", ".github/workflows/ci.yml", "scripts/package-macos-app",
            "vendor/ratex-font-loader/src/lib.rs",
        ):
            assert required("pull_request", "feature/ui", ["README.md", path]), path
        print("Release-build policy checks passed.")
    else:
        event = os.environ["GITHUB_EVENT_NAME"]
        branch = os.environ.get("GITHUB_HEAD_REF", "")
        paths = []
        if event == "pull_request" and not branch.startswith("release/"):
            # Compare the whole PR, including deleted paths and both sides of renames.
            paths = subprocess.check_output([
                "git", "diff", "--name-only", "--no-renames", "-z",
                f"{os.environ['BASE_SHA']}...{os.environ['HEAD_SHA']}", "--",
            ]).decode("utf-8", errors="surrogateescape").split("\0")
        print(f"required={str(required(event, branch, paths)).lower()}")
