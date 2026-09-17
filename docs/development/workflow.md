# Development workflow

Repository manifests, source boundaries, and CI workflows are the source of truth.

## Local checks

Before pushing, run formatting checks, Clippy, workspace and vendored font tests, and whitespace
checks using the existing local build cache:

```sh
just pre-push
```

This is an explicit command, not an installed Git hook. Run `just qa` for the same checks plus an
optimized release build. Neither command includes fuzz compilation or TLA+ checks; when changing
fuzz targets or their dependencies, also run `cargo check --manifest-path fuzz/Cargo.toml --bins --locked`.

## PR and main-branch CI

Every PR retains formatting, Clippy, fuzz compilation, three-platform tests, and TLA+ checks.
Release builds run independently of those checks on every `master` push and every `release/*`
PR. Other PRs run release builds only when they change Cargo manifests or lockfiles, `build.rs`,
Rust toolchain files, `Justfile`/`justfile`, or files under `.cargo/`, `.github/`, `scripts/`, or `vendor/`.
The selection covers the whole PR, including deletions and renames; its runnable policy check is
`python3 scripts/ci-release-build.py --self-test`.

Ordinary source changes receive their release-build validation after merging into `master`;
optimization- or release-linking-only failures may therefore first appear on the main branch.
Release PRs must still pass all checks before merging, as described in [Release workflow](release.md).

## Focused checks

```sh
cargo test -p kcastle-agent
cargo test -p kcastle-desktop
cargo check -p kcastle-desktop
```

Run TLA+ model checks with `just tla-check` and `just tla-self-test`.
See [TLA+ model checking](../architecture/tla/README.md) for prerequisites and model scope.

Use Rust edition 2024 on the stable toolchain. Deny Clippy warnings and use Rust's built-in test
harness.

Storage changes require transaction and fault tests. Session semantics require replay-prefix and
property tests. Desktop timing and trajectory changes require DSH golden fixtures. Native UI
changes also require manual validation in the packaged application.

Update user-facing documentation when core usage changes. Use Conventional Commits:
`<type>(<scope>): <subject>`.
