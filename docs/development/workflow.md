# Development workflow

Repository manifests, source boundaries, and CI workflows are the source of truth.

## Workspace gate

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
cargo build --workspace --release --locked
```

## Focused checks

```sh
cargo test -p kcastle-agent
cargo test -p kcastle-desktop
cargo check -p kcastle-desktop
```

Use Rust edition 2024 on the stable toolchain. Deny Clippy warnings and use Rust's built-in test
harness.

Storage changes require transaction and fault tests. Session semantics require replay-prefix and
property tests. Desktop timing and trajectory changes require DSH golden fixtures. Native UI
changes also require manual validation in the packaged application.

Update user-facing documentation when core usage changes. Use Conventional Commits:
`<type>(<scope>): <subject>`.
