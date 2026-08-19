# kcastle agent harness

## Commands

- `cargo fmt --all`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --locked`
- `cargo build --workspace --release --locked`
- `cargo test -p kcastle-agent`
- `cargo test -p kcastle`
- `cargo test -p kcastle-desktop`
- `cargo check -p kcastle`
- `cargo check -p kcastle-desktop`

The workspace-wide commands cover the agent core and both user interfaces. Package-specific
commands are focused checks: `kcastle` is the terminal UI, while `kcastle-desktop` is the desktop
UI.

## Release workflow

1. Create `release/<version>` from the default branch and update the workspace version and the
   exact internal dependency version together. Stable versions use minor bumps with patch `0`;
   prereleases use Cargo semver such as `0.2.0-alpha.1`.
2. Open a pull request and wait for all CI checks to pass. The pull request author must merge it;
   agents and automation must not merge release pull requests.
3. Only after confirming the pull request was merged, publish a GitHub Release from the merged
   commit using tag `v<version>`. Mark alpha and beta versions as pre-releases. The release workflow
   publishes `kcastle-agent`, then `kcastle`, and uploads native binaries.

## Architecture

kcastle is a Rust workspace with an agent core and two user interfaces:

| Package | Crate | Responsibility |
| --- | --- | --- |
| `kcastle-agent` | `kcastle_agent` library | UI-independent agent runtime, state, commit port, `Session`, compaction, `Env`, and tools |
| `kcastle` | terminal binary | Ratatui/Crossterm rendering, terminal input, approvals, and TUI dependency composition |
| `kcastle-desktop` | desktop binary | GPUI rendering, desktop input, approvals, session orchestration, and desktop dependency composition |

The package dependency directions are `kcastle -> kcastle-agent` and
`kcastle-desktop -> kcastle-agent`. The two UI packages do not depend on each other. The agent core
must never import terminal or desktop infrastructure. It uses `async-openai` directly instead of
maintaining a provider abstraction.

## Keeping this file current

- Treat repository manifests, source boundaries, and CI/release workflows as the source of truth
  when they conflict with this file.
- When an architecture or workflow change makes `AGENTS.md` stale, adapt by proposing an update to
  `AGENTS.md`; do not silently preserve known-outdated guidance.
- Before editing `AGENTS.md` for that reason, tell the user why the update is needed and summarize
  the exact sections or rules that would change. Wait for explicit user approval before applying
  the edit.
- After approval, update the stale guidance in the same task and verify it against the relevant
  manifests, source boundaries, and automation files.

## Core semantics

- An idle `Agent` owns append-only `State` plus a `StateCommit` persistence port.
- `Session` is the default JSONL adapter; `AgentTool` values provide executable capabilities.
- `Agent::start(self, input)` transfers the agent to one background task; `ActiveAgent::finish()`
  returns ownership after the operation settles.
- `RunControl` sends steering, follow-up, approval, and cancellation signals without shared mutable
  agent state.
- Steering runs after the current response and its tools; queueing runs after the agent would
  otherwise settle.
- Cancellation records unresolved tool calls as having unknown side effects.
- Compaction appends a summary boundary and retains recent input batches; it does not mutate prior
  records.
- User interfaces consume `AgentEvent` values and never persist state from UI events.

## Conventions

- Rust edition 2024; stable toolchain.
- Tokio, async-openai Responses API, Serde, Ratatui/Crossterm for the TUI, and GPUI for the desktop
  UI.
- `cargo fmt`; Clippy with warnings denied; built-in Rust test harness.
- For bug fixes, reproduce the failure before implementing the fix.
- Tests protect non-trivial behavior and trust boundaries; avoid tests for trivial configuration.
- Update user-facing documentation whenever core usage changes.
- Prefer concrete structs and enums over one-implementation traits or speculative extension points.
- Conventional Commits: `<type>(<scope>): <subject>`.
