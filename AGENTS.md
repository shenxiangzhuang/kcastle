# K Agent Harness

## Commands

- `cargo fmt --all`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --locked`
- `cargo build --workspace --release --locked`
- `cargo test -p kcastle-agent`
- `cargo check -p kcastle`

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

K is a minimal Rust workspace with two packages:

| Package | Crate | Responsibility |
| --- | --- | --- |
| `kcastle-agent` | `kcastle_agent` | Agent, state, Session, compaction, Env, and shell tool |
| `kcastle` | binary | Ratatui rendering, input, approvals, and dependency composition |

The package dependency direction is `kcastle -> kcastle-agent`. The agent package must never
import terminal infrastructure. It uses `async-openai` directly instead of maintaining a provider
abstraction.

## Core semantics

- An idle `Agent` owns its append-only `Session` and `State`.
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
- Tokio, async-openai Responses API, Serde, Ratatui, Crossterm.
- `cargo fmt`; Clippy with warnings denied; built-in Rust test harness.
- For bug fixes, reproduce the failure before implementing the fix.
- Tests protect non-trivial behavior and trust boundaries; avoid tests for trivial configuration.
- Update user-facing documentation whenever core usage changes.
- Prefer concrete structs and enums over one-implementation traits or speculative extension points.
- Conventional Commits: `<type>(<scope>): <subject>`.
