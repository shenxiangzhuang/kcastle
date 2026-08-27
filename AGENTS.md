# kcastle agent harness

## Commands

- `cargo fmt --all`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --locked`
- `cargo build --workspace --release --locked`
- `cargo test -p kcastle-agent`
- `cargo test -p kcastle-desktop`
- `cargo check -p kcastle-desktop`

The workspace-wide commands cover the agent core and desktop application. Package-specific
commands are focused checks for either `kcastle-agent` or `kcastle-desktop`.

## Release workflow

1. Create `release/<version>` from the default branch and update the workspace version and the
   exact internal dependency version together. Stable versions use minor bumps with patch `0`;
   prereleases use Cargo semver such as `0.2.0-alpha.1`.
2. Open a pull request and wait for all CI checks to pass. The pull request author must merge it;
   agents and automation must not merge release pull requests.
3. Only after confirming the pull request was merged, publish a GitHub Release from the merged
   commit using tag `v<version>`. Mark alpha and beta versions as pre-releases. The release workflow
   publishes `kcastle-agent` and uploads native desktop binaries.

## Architecture

kcastle is a Rust workspace with an agent core and one desktop application:

| Package | Crate | Responsibility |
| --- | --- | --- |
| `kcastle-agent` | `kcastle_agent` library | UI-independent agent runtime, transactional `SessionMachine`, SQLite `SessionStore`, canonical session facts, compaction, `Env`, and tools |
| `kcastle-desktop` | desktop binary | GPUI rendering, desktop input, approvals, session orchestration, and desktop dependency composition |

The package dependency direction is `kcastle-desktop -> kcastle-agent`. The agent core must never
import desktop infrastructure. It owns only the harness runtime and persistence semantics; product
configuration, provider catalogs, session presentation, search formatting, and title policy belong
to the desktop crate. It uses `async-openai` directly instead of maintaining a provider abstraction.

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

- An idle `Agent` owns one replayable `SessionMachine` plus a concrete transactional
  `SessionStore`; there is no second validator or mutable persistence mirror.
- A project-local SQLite WAL database is the only runtime source of truth. JSONL is an explicit
  export format, and pre-v2 session files are intentionally ignored rather than migrated or
  dual-written.
- One logical domain transition is committed as one idempotent transaction with an expected
  session revision. The machine evolves and committed events are published only after the store
  confirms the commit; persistence failures never trigger in-memory rollback.
- Model and tool effects start only after their durable intent is committed. Effect results return
  as correlated commands to the single session owner; unresolved non-idempotent tool attempts are
  recorded as having unknown side effects and are never retried automatically.
- Inputs are submitted and later attached atomically to one run/turn/step. Lifecycle completion,
  cancellation, failure, and crash recovery each use a complete terminal transaction.
- `Agent::start(self, input)` transfers the agent to one background task; `ActiveAgent::finish()`
  returns ownership after the operation settles.
- `RunControl` sends steering, follow-up, approval, and cancellation signals without shared mutable
  agent state.
- Steering runs after the current response and its tools; queueing runs after the agent would
  otherwise settle.
- Compaction commits a summary boundary and retains recent input batches in the canonical replayed
  surface; it does not rewrite prior journal transactions.
- The desktop interface derives durable content only from committed session transactions. Transient
  control signals may drive approvals, connection status, and notices but never conversation,
  trajectory, timing, or session statistics.
- Conversation, trajectory, details, timing, search, and composer statistics are selectors over one
  canonical session document at one revision. Incremental application must equal full replay for
  every committed prefix.

## Conventions

- Rust edition 2024; stable toolchain.
- Tokio, async-openai Responses API, Serde, rusqlite/SQLite WAL for session transactions, and GPUI
  for the desktop UI.
- `cargo fmt`; Clippy with warnings denied; built-in Rust test harness.
- Desktop launches for development, manual UI testing, and acceptance checks must set
  `KCASTLE_DATA_DIR` to a non-empty absolute path dedicated to that task. Never point development
  or test builds at the installed app's live data directory (normally `~/.kcastle`). Use separate
  data roots for builds with incompatible storage schemas; reproduce user-data issues only on
  disposable copies, leaving the original data untouched.
- Ensure `KCASTLE_DATA_DIR` reaches the desktop process, not just its launcher. On macOS, package
  the app first, then launch its executable directly, for example
  `KCASTLE_DATA_DIR="$(mktemp -d /tmp/kcastle-dev.XXXXXX)" target/Kcastle.app/Contents/MacOS/kcastle`.
  Do not assume setting the variable before `just macos-run`, `just macos-run-debug`, or `open`
  passes it to the app launched through Launch Services.
- For bug fixes, reproduce the failure before implementing the fix.
- Tests protect non-trivial behavior and trust boundaries; avoid tests for trivial configuration.
- Session storage changes require transaction/fault tests; session semantics require replay-prefix
  and property tests; desktop timing and trajectory changes require DSH golden fixtures.
- Update user-facing documentation whenever core usage changes.
- Prefer concrete structs and enums over one-implementation traits or speculative extension points.
- Conventional Commits: `<type>(<scope>): <subject>`.
