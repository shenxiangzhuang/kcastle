# K Agent Harness

## Commands

- `just sync`
- `just format`
- `just check`
- `just test`
- `just build`
- `just hooks`
- `just format-pkg agent` / `just check-pkg agent` / `just test-pkg agent`
- `just format-pkg tui` / `just check-pkg tui` / `just test-pkg tui`

## Architecture

K is a minimal Python agent harness with two packages:

| Package | Import | Responsibility |
| --- | --- | --- |
| `kcastle-agent` | `kagent` | Agent core plus Session, Env, and ToolRuntime adapters |
| `kcastle` | `ktui` | Textual rendering, input, approvals, and dependency composition |

Within `kagent`, the dependency direction is `harness -> Agent core`; the core must never import
harness infrastructure. At package level it is `ktui -> kagent`. The project intentionally uses
the OpenAI SDK directly instead of maintaining a provider abstraction.

## Core semantics

- `Agent` owns its append-only `State`.
- `Agent` depends only on state-commit and tool-execution ports for external effects.
- `kagent.harness.Session` persists State; `Env` and `ToolRuntime` execute capabilities.
- `steer()` injects input after the current assistant response and its tool calls.
- `queue()` runs input after the agent would otherwise settle.
- `abort()` cancels the active run.
- Compaction appends a summary marker and projects a retained suffix; it never deletes history.
- User interfaces consume `AgentEvent` values and must not persist State based on UI events.

## Conventions

- Python 3.12+ and modern syntax.
- `asyncio`, OpenAI Responses API, Pydantic, Textual.
- Ruff line length 100; ty; pytest + pytest-asyncio; prek.
- For bug fixes, first add and run a focused regression test that reproduces the failure; then
  implement the fix and rerun that test plus the relevant package checks and tests.
- Prefer a small concrete API over speculative extension points.
- Conventional Commits: `<type>(<scope>): <subject>`.
- Versions use minor bumps only; patch is always `0`.
