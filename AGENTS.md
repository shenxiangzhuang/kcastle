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

## Release workflow

1. Create `release/<version>` from the default branch and update every project and internal
   dependency version consistently. Versions use minor bumps only; patch is always `0`.
2. Open a pull request and wait for all CI checks to pass. The pull request author must merge it;
   agents and automation must not merge release pull requests.
3. Only after confirming the pull request was merged, publish a GitHub Release from the merged
   commit using tag `v<version>` to trigger the release workflow. Mark versions containing `a` or
   `b` (alpha or beta) as pre-releases.

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
- For bug fixes, reproduce the failure before implementing the fix.
- Tests are liabilities, not assets. Add the smallest test only when necessary to protect
  non-trivial behavior; do not test trivial configuration or implementation details.
- Update user-facing documentation whenever core usage changes.
- Prefer a small concrete API over speculative extension points.
- Conventional Commits: `<type>(<scope>): <subject>`.
