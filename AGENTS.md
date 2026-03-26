# K Agent Framework

## Quick commands (use just)

- `just sync` (sync deps for all workspace packages)
- `just format` / `just check`
- `just test`
- `just build`
- `just format-pkg kai` / `just check-pkg kai` / `just test-pkg kai` (per-package)
- `just clean`

Or run tools directly with `uv run`:

- `uv run ruff check` / `uv run ruff format`
- `uv run pyright`
- `uv run pytest`


## PoC stage

We are in PoC stage now, the break changes are acceptable, and we will iterate fast. The main goal is to build a clean and solid foundation for the agent framework, so we can easily build on top of it and add more features in the future. We will focus on the LLM abstraction layer(kai) and the core agent runtime(kagent) first, and then build the agent application(kcastle) on top.


## Project overview

K is a Python monorepo for building AI agents. It provides a unified LLM abstraction layer,
a core agent runtime, and an agent application with multi-endpoint support.

## Tech stack

- Python 3.12+
- Async runtime: asyncio
- Package management/build: uv + uv_build
- Tests: pytest + pytest-asyncio
- Lint/format: ruff
- Type checking: pyright (strict mode)

## Packages

| Package | Import | Description |
|---------|--------|-------------|
| **[kai](packages/kai)** | `import kai` | Unified multi-provider LLM API |
| **[kagent](packages/kagent)** | `import kagent` | Agent runtime with tool calling and state management |
| **[kcastle](packages/kcastle)** | `import kcastle` | Agent application with multi-endpoint support |

### Dependency chain

```
kcastle ──► kagent ──► kai
```

- `kai` is the foundation: LLM provider abstractions, no internal deps.
- `kagent` builds on `kai`: agent runtime, tool calling, state management.
- `kcastle` builds on `kagent`: agent application with multi-endpoint support (CLI, Telegram, Discord).

## Repo structure

```
k/
├── pyproject.toml              # workspace root
├── packages/
│   ├── kai/src/kai/            # LLM abstraction layer
│   ├── kagent/src/kagent/      # core agent framework
│   └── kcastle/src/kcastle/    # agent application
└── references/                 # local-only reference projects (gitignored)
```

## Conventions and quality

- Python >=3.12; use modern Python style (type union `X | Y`, `match/case`, `StrEnum`, `dataclass(slots=True)`, etc.).
- Line length 100.
- Ruff handles lint + format (rules: E, F, UP, B, SIM, I).
- Pyright strict mode for type checks.
- `kai` and `kagent` are typed library packages (include `py.typed`).
- Tests use pytest + pytest-asyncio; files are `tests/test_*.py`.
- Each package has its own `pyproject.toml` with local tool config.
- Workspace members reference each other via `[tool.uv.sources]` with `{ workspace = true }`.

## Git commit messages

Conventional Commits format:

```
<type>(<scope>): <subject>
```

Allowed types:
`feat`, `fix`, `test`, `refactor`, `chore`, `style`, `docs`, `perf`, `build`, `ci`, `revert`.

## Versioning

The project follows a **minor-bump-only** versioning scheme (`MAJOR.MINOR.PATCH`):

- **Patch** version is always `0`. Never bump it.
- **Minor** version is bumped for any change: new features, improvements, bug fixes, etc.
- **Major** version is only changed by explicit manual decision.

Examples: `0.1.0` → `0.2.0` → `0.3.0`; never `0.1.1`.

This rule applies to all packages in the repo.

## Release workflow

Use this release process:

1. Create a `release/<name>` branch for the release work and open a PR.
2. Merge the release PR into `master`.
3. Create the release tag from `master` after the PR is merged.
4. Use a GitHub Releases friendly tag format:
   - stable: `vMAJOR.MINOR.PATCH`
   - prerelease: `vMAJOR.MINOR.PATCH-alpha.N`, `vMAJOR.MINOR.PATCH-beta.N`, `vMAJOR.MINOR.PATCH-rc.N`

Examples:
- `v0.1.0`
- `v0.2.0-alpha.1`
- `v0.2.0-beta.1`
- `v1.0.0-rc.1`

## Release notes

Release notes should use this structure:

1. `## Highlights`
2. PR list
3. `## Notes`

Recommended format:

```md
## Highlights
- item 1
- item 2

## Included pull requests
- #123 — `feat(scope): summary`
- #124 — `chore(scope): summary`

## Notes
- note 1
- note 2
```
