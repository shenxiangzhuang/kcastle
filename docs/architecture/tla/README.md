# TLA+ model checking

Executable models check selected architecture contracts alongside implementation tests.
Each model documents its bounds, assumptions, properties, and implementation mapping;
a passing model check does not prove that the Rust implementation refines the model.

| Model | Architecture contract | Scope |
| --- | --- | --- |
| [Chat presentation](chat-presentation/README.md) | [Desktop](../desktop.md#chat-viewport) | Viewport demand, freshness, cancellation, and bounded workers |
| [Inline HTML](html-preview/README.md) | [Desktop](../desktop.md#inline-html-previews) | Multiple live documents, visibility, retention and stale callback rejection |
| [HTML scrolling](html-scroll/README.md) | [Desktop](../desktop.md#inline-html-previews) | Exclusive wheel ownership, short content, boundary handoff and eventual delivery |
| [Session tools](session-tools/README.md) | [Session](../session.md) | Authorization, dispatch intent, commit receipts, ordered attachment, and cancellation/crash recovery |

The [desktop update migration](../desktop.md#desktop-updates) changes release-feed routing,
not the session or Chat state machines modeled here. Retention of the Universal bridge feed,
package availability, and Velopack app replacement are outside these models; release/updater
checks and the documented native migration procedure cover those boundaries.

## Layout

```text
tla/
  README.md
  check
  .gitignore
  session-tools/
    README.md
    SessionTools.tla
    SessionTools.cfg
    self-test
  .local/
    tools/
    session-tools/
```

Model sources, configuration, validation scripts, and generated artifacts stay in
this directory. Each independent model has its own subdirectory. Tools are cached
once; logs, traces, state files, and generated configurations are separated by model
and invocation under the ignored `.local/` directory.

## Run

Assume Java 11+ is already available in the environment (CI uses Java 21).
Bash, `curl`, `shasum`, and `grep` are also required; `just` is optional.
Run these commands from the repository root:

```sh
# All models:
just tla-check
just tla-self-test

# One model:
just tla-check session-tools
just tla-self-test session-tools

# Equivalent commands without just:
docs/architecture/tla/check
docs/architecture/tla/check self-test
docs/architecture/tla/check check session-tools
docs/architecture/tla/check self-test session-tools
```

The runner downloads the pinned [TLA+ tools 1.7.4 release](https://github.com/tlaplus/tlaplus/releases/tag/v1.7.4)
on first use and verifies its SHA-256 on every run. Each invocation prints its output
directory. TLC uses one worker, fixed seed/fingerprint settings, and a 2 GiB heap.
No binaries or generated traces are committed.

CI runs both commands for all models in the `tla` job. Check results belong to the
corresponding commit's CI run; this index does not maintain a separate verification
status. These checks remain separate from `just qa` so ordinary Rust development
does not require Java.

## Add or evolve a model

- Create a subdirectory named for the protocol or responsibility being checked.
  Use lowercase letters, digits, and hyphens; `all` is reserved for the runner.
- Keep one default `.cfg` with a matching root `.tla` module. Additional `.tla`
  modules can live alongside it; the runner copies them into the run directory.
- Add a `self-test` Bash script for model-specific sensitivity or reachability checks.
  The runner sources it in the isolated run directory with `run_tlc CONFIG_NAME`
  available. Generated configurations and traces stay in that directory; unexpected
  outcomes must return a nonzero exit code.
- Document scope, assumptions, properties, and code mapping in the model's `README.md`.
  Link the architecture document and model description in both directions, and add
  the model to this index. New model directories are discovered automatically.

Keep architecture contracts and the corresponding models consistent in the same
change. The concise maintenance rule lives in [AGENTS.md](../../../AGENTS.md).
