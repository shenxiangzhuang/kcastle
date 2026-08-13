# K in Castle

A minimal native agent harness.

## Packages

| Package | Responsibility |
| --- | --- |
| [`kcastle-agent`](crates/agent) | Responses agent core, state, sessions, compaction, and tools |
| [`kcastle`](crates/tui) | Ratatui application, input, approvals, and dependency composition |

The dependency direction is strictly `kcastle -> kcastle-agent`. The agent package uses
`async-openai` directly and does not maintain a provider abstraction.

## Install

Install the published binary with Cargo:

```bash
cargo install kcastle --locked
```

From a source checkout:

```bash
cargo install --path crates/tui --locked
```

## Get started

Configure one provider:

```bash
# DeepSeek, selected first when both keys are present
export DEEPSEEK_API_KEY=...

# Or OpenAI
export OPENAI_API_KEY=...

kcastle
```

For a non-interactive run:

```bash
kcastle --prompt "Explain this repository"
```

K writes append-only native JSONL sessions under `~/.kcastle/sessions`. Type `/` commands directly
in the composer:

- `/resume` — open a saved session
- `/model` — switch between configured backends
- `/compact [focus]` — summarize older context
- `/permissions` — toggle tool approval prompts
- `/queue <message>` — run after the active task settles
- `/help` — show command help
- `/exit` — exit

Submitting ordinary text during a run steers the next model turn. Press `Escape` to abort the
active model or tool operation and `Ctrl-C` to exit.

## Develop

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
cargo build --workspace --release --locked
```

## License

[Apache-2.0](LICENSE)

## Acknowledgements

Inspired by [pi](https://github.com/badlogic/pi-mono).
