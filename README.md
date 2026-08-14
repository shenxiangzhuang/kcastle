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

Tools are denied in non-interactive mode unless explicitly enabled:

```bash
kcastle --prompt "Inspect the repository" --allow-tools
```

K writes append-only native JSONL sessions under `~/.kcastle/sessions`. The TUI creates a session
only when the first message is submitted, so opening and exiting K does not leave an empty session.
Press `/` in an empty composer to open the searchable command dashboard:

- `/session` — manage, switch, and delete saved sessions
- `/model` — switch between configured backends
- `/compact [focus]` — summarize older context
- `/permissions` — toggle tool approval prompts
- `/tool` — browse all tool calls newest first
- `/queue <message>` — run after the active task settles
- `/help` — show command help
- `/exit` — exit

Submitting ordinary text during a model or tool run steers the next model turn; input is rejected
while manual compaction is running. Use modified `Enter` for a newline, and `Up`/`Down` at the
composer boundary to recall submitted input. Press `Escape` to abort the active operation and
`Ctrl-C` to exit. Completed history uses the terminal's native scrollback, selection, and copying.
Streaming assistant output advances that native scrollback while the TUI retains only a short live
tail above the composer.
Tool status lines appear inline where calls occur, with one colored icon per call and no tool output
added to terminal history. Consecutive calls share a row until assistant text creates a new row
below that text. `/tool` lists all calls newest first with start time and duration; use
`Tab`/`Shift-Tab` to move and `Enter` to open details. Tool details support `Up`/`Down`, `j`/`k`,
`PageUp`, `PageDown`, `Home`, and `End`; approval details support the page and boundary keys.
Displayed timestamps use the system time zone and include its UTC offset.
The allow-all permission mode is confirmed explicitly and stored per session.

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
