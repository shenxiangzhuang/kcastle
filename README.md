# K in Castle

A minimal agent harness.

## Packages

| Package | PyPI | Description |
| --- | --- | --- |
| **[`kcastle-agent`](packages/agent)** | [![PyPI](https://img.shields.io/pypi/v/kcastle-agent?color=%2334D058)](https://pypi.org/project/kcastle-agent/) | Agent core and harness infrastructure |
| **[`kcastle`](packages/tui)** | [![PyPI](https://img.shields.io/pypi/v/kcastle?color=%2334D058)](https://pypi.org/project/kcastle/) | Textual interface and CLI |

## Install

K requires Python 3.12 or later. Run it directly with uv:

```bash
uvx kcastle
```

Or install a persistent command:

```bash
uv tool install kcastle
kcastle
```

Upgrade a persistent installation, including pre-releases:

```bash
kcastle self update
```

## Get started

Configure one provider:

```bash
# DeepSeek (deepseek-v4-flash)
export DEEPSEEK_API_KEY=...

# Or OpenAI (gpt-5.5)
export OPENAI_API_KEY=...

uvx kcastle
```

K automatically selects the configured provider. DeepSeek takes precedence when both keys are
present.

Each launch creates an append-only JSONL session under `~/.kcastle/sessions`; its title comes from
the first user message and its metadata records the creation time. Use `/resume` to switch sessions.

Type `/` in an empty composer to open the built-in command list:

- `/resume` — Switch to a saved session.
- `/model` — Switch between detected model backends.
- `/compact` — Compact the current context.
- `/permissions` — Toggle between approval prompts and allowing all tools.
- `/queue <message>` — Run a message after the current task settles. Available while K is running.
- `/exit` — Exit K.

During a run, submit text normally to steer the next model turn. Press `Escape` to cancel the
active operation.

From a development checkout, use `uv sync` followed by `uv run kcastle`.

## Develop

```bash
just format
just check
just test
just build
```

## License

[Apache-2.0](./LICENSE)

## Acknowledgements

Inspired by [pi](https://github.com/earendil-works/pi).
