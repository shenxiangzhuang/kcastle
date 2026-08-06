# K in Castle

K is a small, stateful agent harness with a Textual interface.

The project deliberately has two boundaries:

- [`agent`](packages/agent): the Agent core plus surrounding harness infrastructure. The core
  owns cognition; `kagent.harness` provides Session, Env, and executable tools.
- [`tui`](packages/tui): a thin Textual adapter. It selects a session path and environment, then
  owns only terminal rendering, input, and approval prompts.

## Run

```bash
# DeepSeek (deepseek-v4-flash)
export DEEPSEEK_API_KEY=...

# Or OpenAI (gpt-5.5)
export OPENAI_API_KEY=...

uvx kcastle

# From a development checkout
uv sync
uv run kcastle
```

K automatically selects the configured provider. DeepSeek takes precedence when both keys are
present. Use `--model` or `--context-window` to override the provider defaults.

Each launch creates an append-only JSONL session under `~/.kcastle/sessions`; its title comes from
the first user message and its metadata records the creation time. Use `--session PATH` to resume a
specific session directly.

Type `/` in an empty composer to open the command list. Use `/resume` to switch sessions,
`/model` to switch detected backends, `/compact` to compact context, `/permissions` to switch
between approval prompts and allowing all tools, or `/exit` to leave K. During a run, submit text
normally to steer the next model turn or use `/queue message` to run it after the current task
settles. Press `Escape` to cancel the active operation.

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

Inspired by [pi-mono](https://github.com/badlogic/pi-mono).
