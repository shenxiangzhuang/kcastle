# kcastle

The official Textual interface for `kcastle-agent`.

```bash
export DEEPSEEK_API_KEY=...  # deepseek-v4-flash
# or: export OPENAI_API_KEY=...  # gpt-5.5
uvx kcastle
```

The configured provider is selected automatically; DeepSeek takes precedence when both keys are
present.

Each launch creates its own titled, timestamped JSONL session under `~/.kcastle/sessions`.
Use `/resume` to switch sessions.

Type `/` in an empty composer to open the built-in commands. `/resume` switches sessions,
`/model` switches detected backends, `/compact` compacts context, `/permissions` switches between
approval prompts and allowing all tools, and `/exit` exits. While the agent is running, submitted
text steers the current task; prefix a message with `/queue ` to run it after the current task
settles. Permission modes are remembered per session, and enabling allow all requires confirmation.
Press Escape to cancel the current operation.
