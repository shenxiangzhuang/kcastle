# kai Examples

Runnable examples demonstrating kai's features. Each file is self-contained — just set
`DEEPSEEK_API_KEY` and run:

```bash
export DEEPSEEK_API_KEY=sk-...
uv run python examples/basic.py
```

## Examples

| Example | Description |
|---------|-------------|
| [basic.py](basic.py) | Simplest usage — `complete()` and `stream()` in one file |
| [tool_calling.py](tool_calling.py) | Tool definitions with manual execute-and-respond loop (`complete` and `stream`) |
| [agent_loop.py](agent_loop.py) | Multi-round agent loop with tool calling and rich terminal output |
| [error_handling.py](error_handling.py) | Graceful error handling for both `complete()` and `stream()` |
| [multimodal.py](multimodal.py) | Image + text input (vision) |
