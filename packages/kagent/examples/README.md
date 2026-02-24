# kagent examples

Five self-contained scripts that cover the full three-level API.
Read them in order — each builds on the concepts of the previous one.

## Prerequisites

```bash
export DEEPSEEK_API_KEY="sk-..."
cd <repo-root>
uv sync
```

## Examples

| File | Level | What it shows |
|------|-------|---------------|
| [`01_quickstart.py`](01_quickstart.py) | 2 — `Agent` | Simplest possible agent: one question, one answer |
| [`02_tools.py`](02_tools.py) | 2 — `Agent` | Define typed tools, observe the full event stream |
| [`03_multi_turn.py`](03_multi_turn.py) | 2 — `Agent` | Persistent history across turns; `steer()` and `abort()` |
| [`04_agent_loop.py`](04_agent_loop.py) | 1 — `agent_loop` | Custom `build_context`, `should_continue`, `on_tool_result` |
| [`05_agent_step.py`](05_agent_step.py) | 0 — `agent_step` | Single-step primitive; you own the loop and state |

## Running

```bash
uv run python packages/kagent/examples/01_quickstart.py
uv run python packages/kagent/examples/02_tools.py
# … and so on
```

## The three levels at a glance

```
Level 2 — Agent          stateful SDK, interactive control (steer / abort / follow_up)
  └─ Level 1 — agent_loop    multi-turn loop, plain-function callbacks
       └─ Level 0 — agent_step    one LLM call + tool execution, you manage state
```

Pick the lowest level that gives you the control you need.
Most applications start at **Level 2** and drop down only when required.

## Tool definition pattern

```python
from pydantic import BaseModel, Field
from kai import Tool, ToolResult

class MyTool(Tool):
    name: str = "my_tool"
    description: str = "What this tool does."

    class Params(BaseModel):
        query: str = Field(description="The input query")

    async def execute(self, params: "MyTool.Params") -> ToolResult:
        return ToolResult(output=f"result for {params.query}")
```

JSON Schema is auto-generated from `Params` — no manual `parameters` dict needed.
