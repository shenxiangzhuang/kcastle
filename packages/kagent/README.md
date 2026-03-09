# kagent

[![PyPI](https://img.shields.io/pypi/v/kcastle-agent?color=%2334D058&label=pypi%20package)](https://pypi.org/project/kcastle-agent/)

Core agent runtime for the K agent framework — a context-first, three-level agent SDK built on `kai`.

## Key Principles

1. **Context-first** — Context construction is managed by ``ContextBuilder`` — a protocol with built-in implementations for common patterns. You control exactly what goes to the LLM.
2. **Three levels** — `agent_step()` → `agent_loop()` → `Agent`. Pick the level that fits.
3. **No framework concepts** — Callbacks are plain function parameters. No Hooks, no Middleware.
4. **Unified tools** — Uses `kai.Tool` directly. Define inner `Params(BaseModel)` for typed, auto-schema tools.
5. **kai-native** — Uses `kai.Message`, `kai.Context`, `kai.Provider`, `kai.StreamEvent` directly.
6. **App-owned provider assembly** — `kagent` consumes a concrete `kai.Provider`; provider factory/registry concerns stay in the application layer.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Application (kcode, custom agent)                   │
│  ┌────────────────────────────────────────────────┐  │
│  │  Level 2: Agent (stateful SDK)                 │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │  Level 1: agent_loop() (multi-turn)      │  │  │
│  │  │  ┌────────────────────────────────────┐  │  │  │
│  │  │  │  Level 0: agent_step() (one turn)  │  │  │  │
│  │  │  │  ┌──────────────────────────────┐  │  │  │  │
│  │  │  │  │  kai.stream() (LLM call)     │  │  │  │  │
│  │  │  │  └──────────────────────────────┘  │  │  │  │
│  │  │  └────────────────────────────────────┘  │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

| Level | Function | Context control | Use case |
|-------|----------|-----------------|----------|
| 0 | `agent_step()` | Caller builds `kai.Context` | Full control, custom loops |
| 1 | `agent_loop()` | `context_builder` param | Standard multi-turn agent |
| 2 | `Agent` | Auto-managed + callbacks | Stateful SDK, interactive apps |

## Quick Start

### Define a Tool

Subclass `kai.Tool`, define inner `Params(BaseModel)`, and override `execute()`:

```python
from pydantic import BaseModel, Field
from kai import Tool, ToolResult


class GetWeather(Tool):
    name: str = "get_weather"
    description: str = "Get weather for a location."

    class Params(BaseModel):
        location: str = Field(description="City name")

    async def execute(self, params: GetWeather.Params) -> ToolResult:
        return ToolResult(output=f"Sunny in {params.location}")
```

JSON Schema is auto-generated from the `Params` model — no manual `parameters` dict needed.

### Level 2: Agent (recommended)

The simplest way to use kagent — stateful, multi-turn, streaming or one-shot:

```python
from kagent import Agent

agent = Agent(
    provider=my_provider,
    system="You are a helpful assistant.",
    tools=[GetWeather()],
)

# One-shot
msg = await agent.complete("What's the weather in Tokyo?")
print(msg.extract_text())

# Streaming
async for event in agent.run("What's the weather in Paris?"):
    match event:
        case TurnEnd(message=msg):
            print(msg.extract_text())

# Multi-turn (state persists)
await agent.complete("Remember: my name is Alice.")
msg = await agent.complete("What's my name?")  # "Alice"
```

### Level 1: agent_loop()

For more control — you own the state, customize context via callbacks:

```python
from kagent import AgentState, agent_loop, TurnEnd

state = AgentState(
    system="You are helpful.",
    messages=[Message(role="user", content="What's the weather?")],
    tools=[GetWeather()],
)

async for event in agent_loop(provider=my_provider, state=state):
    match event:
        case TurnEnd(message=msg):
            print(msg.extract_text())

# state.messages is mutated in-place with the full conversation
```

**Context customization** via `context_builder`:

```python
from kagent import SlidingWindowBuilder

# Built-in: sliding window
async for event in agent_loop(
    provider=p, state=state,
    context_builder=SlidingWindowBuilder(window_size=20),
):
    ...

# Custom: implement the ContextBuilder protocol
class MyBuilder:
    async def build(self, state: AgentState) -> Context:
        return Context(
            system=f"{state.system}\nTime: {datetime.now()}",
            messages=state.messages[-20:],
            tools=state.tools,
        )

async for event in agent_loop(provider=p, state=state, context_builder=MyBuilder()):
    ...
```

### Level 0: agent_step()

Full control — you build the `kai.Context`, manage state yourself:

```python
from kagent import agent_step, TurnEnd

context = Context(
    system="You are helpful.",
    messages=[Message(role="user", content="Hello!")],
    tools=my_tools,
)
async for event in agent_step(provider=provider, context=context, tools=my_tools):
    match event:
        case TurnEnd(message=msg):
            print(msg.extract_text())
```

## Callbacks

All callbacks are plain `async` functions passed as kwargs — same at Level 1 and Level 2:

| Callback | Signature | Purpose |
|----------|-----------|---------|
| `context_builder` | `ContextBuilder` protocol | Control what goes to the LLM |
| `on_tool_result` | `(call_id, tool_name, ToolResult) -> ToolResult` | Intercept/modify tool results |
| `should_continue` | `(AgentState, Message) -> bool` | Custom loop termination |

## Interactive Control (Level 2 only)

```python
agent.steer(Message(role="user", content="Focus on X instead."))  # interrupt current run
agent.follow_up(Message(role="user", content="Now do Y."))        # queue after current run
agent.abort()                                                      # cancel current run
```

## More

See [docs/arch.md](docs/arch.md) for detailed architecture, algorithm pseudo-code, context flow, and design comparisons.
