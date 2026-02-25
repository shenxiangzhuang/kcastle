# kagent — Architecture

Detailed architecture documentation for kagent. For quick start and overview, see the [README](../README.md).

## Context Flow

```
Mutable state ──► Immutable snapshot ──► LLM call
(AgentState)       (kai.Context)          (kai.stream)
     │                  ▲
     │  ContextBuilder  │
     └──────────────────┘
          .build()
```

Context is built by a **`ContextBuilder`** — a protocol with a single method:
`async def build(self, state: AgentState) -> Context`. This is the single point
where all context customization happens — compaction, pruning, system prompt
injection, tool filtering.

### Built-in Builders

| Builder | Description |
|---------|-------------|
| `DefaultBuilder` | Pass-through — sends all messages (implicit default) |
| `SlidingWindowBuilder` | Keeps first message + last N messages |
| `CompactingBuilder` | Summarizes older messages via LLM call |
| `AdaptiveBuilder` | Delegates to a named builder; switchable at runtime |

`ContextSwitchTool` (created via `AdaptiveBuilder.create_tool()`) lets the
agent itself choose its context strategy at runtime.

## Module Layout

```
kagent/
├── __init__.py          # Public API re-exports
├── py.typed             # PEP 561 typed marker
├── context.py           # ContextBuilder protocol + built-in implementations
├── event.py             # AgentEvent — lifecycle event discriminated union
├── state.py             # AgentState dataclass
├── step.py              # agent_step() — single-turn primitive
├── loop.py              # agent_loop() — multi-turn loop
└── agent.py             # Agent — stateful SDK with steering & follow-up
```

Tool types (`Tool`, `ToolResult`) live in `kai.tool` — kagent re-exports them for convenience.

---

## Event Types

Lifecycle events emitted as `AgentEvent` discriminated union:

| Event | Emitted by | Description |
|-------|-----------|-------------|
| `AgentStart` | `agent_loop` | Loop begins |
| `AgentEnd` | `agent_loop` | Loop ends (with final messages) |
| `TurnStart` | `agent_step` | Single LLM turn begins |
| `TurnEnd` | `agent_step` | Turn ends (with assistant message + tool results) |
| `StreamChunk` | `agent_step` | Wraps a `kai.StreamEvent` |
| `ToolExecStart` | `agent_step` | Tool execution begins (call_id, name, args) |
| `ToolExecEnd` | `agent_step` | Tool execution ends (call_id, name, result) |
| `AgentError` | both | Wraps an exception |

```python
type AgentEvent = (
    AgentStart | AgentEnd
    | TurnStart | TurnEnd
    | StreamChunk
    | ToolExecStart | ToolExecEnd
    | AgentError
)
```

## AgentState

The mutable conversation state. All stateful data lives here.

```python
@dataclass
class AgentState:
    system: str | None = None
    messages: list[Message] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
```

- **Mutable by design** — the loop appends messages in-place.
- **`state` is public** on `Agent` — users can read `agent.state.messages`, modify `agent.state.system`, swap `agent.state.tools` between runs.

## Level 0: agent_step() Algorithm

The lowest-level API. Accepts a `kai.Context` directly — the caller has full control.

```
yield TurnStart

# Stream LLM response
async for stream_event in kai.stream(provider, context):
    yield StreamChunk(event=stream_event)
    if DoneEvent: assistant_msg = stream_event.message
    if ErrorEvent: yield AgentError → return

# Execute tool calls (if any)
tool_map = {tool.name: tool for tool in tools}
tool_results = []
for tool_call in assistant_msg.tool_calls:
    tool = tool_map.get(tool_call.name)
    args = json.loads(tool_call.arguments)
    yield ToolExecStart(call_id=..., tool_name=..., arguments=args)

    try:
        result = await _execute_tool(tool, args)  # Pydantic validation + execute(params)
    except Exception as e:
        result = ToolResult.error(str(e))

    if on_tool_result:
        result = await on_tool_result(call_id, tool_name, result)

    yield ToolExecEnd(call_id=..., tool_name=..., result=result, is_error=result.is_error)
    tool_results.append(Message.tool_result(tool_call.id, result.output, is_error=result.is_error))

yield TurnEnd(message=assistant_msg, tool_results=tool_results)
```

## Level 1: agent_loop() Algorithm

Wraps `agent_step()` with automatic message management and loop control.

```
yield AgentStart

for turn_count in range(max_turns):
    # === Single context construction point ===
    context = await context_builder.build(state)  # ContextBuilder protocol

    # Delegate to agent_step
    async for event in agent_step(provider=provider, context=context, tools=state.tools):
        yield event

        if isinstance(event, TurnEnd):
            state.messages.append(event.message)
            for tool_msg in event.tool_results:
                state.messages.append(tool_msg)

        if isinstance(event, AgentError):
            yield AgentEnd(messages=state.messages)
            return

    # Decide whether to continue
    if should_continue:
        if not await should_continue(state, assistant_msg):
            break
    elif not assistant_msg.tool_calls:
        break

yield AgentEnd(messages=state.messages)
```

## Callback / Extension Points

```python
class ContextBuilder(Protocol):
    async def build(self, state: AgentState) -> Context: ...
"""Build a kai.Context from the current agent state.
The SINGLE POINT where all context customization happens."""

type OnToolResultFn = Callable[[str, str, ToolResult], Awaitable[ToolResult]]
"""Called after tool execution. Receives (call_id, tool_name, result).
Return (possibly modified) result."""

type ShouldContinueFn = Callable[[AgentState, Message], Awaitable[bool]]
"""After each turn, decide whether to continue looping.
Receives (state, assistant_message). Return False to stop."""
```

## ContextBuilder Examples

Built-in builders cover common patterns:

```python
from kagent import (
    DefaultBuilder,
    SlidingWindowBuilder,
    CompactingBuilder,
    AdaptiveBuilder,
)

# Pass-through (default when no builder specified)
builder = DefaultBuilder()

# Keep first message + last 20
builder = SlidingWindowBuilder(window_size=20)

# Summarize older messages via LLM when conversation exceeds 30 messages
builder = CompactingBuilder(provider, threshold=30, max_preserved=6)

# Agent-controlled switching between strategies
adaptive = AdaptiveBuilder(
    builders={
        "full": DefaultBuilder(),
        "window": SlidingWindowBuilder(window_size=20),
        "compact": CompactingBuilder(provider, threshold=30),
    },
    default="full",
)
agent = Agent(provider=p, context_builder=adaptive, tools=[adaptive.create_tool()])
```

Custom builders implement the protocol (structural typing — no inheritance required):

```python
# Dynamic system prompt
class DynamicSystemBuilder:
    async def build(self, state: AgentState) -> Context:
        system = f"{state.system}\nTime: {datetime.now()}"
        return Context(system=system, messages=state.messages, tools=state.tools)

# Tool filtering per turn
class ToolFilterBuilder:
    async def build(self, state: AgentState) -> Context:
        active_tools = select_tools(state)
        return Context(system=state.system, messages=state.messages, tools=active_tools)
```

## Level 2: Agent Design Notes

- **Same callbacks as `agent_loop()`** — the Agent constructor takes the same kwargs. No wrapping.
- **`run()` + `complete()`** — mirrors `kai.stream()` + `kai.complete()`. `run()` for streaming, `complete()` for scripting.
- **Steering and follow-up** are Agent-level features (not loop-level). They require statefulness (queues, current-run awareness).

## Design Comparison

| Aspect | pi-agent-core | kimi-cli (soul) | **kagent** |
|--------|--------------|-----------------|-----------|
| Language | TypeScript | Python | Python |
| Levels | 2 (loop + Agent) | 1 (KimiSoul) | 3 (step + loop + Agent) |
| Core loop | `agentLoop()` async generator | `_agent_loop()` method | `agent_step()` + `agent_loop()` |
| Tool type | `AgentTool` interface | `CallableTool` class | `kai.Tool` subclass |
| Events | `AgentEvent` union | `WireMessage` via Wire | `AgentEvent` union |
| State | `AgentState` interface | `Context` + `Runtime` | `AgentState` dataclass |
| Extensibility | Config callbacks (kwargs) | Wire + Approval + ContextVar | Callbacks (kwargs) |
| Streaming | `EventStream` push-based | Wire SPMC broadcast | `AsyncIterator` pull-based |
| Context build | `transformContext` + `convertToLlm` | Compaction in loop | `ContextBuilder` protocol |
| Message types | `Message` + `CustomAgentMessages` | `Message` | `kai.Message` directly |
| SDK API | `prompt()` + subscribe | N/A | `run()` + `complete()` |

## What's NOT in kagent (by design)

These belong in application layers (`kcode`, etc.):

- **Persistence** — Session saving, JSONL, checkpoints.
- **Approval UI** — via `on_tool_result` callback.
- **Specific tools** — File, shell, web tools.
- **UI/rendering** — Terminal UI, wire protocol.
- **Agent specs** — YAML/config-driven agent definitions.
- **Retry logic** — In provider wrapper or custom `ContextBuilder`.
- **Sub-agents** — Implemented as a tool.
