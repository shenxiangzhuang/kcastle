"""Example 02 — Multi-tool Agent (Level 2: Agent)

A comprehensive example showing both non-streaming and streaming usage
with multiple tools and multi-turn conversations to solve complex problems.

Demonstrates:
  1. Non-streaming: complete() — ask, get answer, ask follow-up.
  2. Streaming:     run()      — observe tool calls and text in real time.
  3. Interactive:   steer() / abort() — redirect or cancel a running agent.

Run:
    export MINIMAX_API_KEY=...
    uv run python examples/basics/02_multi_tool_agent.py              # all demos
    uv run python examples/basics/02_multi_tool_agent.py non_stream   # demo 1
    uv run python examples/basics/02_multi_tool_agent.py stream       # demo 2
    uv run python examples/basics/02_multi_tool_agent.py interactive  # demo 3
"""

import asyncio
import os
import sys
from datetime import UTC, datetime

from kai import AnthropicMessages, Message, Tool, ToolResult
from kai.types.stream import TextDeltaEvent
from pydantic import BaseModel, Field

from kagent import (
    Agent,
    AgentAbort,
    AgentEnd,
    AgentError,
    AgentEvent,
    StreamChunk,
    ToolExecEnd,
    ToolExecStart,
    TurnEnd,
)


def make_provider() -> AnthropicMessages:
    # return AnthropicMessages(model="claude-sonnet-4-20250514")
    # return OpenAICompletions(model="gpt-4o")
    return AnthropicMessages(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/anthropic",
    )
    # return AnthropicMessages(
    #     model="MiniMax-M2.5",
    #     api_key=os.environ.get("MINIMAX_API_KEY"),
    #     base_url="https://api.minimaxi.com/anthropic",
    # )


# ---------------------------------------------------------------------------
# Tools — a small toolkit for the agent
# ---------------------------------------------------------------------------


class Calculator(Tool):
    name: str = "calculator"
    description: str = "Evaluate an arithmetic expression. Returns the numeric result."

    class Params(BaseModel):
        expression: str = Field(description="e.g. '(3 + 5) * 2'")

    async def execute(self, params: "Calculator.Params") -> ToolResult:
        try:
            value = eval(params.expression, {"__builtins__": {}})  # noqa: S307
            return ToolResult(output=str(value))
        except Exception as e:
            return ToolResult.error(str(e))


class GetWeather(Tool):
    name: str = "get_weather"
    description: str = "Get the current weather for a city."

    class Params(BaseModel):
        city: str = Field(description="City name")

    async def execute(self, params: "GetWeather.Params") -> ToolResult:
        # Stub — replace with a real API.
        data = {"Tokyo": "Sunny 22°C", "Paris": "Cloudy 15°C", "NYC": "Rainy 10°C"}
        weather = data.get(params.city, f"Clear 20°C in {params.city}")
        return ToolResult(output=weather)


class GetTime(Tool):
    name: str = "get_time"
    description: str = "Return the current UTC time."

    class Params(BaseModel):
        pass

    async def execute(self, params: "GetTime.Params") -> ToolResult:
        now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        return ToolResult(output=now)


TOOLS: list[Tool] = [Calculator(), GetWeather(), GetTime()]


# ---------------------------------------------------------------------------
# Demo 1 — Non-streaming: complete() + multi-turn follow-up
# ---------------------------------------------------------------------------


async def demo_non_streaming() -> None:
    print("=" * 60)
    print("Demo 1: Non-streaming (complete) + multi-turn")
    print("=" * 60)

    agent = Agent(
        llm=make_provider(),
        system="You are a helpful assistant. Use tools when needed. Be concise.",
        tools=TOOLS,
    )

    # Turn 1 — the agent will call get_weather + calculator tools internally.
    q1 = "What's the weather in Tokyo and Paris? Also compute (123 * 456) + (789 / 3)."
    print(f">>> {q1}\n")
    reply = await agent.complete(q1)
    print(f"[turn 1] {reply.extract_text()}")

    # Turn 2 — follow-up uses the same conversation history.
    q2 = (
        "Now tell me the current time, and convert that computation "
        "result from turn 1 to hexadecimal."
    )
    print(f"\n>>> {q2}\n")
    reply = await agent.complete(q2)
    print(f"[turn 2] {reply.extract_text()}")

    print(f"\n  ({len(agent.state.messages)} messages in history)")
    print()


# ---------------------------------------------------------------------------
# Demo 2 — Streaming: run() with full event observation
# ---------------------------------------------------------------------------


async def demo_streaming() -> None:
    print("=" * 60)
    print("Demo 2: Streaming (run) with event observation")
    print("=" * 60)

    agent = Agent(
        llm=make_provider(),
        system="You are a helpful assistant. Use tools when needed. Be concise.",
        tools=TOOLS,
    )

    print(">>> What time is it? What's 2^10 + 3^7? What's the weather in NYC?\n")

    async for event in agent.run("What time is it? What's 2^10 + 3^7? What's the weather in NYC?"):
        match event:
            case StreamChunk(event=e) if isinstance(e, TextDeltaEvent):
                print(e.delta, end="", flush=True)

            case ToolExecStart(tool_name=name, arguments=args):
                print(f"\n  [tool] {name}({args})")

            case ToolExecEnd(tool_name=name, result=result, is_error=err):
                status = "ERR" if err else "ok"
                print(f"  [{status}]  {name} → {result.output}")

            case TurnEnd(message=msg) if not msg.tool_calls:
                print(f"\n\n[final] {msg.extract_text()}")

            case AgentError(error=err):
                print(f"\n[error] {err}")
            case _:
                pass

    print()


# ---------------------------------------------------------------------------
# Demo 3 — Interactive: steer() and abort()
# ---------------------------------------------------------------------------


async def demo_interactive() -> None:
    print("=" * 60)
    print("Demo 3: Interactive — steer() and abort()")
    print("=" * 60)

    # --- steer ---
    print("\n-- steer() --")
    agent = Agent(
        llm=make_provider(),
        system="You are a helpful assistant. Use tools when needed.",
        tools=TOOLS,
    )

    agent.steer(Message(role="user", content="Actually, answer in French."))
    reply = await agent.complete("What's 2 + 2?")
    print(f"[steered] {reply.extract_text()}")

    # --- abort ---
    print("\n-- abort() --")
    agent = Agent(
        llm=make_provider(),
        system="You are a helpful assistant.",
    )

    async def cancel_after(delay: float) -> None:
        await asyncio.sleep(delay)
        agent.abort()

    asyncio.create_task(cancel_after(3))

    events: list[AgentEvent] = []
    async for event in agent.run("Write a very long essay about the history of computing."):
        events.append(event)
        if isinstance(event, AgentAbort):
            print(f"[aborted] {len(events)} events collected")
        elif isinstance(event, AgentEnd):
            print(f"[ended] {len(event.messages)} messages in history")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "non_stream"):
        await demo_non_streaming()
    if mode in ("all", "stream"):
        await demo_streaming()
    if mode in ("all", "interactive"):
        await demo_interactive()


if __name__ == "__main__":
    asyncio.run(main())
