"""Example 03 — Advanced: agent_loop (Level 1)

Use agent_loop() when you need more control than Agent provides
but don't want to manage the step-by-step loop yourself.

Demonstrates:
  - build_context: the single hook for controlling what reaches the LLM
    (inject dynamic system info, trim history, swap tools, etc.)
  - should_continue: custom loop termination beyond "stop when no tool_calls"
  - on_tool_result: intercept / log / modify results before they return to the LLM
  - State is mutated in-place; inspect state.messages after the run.

Run:
    export DEEPSEEK_API_KEY=sk-...
    uv run python examples/03_advanced_agent_loop.py
"""

import asyncio
import os
from datetime import UTC, datetime

from kai import Anthropic, Context, Message, Tool, ToolResult
from pydantic import BaseModel

from kagent import AgentError, AgentState, TurnEnd, agent_loop


def make_provider() -> Anthropic:
    # return Anthropic(model="claude-sonnet-4-20250514")
    # return OpenAICompletions(model="gpt-4o")
    return Anthropic(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/anthropic",
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class GetTime(Tool):
    name: str = "get_time"
    description: str = "Return the current UTC time."

    class Params(BaseModel):
        pass  # no parameters needed

    async def execute(self, params: "GetTime.Params") -> ToolResult:
        now = datetime.now(tz=UTC).strftime("%H:%M:%S UTC")
        return ToolResult(output=now)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


async def build_context(state: AgentState) -> Context:
    """Inject the current date into the system prompt and keep only recent history."""
    today = datetime.now(tz=UTC).strftime("%A, %B %d %Y")
    return Context(
        system=f"{state.system}\nToday is {today}.",
        messages=state.messages[-20:],  # sliding window — avoids unbounded growth
        tools=state.tools,
    )


async def on_tool_result(call_id: str, tool_name: str, result: ToolResult) -> ToolResult:
    """Log every tool call and its result."""
    print(f"[intercepted] {tool_name} → {result.output}")
    return result  # pass through unchanged (could modify here)


async def should_continue(state: AgentState, assistant_msg: Message) -> bool:
    """Stop the loop after the first text-only reply (no tool calls)."""
    return bool(assistant_msg.tool_calls)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    # You own the state — create it and seed the first user message.
    state = AgentState(
        system="You are a concise assistant.",
        messages=[Message(role="user", content="What time is it right now?")],
        tools=[GetTime()],
    )

    print("Running agent_loop…\n")
    async for event in agent_loop(
        provider=make_provider(),
        state=state,
        build_context=build_context,
        on_tool_result=on_tool_result,
        should_continue=should_continue,
    ):
        match event:
            case TurnEnd(message=msg) if msg.extract_text():
                print(f"[assistant] {msg.extract_text()}")

            case AgentError(error=err):
                print(f"[error] {err}")
            case _:
                pass

    # state.messages now contains the full conversation.
    print(f"\nConversation history ({len(state.messages)} messages):")
    for msg in state.messages:
        print(f"  [{msg.role}] {str(msg.content)[:80]}")


if __name__ == "__main__":
    asyncio.run(main())
