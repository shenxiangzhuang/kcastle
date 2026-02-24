"""Example 04 — Advanced: agent_step (Level 0)

agent_step() is the lowest-level primitive: one LLM call + tool execution.
You build the Context, call agent_step(), and handle state yourself.

Use this level when you need:
  - Custom looping logic (e.g. tree search, beam search, retry strategies)
  - Non-standard context construction per step
  - Direct access to every streaming event with no wrapping

This example manually drives the step loop — appending messages and repeating
until the model stops requesting tools — to show exactly what happens at each
step and why a loop is required.

Run:
    export DEEPSEEK_API_KEY=sk-...
    uv run python examples/04_advanced_agent_step.py
"""

import asyncio
import os

from kai import Anthropic, Context, Message, Tool, ToolResult
from kai.event import TextDeltaEvent
from pydantic import BaseModel, Field

from kagent import agent_step
from kagent.event import AgentError, StreamChunk, ToolExecEnd, ToolExecStart, TurnEnd


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


class Calculator(Tool):
    name: str = "calculator"
    description: str = "Evaluate a simple arithmetic expression and return the result."

    class Params(BaseModel):
        expression: str = Field(description="Arithmetic expression, e.g. '(3 + 5) * 2'")

    async def execute(self, params: "Calculator.Params") -> ToolResult:
        try:
            # eval() is fine in a sandboxed example — don't do this in production.
            value = eval(params.expression, {"__builtins__": {}})  # noqa: S307
            return ToolResult(output=str(value))
        except Exception as e:
            return ToolResult.error(str(e))


# ---------------------------------------------------------------------------
# Helper: run one step and collect results
# ---------------------------------------------------------------------------


async def run_step(
    provider: Anthropic,
    messages: list[Message],
    tools: list[Tool],
    *,
    step_label: str,
) -> tuple[Message, list[Message]]:
    """Execute one agent step. Returns (assistant_msg, tool_result_messages)."""
    context = Context(
        system="You are a precise calculator assistant. Use the calculator tool.",
        messages=messages,
        tools=tools,
    )

    print(f"\n--- {step_label} ---")

    assistant_msg: Message | None = None
    tool_results: list[Message] = []

    async for event in agent_step(provider=provider, context=context, tools=tools):
        match event:
            case StreamChunk(event=e) if isinstance(e, TextDeltaEvent):
                print(e.delta, end="", flush=True)

            case ToolExecStart(tool_name=name, arguments=args):
                print(f"\n[calling tool] {name}({args})")

            case ToolExecEnd(tool_name=name, result=result):
                print(f"[tool result]  {name} → {result.output}")

            case TurnEnd(message=msg, tool_results=tr):
                assistant_msg = msg
                tool_results = tr

            case AgentError(error=err):
                raise RuntimeError(f"Agent error: {err}") from err

    assert assistant_msg is not None
    return assistant_msg, tool_results


# ---------------------------------------------------------------------------
# Main — manually drive the step loop until the model stops calling tools
# ---------------------------------------------------------------------------


async def main() -> None:
    provider = make_provider()
    tools: list[Tool] = [Calculator()]

    # Seed the conversation with the user request.
    messages: list[Message] = [
        Message(role="user", content="What is (123 * 456) + (789 / 3)?"),
    ]

    # Manually loop: run one step at a time, appending results to messages,
    # until the model returns a reply with no tool calls.
    # This is what agent_loop() does internally — here you own every iteration.
    step = 0
    while True:
        step += 1
        assistant_msg, tool_results = await run_step(
            provider, messages, tools, step_label=f"Step {step}"
        )
        messages.append(assistant_msg)
        messages.extend(tool_results)

        if not tool_results:
            # No tool calls → model is done.
            break

    print(f"\n[final answer] {assistant_msg.extract_text()}")


if __name__ == "__main__":
    asyncio.run(main())
