"""Tool calling examples — complete and streaming with tools.

Tools are subclasses of ``Tool`` with an inner ``Params(BaseModel)`` class
and an ``execute(params)`` method. JSON Schema is auto-generated from the
Pydantic model, keeping tool definitions typed and DRY.

Run:
    export DEEPSEEK_API_KEY=sk-...   # default
    # export OPENAI_API_KEY=sk-...   # if using OpenAI
    uv run python examples/tool_calling.py           # both examples
    uv run python examples/tool_calling.py complete   # complete only
    uv run python examples/tool_calling.py stream     # stream only
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from pydantic import BaseModel, Field

from kai import (
    Context,
    Done,
    Message,
    OpenAIChatCompletions,
    TextDelta,
    Tool,
    ToolCallBegin,
    ToolResult,
    complete,
    stream,
)

# ---------------------------------------------------------------------------
# Tools (subclass Tool and override execute())
# ---------------------------------------------------------------------------


class GetWeather(Tool):
    """Get the current weather for a city."""

    name: str = "get_weather"
    description: str = "Get the current weather for a city."

    class Params(BaseModel):
        city: str = Field(description="City name")

    async def execute(self, params: GetWeather.Params) -> ToolResult:
        return ToolResult(output=f"The weather in {params.city} is 22°C and sunny.")


class Calculate(Tool):
    """Evaluate a math expression."""

    name: str = "calculate"
    description: str = "Evaluate a math expression."

    class Params(BaseModel):
        expression: str = Field(description="Math expression")

    async def execute(self, params: Calculate.Params) -> ToolResult:
        result = str(eval(params.expression))  # noqa: S307
        return ToolResult(output=result)


TOOLS: list[Tool] = [GetWeather(), Calculate()]


TOOL_MAP: dict[str, Tool] = {t.name: t for t in TOOLS}


async def execute_tool_call(name: str, arguments: str) -> str:
    """Parse arguments and execute a tool."""
    tool = TOOL_MAP.get(name)
    if tool is None:
        return f"Unknown tool: {name}"
    args = json.loads(arguments)
    result = await tool.execute(args)
    return result.output


def make_provider() -> OpenAIChatCompletions:
    # return OpenAIChatCompletions(model="gpt-4o")
    return OpenAIChatCompletions(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )


# ---------------------------------------------------------------------------
# Example 1: complete + tool calling (minimal working example)
# ---------------------------------------------------------------------------


async def example_complete() -> None:
    """MWE: complete() with two tools."""
    print("=" * 60)
    print("Example 1: complete + tool calling")
    print("=" * 60)

    provider = make_provider()
    messages: list[Message] = [
        Message(role="user", content="What's the weather in Tokyo? Also compute 123 * 456 + 789."),
    ]

    # Step 1: Ask the model (it may request tool calls)
    response = await complete(provider, Context(messages=messages, tools=TOOLS))
    messages.append(response)

    # Step 2: Execute tool calls if any
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"  Tool call: {tc.name}({tc.arguments})")
            result = await execute_tool_call(tc.name, tc.arguments)
            print(f"  Result:    {result}")
            messages.append(Message.tool_result(tc.id, result))

        # Step 3: Send tool results back to the model
        final = await complete(provider, Context(messages=messages, tools=TOOLS))
        print(f"\nAssistant: {final.extract_text()}")
    else:
        print(f"Assistant: {response.extract_text()}")

    print()


# ---------------------------------------------------------------------------
# Example 2: stream + tool calling
# ---------------------------------------------------------------------------


async def example_stream() -> None:
    """MWE: stream() with two tools."""
    print("=" * 60)
    print("Example 2: stream + tool calling")
    print("=" * 60)

    provider = make_provider()
    messages: list[Message] = [
        Message(role="user", content="What's the weather in Paris? Also compute 2 ** 10 - 1."),
    ]

    # Stream with tool call events
    tool_results: list[Message] = []
    done_msg: Message | None = None
    async for event in stream(provider, Context(messages=messages, tools=TOOLS)):
        match event:
            case TextDelta(delta=text):
                print(text, end="", flush=True)
            case ToolCallBegin(name=name):
                print(f"  [Tool call: {name}]")
            case Done(message=assistant_msg):
                done_msg = assistant_msg  # noqa: F841
                if assistant_msg.tool_calls:
                    for tc in assistant_msg.tool_calls:
                        result = await execute_tool_call(tc.name, tc.arguments)
                        print(f"  [Result:    {result}]")
                        tool_results.append(Message.tool_result(tc.id, result))
                    messages.append(assistant_msg)
                    messages.extend(tool_results)
            case _:
                pass

    # If tool calls were made, send results back
    if tool_results:
        async for event in stream(provider, Context(messages=messages, tools=TOOLS)):
            match event:
                case TextDelta(delta=text):
                    print(text, end="", flush=True)
                case Done():
                    pass
                case _:
                    pass

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "complete"):
        await example_complete()
    if mode in ("all", "stream"):
        await example_stream()


if __name__ == "__main__":
    asyncio.run(main())
