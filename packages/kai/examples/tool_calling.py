"""Tool calling examples — complete and streaming with tools.

Tools are subclasses of ``Tool`` with an ``execute()`` method.
The same Tool object serves as both the LLM schema and the executor.

Run:
    export DEEPSEEK_API_KEY=sk-...   # default
    # export OPENAI_API_KEY=sk-...   # if using OpenAI
    uv run python examples/tool_calling.py           # both examples
    uv run python examples/tool_calling.py complete   # complete only
    uv run python examples/tool_calling.py stream     # stream only
"""

import asyncio
import json
import os
import sys
from typing import Any

from kai import (
    Context,
    DoneEvent,
    Message,
    OpenAICompletions,
    StartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    Tool,
    ToolCallEndEvent,
    ToolCallStartEvent,
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
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    }

    async def execute(self, *, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        city = arguments["city"]
        return ToolResult(output=f"The weather in {city} is 22°C and sunny.")


class Calculate(Tool):
    """Evaluate a math expression."""

    name: str = "calculate"
    description: str = "Evaluate a math expression."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression"},
        },
        "required": ["expression"],
    }

    async def execute(self, *, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        result = str(eval(arguments["expression"]))  # noqa: S307
        return ToolResult(output=result)


TOOLS: list[Tool] = [GetWeather(), Calculate()]


def find_tool(name: str) -> Tool | None:
    """Find a tool by name."""
    for tool in TOOLS:
        if tool.name == name:
            return tool
    return None


async def execute_tool_call(name: str, arguments: str) -> str:
    """Parse arguments and execute a tool."""
    tool = find_tool(name)
    if tool is None:
        return f"Unknown tool: {name}"
    args = json.loads(arguments)
    result = await tool.execute(call_id="", arguments=args)
    return result.output


def make_provider() -> OpenAICompletions:
    # return OpenAICompletions(model="gpt-4o")
    return OpenAICompletions(
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
    async for event in await stream(provider, Context(messages=messages, tools=TOOLS)):
        match event:
            case StartEvent():
                pass
            case TextStartEvent():
                pass
            case TextDeltaEvent(delta=text):
                print(text, end="", flush=True)
            case TextEndEvent():
                print()
            case ToolCallStartEvent(name=name):
                print(f"  [Tool call: {name}]")
            case ToolCallEndEvent(tool_call=tc):
                result = await execute_tool_call(tc.name, tc.arguments)
                print(f"  [Result:    {result}]")
                tool_results.append(Message.tool_result(tc.id, result))
            case DoneEvent(message=assistant_msg):
                # Append the full assistant message (with tool_calls), then results
                if tool_results:
                    # Messages with role 'tool' must follow an assistant message with 'tool_calls'
                    messages.append(assistant_msg)
                    messages.extend(tool_results)
            case _:
                pass

    # If tool calls were made, send results back
    if tool_results:
        async for event in await stream(provider, Context(messages=messages, tools=TOOLS)):
            match event:
                case StartEvent():
                    pass
                case TextStartEvent():
                    pass
                case TextDeltaEvent(delta=text):
                    print(text, end="", flush=True)
                case TextEndEvent():
                    print()
                case DoneEvent():
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
