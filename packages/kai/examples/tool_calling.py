"""Tool calling examples — complete and streaming with tools.

kai provides declarative tool definitions only; execution logic belongs
in the application layer (or in kagent for automated loops).

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
    complete,
    stream,
)

# ---------------------------------------------------------------------------
# Tools & execution
# ---------------------------------------------------------------------------

TOOLS = [
    Tool(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    ),
    Tool(
        name="calculate",
        description="Evaluate a math expression.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"},
            },
            "required": ["expression"],
        },
    ),
]


def execute_tool(name: str, arguments: str) -> str:
    """Simulate tool execution (replace with real logic)."""
    args = json.loads(arguments)
    if name == "get_weather":
        return f"The weather in {args['city']} is 22°C and sunny."
    if name == "calculate":
        return str(eval(args["expression"]))  # noqa: S307
    return f"Unknown tool: {name}"


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
            result = execute_tool(tc.name, tc.arguments)
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
                result = execute_tool(tc.name, tc.arguments)
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
