"""Basic usage — complete() and stream() examples.

Run:
    export DEEPSEEK_API_KEY=sk-...   # default
    # export OPENAI_API_KEY=sk-...   # if using OpenAI
    uv run python examples/basic.py             # both examples
    uv run python examples/basic.py complete     # complete only
    uv run python examples/basic.py stream       # stream only
"""

import asyncio
import os
import sys

from kai import (
    Context,
    Done,
    Message,
    OpenAIChatCompletions,
    TextDelta,
    complete,
    stream,
)


def make_provider() -> OpenAIChatCompletions:
    # return OpenAIChatCompletions(model="gpt-4o")
    return OpenAIChatCompletions(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )


# ---------------------------------------------------------------------------
# Example 1: complete()
# ---------------------------------------------------------------------------


async def example_complete() -> None:
    """Get a complete response in one call."""
    print("=" * 60)
    print("Example 1: complete()")
    print("=" * 60)

    provider = make_provider()
    context = Context(
        system="You are a helpful assistant.",
        messages=[Message(role="user", content="What is the capital of France?")],
    )

    message = await complete(provider, context)
    print(message.extract_text())

    if message.usage:
        print(f"\nTokens: {message.usage.total_tokens}")

    print()


# ---------------------------------------------------------------------------
# Example 2: stream()
# ---------------------------------------------------------------------------


async def example_stream() -> None:
    """Stream text deltas in real time."""
    print("=" * 60)
    print("Example 2: stream()")
    print("=" * 60)

    provider = make_provider()
    context = Context(
        system="You are a helpful assistant.",
        messages=[Message(role="user", content="Tell me a short joke.")],
    )

    async for event in stream(provider, context):
        match event:
            case TextDelta(delta=text):
                print(text, end="", flush=True)
            case Done(message=msg):
                print(f"\n\nDone. Tokens: {msg.usage}")
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
