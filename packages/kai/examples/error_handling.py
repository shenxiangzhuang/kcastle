"""Error handling — graceful handling of provider errors.

Run:
    export DEEPSEEK_API_KEY=sk-...   # default
    # export OPENAI_API_KEY=sk-...   # if using OpenAI
    uv run python examples/error_handling.py
"""

import asyncio

from kai import (
    Context,
    Done,
    Error,
    ErrorKind,
    KaiError,
    Message,
    OpenAIChatCompletions,
    stream,
)


async def example_complete_error_handling() -> None:
    """Error handling with complete() — use try/except."""
    from kai import complete

    # provider = OpenAIChatCompletions(model="gpt-4o", api_key="invalid-key")
    provider = OpenAIChatCompletions(
        model="deepseek-chat",
        api_key="invalid-key",
        base_url="https://api.deepseek.com",
    )
    context = Context(
        messages=[Message(role="user", content="Hello")],
    )

    try:
        message = await complete(provider, context)
        print(message.extract_text())
    except KaiError as e:
        match e.kind:
            case ErrorKind.STATUS:
                print(f"Status error: {e}")
            case ErrorKind.CONNECTION:
                print("Could not connect to the API")
            case ErrorKind.TIMEOUT:
                print("Request timed out")
            case _:
                print(f"Provider error: {e}")


async def example_stream_error_handling() -> None:
    """Error handling with stream() — errors arrive as Error event."""
    # provider = OpenAIChatCompletions(model="gpt-4o", api_key="invalid-key")
    provider = OpenAIChatCompletions(
        model="deepseek-chat",
        api_key="invalid-key",
        base_url="https://api.deepseek.com",
    )
    context = Context(
        messages=[Message(role="user", content="Hello")],
    )

    async for event in stream(provider, context):
        match event:
            case Error(error=error):
                print(f"Stream error: {error}")
            case Done(message=msg):
                print(msg.extract_text())
            case _:
                pass


async def main() -> None:
    print("=== complete() error handling ===")
    await example_complete_error_handling()

    print("\n=== stream() error handling ===")
    await example_stream_error_handling()


if __name__ == "__main__":
    asyncio.run(main())
