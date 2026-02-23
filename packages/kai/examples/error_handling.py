"""Error handling — graceful handling of provider errors.

Run:
    export DEEPSEEK_API_KEY=sk-...   # default
    # export OPENAI_API_KEY=sk-...   # if using OpenAI
    uv run python examples/error_handling.py
"""

import asyncio

from kai import (
    Context,
    DoneEvent,
    ErrorEvent,
    Message,
    OpenAICompletions,
    stream,
)
from kai.errors import ConnectionError, ProviderError, StatusError, TimeoutError


async def example_complete_error_handling() -> None:
    """Error handling with complete() — use try/except."""
    from kai import complete

    # provider = OpenAICompletions(model="gpt-4o", api_key="invalid-key")
    provider = OpenAICompletions(
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
    except StatusError as e:
        print(f"HTTP {e.status_code}: {e}")
    except ConnectionError:
        print("Could not connect to the API")
    except TimeoutError:
        print("Request timed out")
    except ProviderError as e:
        print(f"Provider error: {e}")


async def example_stream_error_handling() -> None:
    """Error handling with stream() — errors arrive as ErrorEvent."""
    # provider = OpenAICompletions(model="gpt-4o", api_key="invalid-key")
    provider = OpenAICompletions(
        model="deepseek-chat",
        api_key="invalid-key",
        base_url="https://api.deepseek.com",
    )
    context = Context(
        messages=[Message(role="user", content="Hello")],
    )

    async for event in await stream(provider, context):
        match event:
            case ErrorEvent(error=error):
                print(f"Stream error: {error}")
                # error.partial contains the partial message accumulated so far
            case DoneEvent(message=msg):
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
