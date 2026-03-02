"""LLM contract and concrete implementations.

Each implementation handles the low-level interface for communicating with a
specific LLM API. They produce raw Chunk streams; ``stream()`` handles
accumulation.
"""

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from kai.chunk import Chunk
from kai.message import Context


@runtime_checkable
class LLM(Protocol):
    """Contract that all LLM implementations must satisfy.

    An implementation is responsible for:
    1. Converting kai types (Context, Tool, Message) into API-specific wire format
    2. Making API calls and streaming back raw Chunks
    3. Mapping API-specific errors to kai error types

    Example implementation::

        class MyLLM:
            @property
            def name(self) -> str:
                return "my-llm"

            @property
            def model(self) -> str:
                return self._model

            async def stream_raw(self, context: Context, **kwargs: Any) -> AsyncIterator[Chunk]:
                # call your API, yield TextChunk, ToolCallStart, etc.
                ...
    """

    @property
    def name(self) -> str:
        """The LLM name (e.g. 'openai', 'anthropic')."""
        ...

    @property
    def model(self) -> str:
        """The model identifier (e.g. 'gpt-4o', 'claude-sonnet-4-20250514')."""
        ...

    def stream_raw(self, context: Context, **kwargs: Any) -> AsyncIterator[Chunk]:
        """Stream raw chunks from the LLM.

        This is the core method implementations must provide. It should:
        1. Convert the Context into the API wire format
        2. Make the streaming API call
        3. Yield Chunk objects as they arrive
        4. Raise kai errors (ConnectionError, TimeoutError, etc.) on failure

        Args:
            context: The conversation context including system prompt, messages, and tools.
            **kwargs: API-specific options (temperature, max_tokens, etc.).

        Yields:
            Raw Chunk objects (TextChunk, ThinkChunk, ToolCallStart, etc.)
        """
        ...


__all__ = [
    "LLM",
]
