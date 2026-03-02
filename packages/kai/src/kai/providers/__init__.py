"""Provider protocol and provider implementations.

Providers implement the low-level interface for communicating with LLM APIs.
They produce raw Chunk streams; the stream() function handles accumulation.
"""

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from kai.chunk import Chunk
from kai.message import Context
from kai.providers.factory import (
    ProviderProfile,
    ProviderRegistry,
    create_provider,
)


@runtime_checkable
class Provider(Protocol):
    """Protocol that all LLM providers must implement.

    A provider is responsible for:
    1. Converting kai types (Context, Tool, Message) into provider-specific wire format
    2. Making API calls and streaming back raw Chunks
    3. Mapping provider-specific errors to kai error types

    Example implementation::

        class MyProvider:
            @property
            def name(self) -> str:
                return "my-provider"

            @property
            def model(self) -> str:
                return self._model

            async def stream_raw(self, context: Context, **kwargs: Any) -> AsyncIterator[Chunk]:
                # call your API, yield TextChunk, ToolCallStart, etc.
                ...
    """

    @property
    def name(self) -> str:
        """The provider name (e.g. 'openai', 'anthropic')."""
        ...

    @property
    def model(self) -> str:
        """The model identifier (e.g. 'gpt-4o', 'claude-sonnet-4-20250514')."""
        ...

    def stream_raw(self, context: Context, **kwargs: Any) -> AsyncIterator[Chunk]:
        """Stream raw chunks from the LLM.

        This is the core method providers must implement. It should:
        1. Convert the Context into the provider's wire format
        2. Make the streaming API call
        3. Yield Chunk objects as they arrive
        4. Raise kai errors (ConnectionError, TimeoutError, etc.) on failure

        Args:
            context: The conversation context including system prompt, messages, and tools.
            **kwargs: Provider-specific options (temperature, max_tokens, etc.).

        Yields:
            Raw Chunk objects (TextChunk, ThinkChunk, ToolCallStart, etc.)
        """
        ...


__all__ = [
    "Provider",
    "ProviderProfile",
    "ProviderRegistry",
    "create_provider",
]
