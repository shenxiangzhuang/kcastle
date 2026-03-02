"""Base abstractions for kai providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from kai.types.message import Context
from kai.types.stream import Chunk


class LLMBase(ABC):
    """Base contract for all kai provider implementations.

    A provider implementation is responsible for:
    1. Converting kai types (Context, Tool, Message) into API-specific wire format
    2. Making API calls and streaming back raw Chunks
    3. Mapping API-specific errors to kai error types
    """

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider identity (e.g. ``openai``, ``deepseek-openai``)."""

    @property
    def name(self) -> str:
        """Backward-compatible alias for ``provider``."""
        return self.provider

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier (e.g. ``gpt-4o``, ``claude-sonnet-4-20250514``)."""

    @abstractmethod
    def stream_raw(self, context: Context, **kwargs: Any) -> AsyncIterator[Chunk]:
        """Stream raw chunks from the provider."""
