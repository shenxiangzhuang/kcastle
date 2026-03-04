"""Base abstractions for kai providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Unpack

from typing_extensions import TypedDict

from kai.types.message import Context
from kai.types.stream import StreamEvent


class GenerationKwargs(TypedDict, total=False):
    """Optional generation parameters for LLM calls.

    Supported by all providers:
        temperature, max_tokens, top_p

    Provider-specific:
        stop        — OpenAI Chat Completions
        top_k       — Anthropic
        reasoning   — OpenAI Responses
    """

    temperature: float
    max_tokens: int
    top_p: float
    stop: str | list[str]
    top_k: int
    presence_penalty: float
    frequency_penalty: float
    reasoning: dict[str, Any]
    extra_body: dict[str, object]


class ProviderBase(ABC):
    """Base contract for all kai provider implementations.

    A provider implementation is responsible for:
    1. Converting kai types (Context, Tool, Message) into API-specific wire format
    2. Making API calls and streaming back StreamEvent objects
    3. Mapping API-specific errors to kai error types
    """

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider identity (e.g. ``openai``, ``deepseek-openai``)."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier (e.g. ``gpt-4o``, ``claude-sonnet-4-20250514``)."""

    @abstractmethod
    async def stream(
        self, context: Context, **kwargs: Unpack[GenerationKwargs]
    ) -> AsyncIterator[StreamEvent]:
        """Stream events from the provider."""
        raise NotImplementedError  # pragma: no cover
        yield  # pragma: no cover  # noqa: RET503
