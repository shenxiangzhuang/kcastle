"""Shared test fixtures: mock provider and tools for kagent tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from kai import Context, Tool, ToolResult
from kai.providers.base import LLMBase
from kai.types.stream import Chunk, TextChunk, ToolCallDelta, ToolCallEnd, ToolCallStart, UsageChunk
from kai.types.usage import TokenUsage
from pydantic import BaseModel, Field


class MockProvider(LLMBase):
    """A mock provider that yields pre-configured chunks.

    Supports multiple turns via a list of chunk sequences.
    The first call to stream_raw returns the first sequence, etc.
    """

    def __init__(
        self,
        turns: Sequence[Sequence[Chunk]],
        *,
        name: str = "mock",
        model: str = "mock-1",
    ) -> None:
        self._turns = list(turns)
        self._call_index = 0
        self._provider = name
        self._model = model

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    async def stream_raw(self, context: Context, **kwargs: Any) -> AsyncIterator[Chunk]:
        if self._call_index >= len(self._turns):
            raise RuntimeError("MockProvider exhausted: no more turns configured")
        chunks = self._turns[self._call_index]
        self._call_index += 1
        for chunk in chunks:
            yield chunk


class ErrorProvider(LLMBase):
    """A mock provider that raises an error on stream_raw."""

    def __init__(self, error: Exception, *, name: str = "error", model: str = "error-1") -> None:
        self._error = error
        self._provider = name
        self._model = model

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    async def stream_raw(self, context: Context, **kwargs: Any) -> AsyncIterator[Chunk]:
        raise self._error
        # Make this a valid async generator
        yield  # type: ignore[misc]  # pragma: no cover


def text_chunks(*texts: str) -> list[Chunk]:
    """Create chunks for a simple text response."""
    result: list[Chunk] = [TextChunk(text=t) for t in texts]
    result.append(UsageChunk(usage=TokenUsage(input_tokens=10, output_tokens=5)))
    return result


def tool_call_chunks(tool_id: str, tool_name: str, arguments_json: str) -> list[Chunk]:
    """Create chunks for a tool call response."""
    return [
        ToolCallStart(id=tool_id, name=tool_name),
        ToolCallDelta(arguments=arguments_json),
        ToolCallEnd(),
        UsageChunk(usage=TokenUsage(input_tokens=10, output_tokens=5)),
    ]


class EchoTool(Tool):
    """A simple echo tool for tests."""

    name: str = "echo"
    description: str = "Echo a message."

    class Params(BaseModel):
        message: str = Field(description="The message to echo")

    async def execute(self, params: EchoTool.Params) -> ToolResult:
        return ToolResult(output=f"echo: {params.message}")


class FailingTool(Tool):
    """A tool that always raises an error."""

    name: str = "failing_tool"
    description: str = "A tool that fails."

    class Params(BaseModel):
        input: str = Field(description="Input value")

    async def execute(self, params: FailingTool.Params) -> ToolResult:
        raise ValueError("Tool execution failed!")


def make_echo_tool() -> Tool:
    """Create a simple echo tool for tests."""
    return EchoTool()


def make_error_tool() -> Tool:
    """Create a tool that always raises an error."""
    return FailingTool()
