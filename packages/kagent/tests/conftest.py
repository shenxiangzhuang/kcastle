"""Shared test fixtures: mock provider and tools for kagent tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from kai import Context, Tool, ToolResult
from kai.providers.base import ProviderBase
from kai.types.stream import (
    StreamEvent,
    TextDelta,
    ToolCallBegin,
    ToolCallDelta,
    ToolCallEnd,
    Usage,
)
from kai.types.usage import TokenUsage
from pydantic import BaseModel, Field


class MockProvider(ProviderBase):
    """A mock provider that yields pre-configured events.

    Supports multiple turns via a list of event sequences.
    The first call to stream returns the first sequence, etc.
    """

    def __init__(
        self,
        turns: Sequence[Sequence[StreamEvent]],
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

    async def stream(self, context: Context, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        if self._call_index >= len(self._turns):
            raise RuntimeError("MockProvider exhausted: no more turns configured")
        events = self._turns[self._call_index]
        self._call_index += 1
        for event in events:
            yield event


class ErrorProvider(ProviderBase):
    """A mock provider that raises an error on stream."""

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

    async def stream(self, context: Context, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        raise self._error
        # Make this a valid async generator
        yield  # type: ignore[misc]  # pragma: no cover


def text_chunks(*texts: str) -> list[StreamEvent]:
    """Create events for a simple text response."""
    result: list[StreamEvent] = [TextDelta(delta=t) for t in texts]
    result.append(Usage(usage=TokenUsage(input_tokens=10, output_tokens=5)))
    return result


def tool_call_chunks(tool_id: str, tool_name: str, arguments_json: str) -> list[StreamEvent]:
    """Create events for a tool call response."""
    return [
        ToolCallBegin(id=tool_id, name=tool_name),
        ToolCallDelta(arguments=arguments_json),
        ToolCallEnd(),
        Usage(usage=TokenUsage(input_tokens=10, output_tokens=5)),
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
