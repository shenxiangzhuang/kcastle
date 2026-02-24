"""Shared test fixtures: mock provider and tools for kagent tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from kai import Context, Tool, ToolResult
from kai.chunk import Chunk, TextChunk, ToolCallDelta, ToolCallEnd, ToolCallStart, UsageChunk
from kai.usage import TokenUsage


class MockProvider:
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
        self._name = name
        self._model = model

    @property
    def name(self) -> str:
        return self._name

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
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    async def execute(self, *, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output=f"echo: {arguments['message']}")


class FailingTool(Tool):
    """A tool that always raises an error."""

    name: str = "failing_tool"
    description: str = "A tool that fails."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
    }

    async def execute(self, *, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        raise ValueError("Tool execution failed!")


def make_echo_tool() -> Tool:
    """Create a simple echo tool for tests."""
    return EchoTool()


def make_error_tool() -> Tool:
    """Create a tool that always raises an error."""
    return FailingTool()
