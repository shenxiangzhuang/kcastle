"""Tests for kagent.step — Level 0: single-turn primitive."""

from __future__ import annotations

import pytest
from conftest import (
    MockProvider,
    make_echo_tool,
    make_error_tool,
    text_chunks,
    tool_call_chunks,
)
from kai import Context, Message, Tool

from kagent.event import (
    StreamChunk,
    ToolExecEnd,
    ToolExecStart,
    TurnEnd,
    TurnStart,
)
from kagent.step import agent_step


def _context(
    system: str | None = None,
    messages: list[Message] | None = None,
    tools: list[Tool] | None = None,
) -> Context:
    """Build a test Context."""
    return Context(
        system=system,
        messages=messages or [Message(role="user", content="hello")],
        tools=tools or [],
    )


class TestAgentStepTextOnly:
    @pytest.mark.asyncio
    async def test_simple_text_response(self) -> None:
        provider = MockProvider([text_chunks("Hello", " world")])
        ctx = _context()

        events = [e async for e in agent_step(llm=provider, context=ctx, tools=[])]

        assert isinstance(events[0], TurnStart)
        # StreamChunks in the middle
        stream_chunks = [e for e in events if isinstance(e, StreamChunk)]
        assert len(stream_chunks) > 0
        # Last event is TurnEnd
        assert isinstance(events[-1], TurnEnd)
        turn_end = events[-1]
        assert isinstance(turn_end, TurnEnd)
        assert turn_end.message.extract_text() == "Hello world"
        assert turn_end.tool_results == []


class TestAgentStepWithTools:
    @pytest.mark.asyncio
    async def test_tool_call_and_execution(self) -> None:
        echo = make_echo_tool()
        provider = MockProvider(
            [
                tool_call_chunks("call_1", "echo", '{"message": "hi"}'),
            ]
        )
        ctx = _context(tools=[echo])

        events = [e async for e in agent_step(llm=provider, context=ctx, tools=[echo])]

        # Should have: TurnStart, StreamChunks..., ToolExecStart, ToolExecEnd, TurnEnd
        assert isinstance(events[0], TurnStart)
        assert isinstance(events[-1], TurnEnd)

        exec_starts = [e for e in events if isinstance(e, ToolExecStart)]
        assert len(exec_starts) == 1
        assert exec_starts[0].tool_name == "echo"
        assert exec_starts[0].arguments == {"message": "hi"}

        exec_ends = [e for e in events if isinstance(e, ToolExecEnd)]
        assert len(exec_ends) == 1
        assert exec_ends[0].is_error is False
        assert exec_ends[0].result.output == "echo: hi"

        turn_end = events[-1]
        assert isinstance(turn_end, TurnEnd)
        assert len(turn_end.tool_results) == 1
        assert turn_end.tool_results[0].role == "tool"

    @pytest.mark.asyncio
    async def test_tool_not_found(self) -> None:
        provider = MockProvider(
            [
                tool_call_chunks("call_1", "nonexistent", '{"x": 1}'),
            ]
        )
        ctx = _context()

        events = [e async for e in agent_step(llm=provider, context=ctx, tools=[])]

        exec_ends = [e for e in events if isinstance(e, ToolExecEnd)]
        assert len(exec_ends) == 1
        assert exec_ends[0].is_error is True
        assert "not found" in exec_ends[0].result.output.lower()

    @pytest.mark.asyncio
    async def test_tool_execution_error(self) -> None:
        failing = make_error_tool()
        provider = MockProvider(
            [
                tool_call_chunks("call_1", "failing_tool", '{"input": "test"}'),
            ]
        )
        ctx = _context(tools=[failing])

        events = [e async for e in agent_step(llm=provider, context=ctx, tools=[failing])]

        exec_ends = [e for e in events if isinstance(e, ToolExecEnd)]
        assert len(exec_ends) == 1
        assert exec_ends[0].is_error is True

    @pytest.mark.asyncio
    async def test_invalid_json_arguments(self) -> None:
        echo = make_echo_tool()
        provider = MockProvider(
            [
                tool_call_chunks("call_1", "echo", "not valid json"),
            ]
        )
        ctx = _context(tools=[echo])

        events = [e async for e in agent_step(llm=provider, context=ctx, tools=[echo])]

        exec_ends = [e for e in events if isinstance(e, ToolExecEnd)]
        assert len(exec_ends) == 1
        assert exec_ends[0].is_error is True
        assert "Invalid JSON" in exec_ends[0].result.output
