"""Tests for kagent.event module."""

from kai import Message, StartEvent, ToolResult

from kagent.event import (
    AgentEnd,
    AgentError,
    AgentEvent,
    AgentStart,
    StreamChunk,
    ToolExecEnd,
    ToolExecStart,
    TurnEnd,
    TurnStart,
)


class TestEventTypes:
    def test_turn_start(self) -> None:
        ev = TurnStart()
        assert ev.type == "turn_start"

    def test_turn_end(self) -> None:
        msg = Message(role="assistant", content="hello")
        ev = TurnEnd(message=msg, tool_results=[])
        assert ev.type == "turn_end"
        assert ev.message.extract_text() == "hello"
        assert ev.tool_results == []

    def test_stream_chunk(self) -> None:
        kai_event = StartEvent()
        ev = StreamChunk(event=kai_event)
        assert ev.type == "stream_chunk"
        assert ev.event is kai_event

    def test_tool_exec_start(self) -> None:
        ev = ToolExecStart(call_id="c1", tool_name="test", arguments={"x": 1})
        assert ev.type == "tool_exec_start"
        assert ev.call_id == "c1"
        assert ev.tool_name == "test"

    def test_tool_exec_end(self) -> None:
        result = ToolResult(output="ok")
        ev = ToolExecEnd(call_id="c1", tool_name="test", result=result, is_error=False)
        assert ev.type == "tool_exec_end"
        assert ev.is_error is False

    def test_agent_start(self) -> None:
        ev = AgentStart()
        assert ev.type == "agent_start"

    def test_agent_end(self) -> None:
        msgs = [Message(role="user", content="hi")]
        ev = AgentEnd(messages=msgs)
        assert ev.type == "agent_end"
        assert len(ev.messages) == 1

    def test_agent_error(self) -> None:
        err = RuntimeError("boom")
        ev = AgentError(error=err)
        assert ev.type == "agent_error"
        assert ev.error is err


class TestAgentEventUnion:
    def test_pattern_matching(self) -> None:
        event: AgentEvent = TurnStart()
        match event:
            case TurnStart():
                matched = True
            case _:
                matched = False
        assert matched
