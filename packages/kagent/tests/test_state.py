"""Tests for kagent.state module."""

from kai import Message

from kagent.state import AgentState
from kagent.trace import Trace, TraceEntry


class TestAgentState:
    def test_default_state(self) -> None:
        state = AgentState()
        assert state.system is None
        assert state.messages == []
        assert state.tools == []

    def test_with_values(self) -> None:
        trace = Trace()
        trace.append(TraceEntry.user(Message(role="user", content="hi")))
        state = AgentState(
            system="You are helpful.",
            trace=trace,
        )
        assert state.system == "You are helpful."
        assert len(state.messages) == 1

    def test_messages_derived_from_trace(self) -> None:
        state = AgentState()
        msg = Message(role="user", content="hello")
        state.trace.append(TraceEntry.user(msg))
        assert len(state.messages) == 1
        assert state.messages[0].extract_text() == "hello"

    def test_mutable_system(self) -> None:
        state = AgentState(system="v1")
        state.system = "v2"
        assert state.system == "v2"

    def test_independent_instances(self) -> None:
        """Each instance has its own trace (no shared default)."""
        s1 = AgentState()
        s2 = AgentState()
        s1.trace.append(TraceEntry.user(Message(role="user", content="x")))
        assert len(s2.messages) == 0

    def test_trace_identity(self) -> None:
        """Each trace has a unique ID."""
        s1 = AgentState()
        s2 = AgentState()
        assert s1.trace.id != s2.trace.id
