"""Tests for kagent.state module."""

from kai import Message

from kagent.state import AgentState


class TestAgentState:
    def test_default_state(self) -> None:
        state = AgentState()
        assert state.system is None
        assert state.messages == []
        assert state.tools == []

    def test_with_values(self) -> None:
        state = AgentState(
            system="You are helpful.",
            messages=[Message(role="user", content="hi")],
        )
        assert state.system == "You are helpful."
        assert len(state.messages) == 1

    def test_mutable_messages(self) -> None:
        state = AgentState()
        state.messages.append(Message(role="user", content="hello"))
        assert len(state.messages) == 1

    def test_mutable_system(self) -> None:
        state = AgentState(system="v1")
        state.system = "v2"
        assert state.system == "v2"

    def test_independent_instances(self) -> None:
        """Each instance has its own list (no shared default)."""
        s1 = AgentState()
        s2 = AgentState()
        s1.messages.append(Message(role="user", content="x"))
        assert len(s2.messages) == 0
