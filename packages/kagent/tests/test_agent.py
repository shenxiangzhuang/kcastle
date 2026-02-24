"""Tests for kagent.agent — Level 2: stateful agent SDK."""

from __future__ import annotations

import pytest
from conftest import (
    MockProvider,
    make_echo_tool,
    text_chunks,
    tool_call_chunks,
)
from kai import Message, ToolResult

from kagent.agent import Agent
from kagent.event import AgentStart, TurnEnd


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_simple_run(self) -> None:
        provider = MockProvider([text_chunks("Hello!")])
        agent = Agent(provider=provider, system="You are helpful.")

        events = [e async for e in agent.run("Hi")]

        assert isinstance(events[0], AgentStart)
        turn_ends = [e for e in events if isinstance(e, TurnEnd)]
        assert len(turn_ends) == 1
        assert turn_ends[0].message.extract_text() == "Hello!"

    @pytest.mark.asyncio
    async def test_state_persists_across_runs(self) -> None:
        provider = MockProvider(
            [
                text_chunks("First response"),
                text_chunks("Second response"),
            ]
        )
        agent = Agent(provider=provider, system="Remember things.")

        _ = [e async for e in agent.run("Message 1")]
        _ = [e async for e in agent.run("Message 2")]

        # Should have 4 messages: user1, assistant1, user2, assistant2
        assert len(agent.state.messages) == 4
        assert agent.state.messages[0].role == "user"
        assert agent.state.messages[1].role == "assistant"
        assert agent.state.messages[2].role == "user"
        assert agent.state.messages[3].role == "assistant"


class TestAgentComplete:
    @pytest.mark.asyncio
    async def test_complete_returns_message(self) -> None:
        provider = MockProvider([text_chunks("The answer is 4.")])
        agent = Agent(provider=provider, system="You are a calculator.")

        msg = await agent.complete("What's 2+2?")
        assert msg.extract_text() == "The answer is 4."

    @pytest.mark.asyncio
    async def test_complete_multi_turn(self) -> None:
        provider = MockProvider(
            [
                text_chunks("I'll remember that."),
                text_chunks("Your name is Alice."),
            ]
        )
        agent = Agent(provider=provider)

        await agent.complete("My name is Alice.")
        msg = await agent.complete("What's my name?")
        assert msg.extract_text() == "Your name is Alice."


class TestAgentWithTools:
    @pytest.mark.asyncio
    async def test_complete_with_tool_calls(self) -> None:
        echo = make_echo_tool()
        provider = MockProvider(
            [
                tool_call_chunks("c1", "echo", '{"message": "test"}'),
                text_chunks("Tool said: echo: test"),
            ]
        )
        agent = Agent(provider=provider, tools=[echo])

        msg = await agent.complete("Echo test")
        assert msg.extract_text() == "Tool said: echo: test"


class TestAgentState:
    def test_state_is_accessible(self) -> None:
        provider = MockProvider([])
        agent = Agent(provider=provider, system="Hello")
        assert agent.state.system == "Hello"
        assert agent.state.messages == []

    def test_state_is_mutable(self) -> None:
        provider = MockProvider([])
        agent = Agent(provider=provider, system="v1")
        agent.state.system = "v2"
        assert agent.state.system == "v2"

    def test_is_running_initially_false(self) -> None:
        provider = MockProvider([])
        agent = Agent(provider=provider)
        assert agent.is_running is False


class TestAgentFollowUp:
    @pytest.mark.asyncio
    async def test_follow_up_triggers_additional_run(self) -> None:
        provider = MockProvider(
            [
                text_chunks("First done."),
                text_chunks("Follow-up done."),
            ]
        )
        agent = Agent(provider=provider)

        # Queue a follow-up before running
        agent.follow_up(Message(role="user", content="And do this too."))

        events = [e async for e in agent.run("Do something.")]

        turn_ends = [e for e in events if isinstance(e, TurnEnd)]
        # Should have 2 turns: original + follow-up
        assert len(turn_ends) == 2
        assert turn_ends[0].message.extract_text() == "First done."
        assert turn_ends[1].message.extract_text() == "Follow-up done."


class TestAgentCallbacks:
    @pytest.mark.asyncio
    async def test_on_tool_result_callback(self) -> None:
        echo = make_echo_tool()

        async def modify(call_id: str, name: str, result: ToolResult) -> ToolResult:
            return ToolResult(output="intercepted!")

        provider = MockProvider(
            [
                tool_call_chunks("c1", "echo", '{"message": "hi"}'),
                text_chunks("Done"),
            ]
        )
        agent = Agent(provider=provider, tools=[echo], on_tool_result=modify)

        await agent.complete("go")
        # The tool result should have been intercepted
        assert agent.state.messages[2].role == "tool"
