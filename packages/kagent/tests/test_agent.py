"""Tests for kagent.agent — Level 2: stateful agent SDK."""

from __future__ import annotations

import pytest
from conftest import (
    ErrorProvider,
    MockProvider,
    make_echo_tool,
    text_chunks,
    tool_call_chunks,
)
from kai import Message, ToolResult

from kagent.agent import Agent
from kagent.event import AgentAbort, AgentEnd, AgentError, AgentEvent, AgentStart, TurnEnd


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

    def test_provider_setter_is_alias_for_replace(self) -> None:
        provider1 = MockProvider([])
        provider2 = MockProvider([])
        agent = Agent(provider=provider1)

        agent.provider = provider2

        assert agent.provider is provider2

    def test_provider_setter_raises_when_running(self) -> None:
        provider1 = MockProvider([])
        provider2 = MockProvider([])
        agent = Agent(provider=provider1)
        agent._running = True  # pyright: ignore[reportPrivateUsage]

        with pytest.raises(RuntimeError, match="Cannot replace provider while agent is running"):
            agent.provider = provider2


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


class TestAgentAbort:
    @pytest.mark.asyncio
    async def test_abort_emits_agent_abort_and_agent_end(self) -> None:
        """abort() should emit AgentAbort followed by AgentEnd, never silently exit."""
        echo = make_echo_tool()
        provider = MockProvider(
            [
                # Turn 1: tool call — agent will loop
                tool_call_chunks("c1", "echo", '{"message": "hi"}'),
                # Turn 2: another tool call — but abort fires before this
                tool_call_chunks("c2", "echo", '{"message": "bye"}'),
                text_chunks("Done"),
            ]
        )
        agent = Agent(provider=provider, tools=[echo])

        events: list[AgentEvent] = []
        async for event in agent.run("go"):
            events.append(event)
            # Abort after the first turn ends
            if isinstance(event, TurnEnd):
                agent.abort()

        # Should have AgentAbort followed by AgentEnd
        abort_events = [e for e in events if isinstance(e, AgentAbort)]
        end_events = [e for e in events if isinstance(e, AgentEnd)]
        assert len(abort_events) == 1, "Expected exactly one AgentAbort event"
        assert len(end_events) == 1, "Expected exactly one AgentEnd event"

        # AgentAbort should come before AgentEnd
        abort_idx = events.index(abort_events[0])
        end_idx = events.index(end_events[0])
        assert abort_idx < end_idx, "AgentAbort should precede AgentEnd"

        # Both should carry the current messages
        assert len(abort_events[0].messages) > 0
        assert len(end_events[0].messages) > 0

    @pytest.mark.asyncio
    async def test_abort_before_run_is_noop(self) -> None:
        """abort() before run starts should be a no-op."""
        provider = MockProvider([text_chunks("Hi")])
        agent = Agent(provider=provider)

        # abort before run — should not crash
        agent.abort()

        msg = await agent.complete("Hello")
        assert msg.extract_text() == "Hi"

    @pytest.mark.asyncio
    async def test_abort_stops_agent_running_flag(self) -> None:
        """After abort, is_running should be False."""
        provider = MockProvider(
            [
                tool_call_chunks("c1", "echo", '{"message": "hi"}'),
                text_chunks("Done"),
            ]
        )
        echo = make_echo_tool()
        agent = Agent(provider=provider, tools=[echo])

        async for event in agent.run("go"):
            if isinstance(event, TurnEnd):
                agent.abort()

        assert agent.is_running is False


class TestAgentErrorPropagation:
    @pytest.mark.asyncio
    async def test_complete_raises_on_provider_error(self) -> None:
        """complete() should raise RuntimeError with the original error as cause."""
        from kai.errors import ProviderError

        provider = ErrorProvider(ProviderError("API connection failed"))
        agent = Agent(provider=provider, system="test")

        with pytest.raises(RuntimeError, match="API connection failed") as exc_info:
            await agent.complete("Hello")

        assert isinstance(exc_info.value.__cause__, ProviderError)

    @pytest.mark.asyncio
    async def test_run_yields_agent_error_on_provider_error(self) -> None:
        """run() should yield AgentError when the provider raises."""
        from kai.errors import ProviderError

        provider = ErrorProvider(ProviderError("stream broke"))
        agent = Agent(provider=provider, system="test")

        events = [e async for e in agent.run("Hello")]

        error_events = [e for e in events if isinstance(e, AgentError)]
        assert len(error_events) == 1
        assert "stream broke" in str(error_events[0].error)

        end_events = [e for e in events if isinstance(e, AgentEnd)]
        assert len(end_events) == 1

    @pytest.mark.asyncio
    async def test_complete_raises_on_empty_response(self) -> None:
        """complete() should raise when the provider returns no content."""
        from kai.chunk import UsageChunk
        from kai.usage import TokenUsage

        # A provider that returns only a usage chunk (no text, no tool call).
        provider = MockProvider([[UsageChunk(usage=TokenUsage(input_tokens=0, output_tokens=0))]])
        agent = Agent(provider=provider, system="test")

        with pytest.raises(RuntimeError, match="Agent loop failed"):
            await agent.complete("Hello")
