"""Tests for kagent.agent — pure handler with single handle() interface."""

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

from kagent.agent import Agent, complete
from kagent.event import AgentEnd, AgentError, AgentStart, TurnEnd
from kagent.state import AgentState
from kagent.trace.entry import TraceEntry


class TestAgentHandle:
    @pytest.mark.asyncio
    async def test_simple_handle(self) -> None:
        provider = MockProvider([text_chunks("Hello!")])
        agent = Agent(llm=provider, system="You are helpful.")
        state = AgentState(system=agent.system, tools=list(agent.tools))
        state.trace.append(TraceEntry.user(Message(role="user", content="Hi")))

        events = [e async for e in agent.handle(state)]

        assert isinstance(events[0], AgentStart)
        turn_ends = [e for e in events if isinstance(e, TurnEnd)]
        assert len(turn_ends) == 1
        assert turn_ends[0].message.extract_text() == "Hello!"

    @pytest.mark.asyncio
    async def test_state_persists_across_handles(self) -> None:
        provider = MockProvider(
            [
                text_chunks("First response"),
                text_chunks("Second response"),
            ]
        )
        agent = Agent(llm=provider, system="Remember things.")
        state = AgentState(system=agent.system, tools=list(agent.tools))

        # First interaction
        state.trace.append(TraceEntry.user(Message(role="user", content="Message 1")))
        _ = [e async for e in agent.handle(state)]

        # Second interaction
        state.trace.append(TraceEntry.user(Message(role="user", content="Message 2")))
        _ = [e async for e in agent.handle(state)]

        # Should have 4 messages: user1, assistant1, user2, assistant2
        assert len(state.messages) == 4
        assert state.messages[0].role == "user"
        assert state.messages[1].role == "assistant"
        assert state.messages[2].role == "user"
        assert state.messages[3].role == "assistant"


class TestComplete:
    @pytest.mark.asyncio
    async def test_complete_returns_message(self) -> None:
        provider = MockProvider([text_chunks("The answer is 4.")])
        agent = Agent(llm=provider, system="You are a calculator.")

        msg = await complete(agent, "What's 2+2?")
        assert msg.extract_text() == "The answer is 4."

    @pytest.mark.asyncio
    async def test_complete_multi_turn_with_shared_state(self) -> None:
        provider = MockProvider(
            [
                text_chunks("I'll remember that."),
                text_chunks("Your name is Alice."),
            ]
        )
        agent = Agent(llm=provider)
        state = AgentState(system=agent.system, tools=list(agent.tools))

        await complete(agent, "My name is Alice.", state=state)
        msg = await complete(agent, "What's my name?", state=state)
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
        agent = Agent(llm=provider, tools=[echo])

        msg = await complete(agent, "Echo test")
        assert msg.extract_text() == "Tool said: echo: test"


class TestAgentConfig:
    def test_agent_stores_config(self) -> None:
        provider = MockProvider([])
        agent = Agent(llm=provider, system="Hello", max_turns=50)
        assert agent.system == "Hello"
        assert agent.max_turns == 50
        assert agent.llm is provider
        assert agent.tools == []

    def test_llm_is_mutable(self) -> None:
        provider1 = MockProvider([])
        provider2 = MockProvider([])
        agent = Agent(llm=provider1)

        agent.llm = provider2

        assert agent.llm is provider2


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
        agent = Agent(llm=provider, tools=[echo], on_tool_result=modify)
        state = AgentState(system=agent.system, tools=list(agent.tools))
        state.trace.append(TraceEntry.user(Message(role="user", content="go")))

        _ = [e async for e in agent.handle(state)]
        # The tool result should have been intercepted
        assert state.messages[2].role == "tool"


class TestAgentErrorPropagation:
    @pytest.mark.asyncio
    async def test_complete_raises_on_provider_error(self) -> None:
        from kai.errors import ErrorKind, KaiError

        provider = ErrorProvider(KaiError(ErrorKind.PROVIDER, "API connection failed"))
        agent = Agent(llm=provider, system="test")

        with pytest.raises(RuntimeError, match="API connection failed") as exc_info:
            await complete(agent, "Hello")

        assert isinstance(exc_info.value.__cause__, KaiError)

    @pytest.mark.asyncio
    async def test_handle_yields_agent_error_on_provider_error(self) -> None:
        from kai.errors import ErrorKind, KaiError

        provider = ErrorProvider(KaiError(ErrorKind.PROVIDER, "stream broke"))
        agent = Agent(llm=provider, system="test")
        state = AgentState(system=agent.system, tools=list(agent.tools))
        state.trace.append(TraceEntry.user(Message(role="user", content="Hello")))

        events = [e async for e in agent.handle(state)]

        error_events = [e for e in events if isinstance(e, AgentError)]
        assert len(error_events) == 1
        assert "stream broke" in str(error_events[0].error)

        end_events = [e for e in events if isinstance(e, AgentEnd)]
        assert len(end_events) == 1

    @pytest.mark.asyncio
    async def test_complete_raises_on_empty_response(self) -> None:
        from kai.types.stream import Usage
        from kai.types.usage import TokenUsage

        provider = MockProvider([[Usage(usage=TokenUsage(input_tokens=0, output_tokens=0))]])
        agent = Agent(llm=provider, system="test")

        with pytest.raises(RuntimeError, match="Agent failed"):
            await complete(agent, "Hello")
