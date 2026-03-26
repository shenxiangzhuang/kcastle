"""Tests for kagent.runtime — AgentRuntime actor."""

from __future__ import annotations

import pytest
from conftest import MockProvider, make_echo_tool, text_chunks, tool_call_chunks
from kai import Message

from kagent.agent import Agent
from kagent.event import AgentAbort, AgentEnd, AgentEvent, AgentStart, TurnEnd
from kagent.runtime import AgentRuntime
from kagent.signal import UserInput


class TestRuntimeSend:
    @pytest.mark.asyncio
    async def test_simple_send(self) -> None:
        provider = MockProvider([text_chunks("Hello!")])
        agent = Agent(llm=provider, system="You are helpful.")
        runtime = AgentRuntime(agent, can_spawn=False)
        await runtime.start()

        try:
            events = [e async for e in runtime.send(UserInput(text="Hi"))]

            assert isinstance(events[0], AgentStart)
            turn_ends = [e for e in events if isinstance(e, TurnEnd)]
            assert len(turn_ends) == 1
            assert turn_ends[0].message.extract_text() == "Hello!"
        finally:
            await runtime.stop()

    @pytest.mark.asyncio
    async def test_state_persists_across_sends(self) -> None:
        provider = MockProvider(
            [
                text_chunks("First"),
                text_chunks("Second"),
            ]
        )
        agent = Agent(llm=provider, system="Remember.")
        runtime = AgentRuntime(agent, can_spawn=False)
        await runtime.start()

        try:
            _ = [e async for e in runtime.send(UserInput(text="Message 1"))]
            _ = [e async for e in runtime.send(UserInput(text="Message 2"))]

            assert len(runtime.state.messages) == 4
        finally:
            await runtime.stop()

    @pytest.mark.asyncio
    async def test_send_with_tools(self) -> None:
        echo = make_echo_tool()
        provider = MockProvider(
            [
                tool_call_chunks("c1", "echo", '{"message": "test"}'),
                text_chunks("Tool said: echo: test"),
            ]
        )
        agent = Agent(llm=provider, tools=[echo])
        runtime = AgentRuntime(agent, can_spawn=False)
        await runtime.start()

        try:
            events = [e async for e in runtime.send(UserInput(text="Echo test"))]

            turn_ends = [e for e in events if isinstance(e, TurnEnd)]
            assert len(turn_ends) == 2  # tool call turn + final response turn
        finally:
            await runtime.stop()


class TestRuntimeAbort:
    @pytest.mark.asyncio
    async def test_abort_during_run(self) -> None:
        echo = make_echo_tool()
        provider = MockProvider(
            [
                tool_call_chunks("c1", "echo", '{"message": "hi"}'),
                tool_call_chunks("c2", "echo", '{"message": "bye"}'),
                text_chunks("Done"),
            ]
        )
        agent = Agent(llm=provider, tools=[echo])
        runtime = AgentRuntime(agent, can_spawn=False)
        await runtime.start()

        try:
            events: list[AgentEvent] = []
            async for event in runtime.send(UserInput(text="go")):
                events.append(event)
                # Abort after the first turn ends
                if isinstance(event, TurnEnd):
                    runtime.abort()

            abort_events = [e for e in events if isinstance(e, AgentAbort)]
            end_events = [e for e in events if isinstance(e, AgentEnd)]
            assert len(abort_events) == 1
            assert len(end_events) == 1
        finally:
            await runtime.stop()

    @pytest.mark.asyncio
    async def test_abort_before_send_is_noop(self) -> None:
        provider = MockProvider([text_chunks("Hi")])
        agent = Agent(llm=provider)
        runtime = AgentRuntime(agent, can_spawn=False)
        await runtime.start()

        try:
            runtime.abort()  # should not crash

            events = [e async for e in runtime.send(UserInput(text="Hello"))]
            turn_ends = [e for e in events if isinstance(e, TurnEnd)]
            assert len(turn_ends) == 1
        finally:
            await runtime.stop()


class TestRuntimeSteer:
    @pytest.mark.asyncio
    async def test_steer_injects_message(self) -> None:
        echo = make_echo_tool()
        provider = MockProvider(
            [
                tool_call_chunks("c1", "echo", '{"message": "hi"}'),
                text_chunks("Steered response"),
            ]
        )
        agent = Agent(llm=provider, tools=[echo])
        runtime = AgentRuntime(agent, can_spawn=False)
        await runtime.start()

        try:
            events: list[AgentEvent] = []
            async for event in runtime.send(UserInput(text="go")):
                events.append(event)
                if isinstance(event, TurnEnd) and not any(
                    isinstance(e, AgentEnd) for e in events
                ):
                    # Steer after first turn
                    runtime.steer(Message(role="user", content="Actually, do this instead."))

            turn_ends = [e for e in events if isinstance(e, TurnEnd)]
            # Should have at least 2 turns: original + steered
            assert len(turn_ends) >= 2
        finally:
            await runtime.stop()


class TestRuntimeLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        provider = MockProvider([])
        agent = Agent(llm=provider)
        runtime = AgentRuntime(agent, can_spawn=False)

        assert not runtime.is_running
        await runtime.start()
        assert runtime.is_running
        await runtime.stop()
        assert not runtime.is_running

    @pytest.mark.asyncio
    async def test_double_start_raises(self) -> None:
        provider = MockProvider([])
        agent = Agent(llm=provider)
        runtime = AgentRuntime(agent, can_spawn=False)
        await runtime.start()

        try:
            with pytest.raises(RuntimeError, match="already started"):
                await runtime.start()
        finally:
            await runtime.stop()

    @pytest.mark.asyncio
    async def test_properties(self) -> None:
        provider = MockProvider([])
        agent = Agent(llm=provider, system="test")
        runtime = AgentRuntime(agent, can_spawn=False)

        assert runtime.agent is agent
        assert runtime.state.system == "test"
        assert runtime.children == {}
