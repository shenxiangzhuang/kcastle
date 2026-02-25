"""Tests for kagent.loop — Level 1: multi-turn loop."""

from __future__ import annotations

import pytest
from conftest import (
    MockProvider,
    make_echo_tool,
    text_chunks,
    tool_call_chunks,
)
from kai import Context, Message, ToolResult

from kagent.event import (
    AgentEnd,
    AgentStart,
    ToolExecEnd,
    TurnEnd,
)
from kagent.loop import agent_loop
from kagent.state import AgentState


class TestAgentLoopTextOnly:
    @pytest.mark.asyncio
    async def test_single_turn_text(self) -> None:
        provider = MockProvider([text_chunks("Hello!")])
        state = AgentState(
            system="You are helpful.",
            messages=[Message(role="user", content="Hi")],
        )

        events = [e async for e in agent_loop(provider=provider, state=state)]

        assert isinstance(events[0], AgentStart)
        assert isinstance(events[-1], AgentEnd)

        turn_ends = [e for e in events if isinstance(e, TurnEnd)]
        assert len(turn_ends) == 1
        assert turn_ends[0].message.extract_text() == "Hello!"

        # State should be mutated
        assert len(state.messages) == 2  # user + assistant
        assert state.messages[1].role == "assistant"


class TestAgentLoopWithTools:
    @pytest.mark.asyncio
    async def test_tool_call_then_text_response(self) -> None:
        echo = make_echo_tool()
        provider = MockProvider(
            [
                # Turn 1: LLM calls the echo tool
                tool_call_chunks("call_1", "echo", '{"message": "hi"}'),
                # Turn 2: LLM responds with text (no more tool calls)
                text_chunks("Done!"),
            ]
        )
        state = AgentState(
            system="Use tools.",
            messages=[Message(role="user", content="echo something")],
            tools=[echo],
        )

        events = [e async for e in agent_loop(provider=provider, state=state)]

        turn_ends = [e for e in events if isinstance(e, TurnEnd)]
        assert len(turn_ends) == 2

        # First turn: tool was called
        assert len(turn_ends[0].tool_results) == 1

        # Second turn: text response
        assert turn_ends[1].message.extract_text() == "Done!"
        assert turn_ends[1].tool_results == []

        # State should have: user, assistant(tool_call), tool_result, assistant(text)
        assert len(state.messages) == 4
        assert state.messages[0].role == "user"
        assert state.messages[1].role == "assistant"
        assert state.messages[2].role == "tool"
        assert state.messages[3].role == "assistant"


class TestAgentLoopMaxTurns:
    @pytest.mark.asyncio
    async def test_max_turns_limits_loop(self) -> None:
        echo = make_echo_tool()
        # Provider always returns tool calls
        provider = MockProvider(
            [
                tool_call_chunks("c1", "echo", '{"message": "1"}'),
                tool_call_chunks("c2", "echo", '{"message": "2"}'),
                tool_call_chunks("c3", "echo", '{"message": "3"}'),
            ]
        )
        state = AgentState(
            messages=[Message(role="user", content="go")],
            tools=[echo],
        )

        events = [e async for e in agent_loop(provider=provider, state=state, max_turns=2)]

        turn_ends = [e for e in events if isinstance(e, TurnEnd)]
        assert len(turn_ends) == 2  # Stopped at max_turns


class TestAgentLoopCallbacks:
    @pytest.mark.asyncio
    async def test_context_builder(self) -> None:
        """A ContextBuilder controls what goes to the LLM."""
        captured_contexts: list[Context] = []

        class CustomBuilder:
            async def build(self, state: AgentState) -> Context:
                ctx = Context(
                    system="Custom system!",
                    messages=state.messages[-1:],  # Only last message
                    tools=list(state.tools),
                )
                captured_contexts.append(ctx)
                return ctx

        provider = MockProvider([text_chunks("Ok")])
        state = AgentState(
            system="Original system",
            messages=[
                Message(role="user", content="old"),
                Message(role="user", content="new"),
            ],
        )

        [
            e
            async for e in agent_loop(
                provider=provider, state=state, context_builder=CustomBuilder()
            )
        ]

        assert len(captured_contexts) == 1
        assert captured_contexts[0].system == "Custom system!"
        # Only the last message should have been sent
        assert len(captured_contexts[0].messages) == 1

    @pytest.mark.asyncio
    async def test_on_tool_result_callback(self) -> None:
        """on_tool_result can modify tool results."""
        echo = make_echo_tool()

        async def modify_result(call_id: str, tool_name: str, result: ToolResult) -> ToolResult:
            return ToolResult(output="MODIFIED")

        provider = MockProvider(
            [
                tool_call_chunks("c1", "echo", '{"message": "hi"}'),
                text_chunks("Ok"),
            ]
        )
        state = AgentState(
            messages=[Message(role="user", content="go")],
            tools=[echo],
        )

        events = [
            e
            async for e in agent_loop(provider=provider, state=state, on_tool_result=modify_result)
        ]

        exec_ends = [e for e in events if isinstance(e, ToolExecEnd)]
        assert len(exec_ends) == 1
        assert exec_ends[0].result.output == "MODIFIED"

    @pytest.mark.asyncio
    async def test_should_continue_callback(self) -> None:
        """should_continue can stop the loop early."""
        echo = make_echo_tool()

        async def stop_always(state: AgentState, msg: Message) -> bool:
            return False

        provider = MockProvider(
            [
                tool_call_chunks("c1", "echo", '{"message": "hi"}'),
            ]
        )
        state = AgentState(
            messages=[Message(role="user", content="go")],
            tools=[echo],
        )

        events = [
            e async for e in agent_loop(provider=provider, state=state, should_continue=stop_always)
        ]

        turn_ends = [e for e in events if isinstance(e, TurnEnd)]
        assert len(turn_ends) == 1  # Only one turn despite tool calls
