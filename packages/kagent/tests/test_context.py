"""Tests for kagent.context — ContextBuilder implementations."""

from __future__ import annotations

import pytest
from conftest import MockProvider, text_chunks
from kai import Message
from kai.types.message import ToolCall

from kagent.context import (
    AdaptiveBuilder,
    CompactingBuilder,
    ContextBuilder,
    ContextSwitchTool,
    DefaultBuilder,
    SlidingWindowBuilder,
)
from kagent.state import AgentState
from kagent.trace import Trace, TraceEntry


def _state_with_msgs(
    *messages: Message,
    system: str | None = None,
) -> AgentState:
    """Create an AgentState with the given messages in the trace."""
    trace = Trace()
    for msg in messages:
        if msg.role == "user":
            trace.append(TraceEntry.user(msg))
        elif msg.role == "assistant":
            trace.append(TraceEntry.assistant(msg))
        elif msg.role == "tool":
            trace.append(TraceEntry.tool_result(msg))
    return AgentState(system=system, trace=trace)


# ---------------------------------------------------------------------------
# DefaultBuilder
# ---------------------------------------------------------------------------


class TestDefaultBuilder:
    @pytest.mark.asyncio
    async def test_pass_through(self) -> None:
        state = _state_with_msgs(
            Message(role="user", content="Hi"),
            system="Hello",
        )
        builder = DefaultBuilder()
        ctx = await builder.build(state)

        assert ctx.system == "Hello"
        assert len(ctx.messages) == 1
        assert ctx.messages[0].extract_text() == "Hi"
        assert list(ctx.tools) == []

    @pytest.mark.asyncio
    async def test_satisfies_protocol(self) -> None:
        assert isinstance(DefaultBuilder(), ContextBuilder)


# ---------------------------------------------------------------------------
# SlidingWindowBuilder
# ---------------------------------------------------------------------------


class TestSlidingWindowBuilder:
    @pytest.mark.asyncio
    async def test_no_truncation_when_within_window(self) -> None:
        state = _state_with_msgs(
            Message(role="user", content="a"),
            Message(role="assistant", content="b"),
            system="sys",
        )
        builder = SlidingWindowBuilder(window_size=10)
        ctx = await builder.build(state)

        assert len(ctx.messages) == 2

    @pytest.mark.asyncio
    async def test_truncation_keeps_first_and_tail(self) -> None:
        msgs = [Message(role="user", content="goal")]  # first message
        for i in range(10):
            msgs.append(Message(role="user", content=f"msg-{i}"))
            msgs.append(Message(role="assistant", content=f"reply-{i}"))

        state = _state_with_msgs(*msgs)
        builder = SlidingWindowBuilder(window_size=4)
        ctx = await builder.build(state)

        # first message + last 4 messages
        assert len(ctx.messages) == 5
        assert ctx.messages[0].extract_text() == "goal"
        assert ctx.messages[-1].extract_text() == "reply-9"

    @pytest.mark.asyncio
    async def test_first_message_not_duplicated(self) -> None:
        """When the first message is already in the tail, don't prepend it."""
        state = _state_with_msgs(
            Message(role="user", content="a"),
            Message(role="assistant", content="b"),
            Message(role="user", content="c"),
        )
        builder = SlidingWindowBuilder(window_size=5)
        ctx = await builder.build(state)

        assert len(ctx.messages) == 3

    @pytest.mark.asyncio
    async def test_drops_orphaned_tool_results(self) -> None:
        """Tool results without matching tool_calls in window are dropped."""
        msgs = [
            Message(role="user", content="goal"),
            # This assistant + tool pair will be outside the window
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="tc-1", name="foo", arguments="{}")],
            ),
            Message(role="tool", content="result-1", tool_call_id="tc-1"),
            # These will be in the window
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="tc-2", name="bar", arguments="{}")],
            ),
            Message(role="tool", content="result-2", tool_call_id="tc-2"),
            Message(role="assistant", content="done"),
        ]
        state = _state_with_msgs(*msgs)
        builder = SlidingWindowBuilder(window_size=3)
        ctx = await builder.build(state)

        # first msg + last 3 (tc-2 assistant, tc-2 tool, "done")
        # tc-1 tool result is not in the tail and its call is not in window
        tool_msgs = [m for m in ctx.messages if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "tc-2"

    def test_invalid_window_size(self) -> None:
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            SlidingWindowBuilder(window_size=0)

    @pytest.mark.asyncio
    async def test_satisfies_protocol(self) -> None:
        assert isinstance(SlidingWindowBuilder(), ContextBuilder)


# ---------------------------------------------------------------------------
# CompactingBuilder
# ---------------------------------------------------------------------------


class TestCompactingBuilder:
    @pytest.mark.asyncio
    async def test_no_compaction_below_threshold(self) -> None:
        provider = MockProvider([])
        builder = CompactingBuilder(provider, threshold=10, max_preserved=4)
        state = _state_with_msgs(
            *(Message(role="user", content=f"msg-{i}") for i in range(5)),
            system="sys",
        )
        ctx = await builder.build(state)

        # Below threshold — all messages passed through
        assert len(ctx.messages) == 5

    @pytest.mark.asyncio
    async def test_compaction_above_threshold(self) -> None:
        # The summarization LLM call returns this
        summary_provider = MockProvider([text_chunks("Summary of old messages.")])
        builder = CompactingBuilder(summary_provider, threshold=5, max_preserved=2)

        state = _state_with_msgs(
            *(Message(role="user", content=f"msg-{i}") for i in range(8)),
            system="sys",
        )
        ctx = await builder.build(state)

        # summary message + last 2 preserved
        assert len(ctx.messages) == 3
        assert "[Conversation summary]" in ctx.messages[0].extract_text()
        assert "Summary of old messages." in ctx.messages[0].extract_text()
        assert ctx.messages[1].extract_text() == "msg-6"
        assert ctx.messages[2].extract_text() == "msg-7"

    @pytest.mark.asyncio
    async def test_cache_prevents_re_summarization(self) -> None:
        summary_provider = MockProvider(
            [
                text_chunks("Summary v1."),
                # Second call would fail if reached — only one turn configured
            ]
        )
        builder = CompactingBuilder(summary_provider, threshold=3, max_preserved=2)

        state = _state_with_msgs(
            *(Message(role="user", content=f"msg-{i}") for i in range(5)),
        )

        # First call — triggers summarization
        ctx1 = await builder.build(state)
        assert "Summary v1." in ctx1.messages[0].extract_text()

        # Second call with same message count — should use cache (no LLM call)
        ctx2 = await builder.build(state)
        assert "Summary v1." in ctx2.messages[0].extract_text()

    def test_invalid_max_preserved(self) -> None:
        provider = MockProvider([])
        with pytest.raises(ValueError, match="max_preserved must be >= 1"):
            CompactingBuilder(provider, max_preserved=0)

    def test_threshold_must_exceed_max_preserved(self) -> None:
        provider = MockProvider([])
        with pytest.raises(ValueError, match="threshold must be > max_preserved"):
            CompactingBuilder(provider, max_preserved=5, threshold=5)

    @pytest.mark.asyncio
    async def test_satisfies_protocol(self) -> None:
        provider = MockProvider([])
        assert isinstance(CompactingBuilder(provider), ContextBuilder)


# ---------------------------------------------------------------------------
# AdaptiveBuilder
# ---------------------------------------------------------------------------


class TestAdaptiveBuilder:
    @pytest.mark.asyncio
    async def test_delegates_to_current(self) -> None:
        adaptive = AdaptiveBuilder(
            builders={
                "full": DefaultBuilder(),
                "window": SlidingWindowBuilder(window_size=2),
            },
            default="full",
        )
        state = _state_with_msgs(
            *(Message(role="user", content=f"msg-{i}") for i in range(10)),
        )

        # Default is "full" — all messages
        ctx = await adaptive.build(state)
        assert len(ctx.messages) == 10

        # Switch to window
        adaptive.switch("window")
        ctx = await adaptive.build(state)
        # first message + last 2
        assert len(ctx.messages) == 3

    @pytest.mark.asyncio
    async def test_switch_unknown_raises(self) -> None:
        adaptive = AdaptiveBuilder(
            builders={"full": DefaultBuilder()},
            default="full",
        )
        with pytest.raises(KeyError, match="Unknown builder"):
            adaptive.switch("nonexistent")

    def test_empty_builders_raises(self) -> None:
        with pytest.raises(ValueError, match="builders must not be empty"):
            AdaptiveBuilder(builders={}, default="x")

    def test_invalid_default_raises(self) -> None:
        with pytest.raises(ValueError, match="not found in builders"):
            AdaptiveBuilder(builders={"a": DefaultBuilder()}, default="b")

    def test_register(self) -> None:
        adaptive = AdaptiveBuilder(
            builders={"full": DefaultBuilder()},
            default="full",
        )
        adaptive.register("window", SlidingWindowBuilder(window_size=5))
        assert "window" in adaptive.available

    def test_properties(self) -> None:
        adaptive = AdaptiveBuilder(
            builders={"a": DefaultBuilder(), "b": DefaultBuilder()},
            default="a",
        )
        assert adaptive.current == "a"
        assert sorted(adaptive.available) == ["a", "b"]

    @pytest.mark.asyncio
    async def test_satisfies_protocol(self) -> None:
        adaptive = AdaptiveBuilder(
            builders={"full": DefaultBuilder()},
            default="full",
        )
        assert isinstance(adaptive, ContextBuilder)


# ---------------------------------------------------------------------------
# ContextSwitchTool
# ---------------------------------------------------------------------------


class TestContextSwitchTool:
    @pytest.mark.asyncio
    async def test_switch_success(self) -> None:
        adaptive = AdaptiveBuilder(
            builders={"full": DefaultBuilder(), "window": SlidingWindowBuilder()},
            default="full",
        )
        tool = adaptive.create_tool()

        result = await tool.execute(ContextSwitchTool.Params(strategy="window"))
        assert not result.is_error
        assert adaptive.current == "window"
        assert "window" in result.output

    @pytest.mark.asyncio
    async def test_switch_unknown_strategy(self) -> None:
        adaptive = AdaptiveBuilder(
            builders={"full": DefaultBuilder()},
            default="full",
        )
        tool = adaptive.create_tool()

        result = await tool.execute(ContextSwitchTool.Params(strategy="nope"))
        assert result.is_error
        assert adaptive.current == "full"  # unchanged

    def test_tool_description_lists_strategies(self) -> None:
        adaptive = AdaptiveBuilder(
            builders={"full": DefaultBuilder(), "compact": DefaultBuilder()},
            default="full",
        )
        tool = adaptive.create_tool()
        assert "full" in tool.description
        assert "compact" in tool.description

    def test_tool_has_parameters(self) -> None:
        adaptive = AdaptiveBuilder(
            builders={"full": DefaultBuilder()},
            default="full",
        )
        tool = adaptive.create_tool()
        assert tool.name == "switch_context_strategy"
        assert "strategy" in str(tool.parameters)
