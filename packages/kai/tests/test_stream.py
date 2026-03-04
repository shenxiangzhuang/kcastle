"""Tests for kai.stream module using a mock provider."""

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from kai.errors import ErrorKind, KaiError
from kai.providers import ProviderBase
from kai.stream import complete, stream
from kai.types.message import Context, Message
from kai.types.stream import (
    Done,
    Error,
    StreamEvent,
    TextDelta,
    ThinkDelta,
    ThinkSignature,
    ToolCallBegin,
    ToolCallDelta,
    ToolCallEnd,
    Usage,
)
from kai.types.usage import TokenUsage


class MockProvider(ProviderBase):
    """A mock provider that yields pre-configured events."""

    def __init__(
        self,
        events: Sequence[StreamEvent],
        *,
        provider: str = "mock",
        model: str = "mock-1",
    ) -> None:
        self._events = events
        self._provider = provider
        self._model = model

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    async def stream(self, context: Context, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        for event in self._events:
            yield event


class ErrorProvider(ProviderBase):
    """A mock provider that raises an error."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    @property
    def provider(self) -> str:
        return "error"

    @property
    def model(self) -> str:
        return "error-1"

    async def stream(self, context: Context, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        raise self._error
        yield  # Make it a generator  # type: ignore[misc]


def _context() -> Context:
    return Context(messages=[Message(role="user", content="hello")])


async def _collect(events: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in events]


class TestStreamTextOnly:
    async def test_simple_text_stream(self) -> None:
        provider = MockProvider(
            [
                TextDelta(delta="Hello"),
                TextDelta(delta=" world"),
                Usage(usage=TokenUsage(input_tokens=10, output_tokens=5)),
            ]
        )
        events = await _collect(stream(provider, _context()))

        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_deltas) == 2
        assert text_deltas[0].delta == "Hello"
        assert text_deltas[1].delta == " world"

        done = [e for e in events if isinstance(e, Done)]
        assert len(done) == 1
        assert done[0].message.extract_text() == "Hello world"
        assert done[0].message.usage is not None
        assert done[0].message.usage.input_tokens == 10
        assert done[0].message.stop_reason == "stop"

    async def test_single_text_delta(self) -> None:
        provider = MockProvider([TextDelta(delta="Hi")])
        events = await _collect(stream(provider, _context()))

        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_deltas) == 1
        assert text_deltas[0].delta == "Hi"

        done = next(e for e in events if isinstance(e, Done))
        assert done.message.extract_text() == "Hi"


class TestStreamThinking:
    async def test_think_then_text(self) -> None:
        provider = MockProvider(
            [
                ThinkDelta(delta="Let me think"),
                ThinkDelta(delta="..."),
                ThinkSignature(signature="sig123"),
                TextDelta(delta="Answer"),
            ]
        )
        events = await _collect(stream(provider, _context()))

        think_deltas = [e for e in events if isinstance(e, ThinkDelta)]
        assert len(think_deltas) == 2

        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_deltas) == 1

        done = next(e for e in events if isinstance(e, Done))
        assert done.message.extract_text() == "Answer"
        # The message should have both think and text parts
        assert done.message.content is not None
        assert len(done.message.content) == 2


class TestStreamToolCalls:
    async def test_tool_call_stream(self) -> None:
        provider = MockProvider(
            [
                ToolCallBegin(id="call_1", name="search"),
                ToolCallDelta(arguments='{"q":'),
                ToolCallDelta(arguments=' "test"}'),
                ToolCallEnd(),
                Usage(usage=TokenUsage(input_tokens=20, output_tokens=10)),
            ]
        )
        events = await _collect(stream(provider, _context()))

        tc_begins = [e for e in events if isinstance(e, ToolCallBegin)]
        assert len(tc_begins) == 1
        assert tc_begins[0].name == "search"

        done = next(e for e in events if isinstance(e, Done))
        assert done.message.tool_calls is not None
        assert len(done.message.tool_calls) == 1
        assert done.message.tool_calls[0].arguments == '{"q": "test"}'
        assert done.message.stop_reason == "tool_use"

    async def test_text_then_tool_call(self) -> None:
        provider = MockProvider(
            [
                TextDelta(delta="I'll search for that."),
                ToolCallBegin(id="call_1", name="search"),
                ToolCallDelta(arguments='{"q": "test"}'),
                ToolCallEnd(),
            ]
        )
        events = await _collect(stream(provider, _context()))

        done = next(e for e in events if isinstance(e, Done))
        assert done.message.extract_text() == "I'll search for that."
        assert done.message.tool_calls is not None

    async def test_parallel_tool_calls(self) -> None:
        """Multiple tool calls should all appear in the final message."""
        provider = MockProvider(
            [
                ToolCallBegin(id="call_1", name="get_system_info"),
                ToolCallDelta(arguments="{}"),
                ToolCallEnd(),
                ToolCallBegin(id="call_2", name="get_python_version"),
                ToolCallDelta(arguments="{}"),
                ToolCallEnd(),
                ToolCallBegin(id="call_3", name="get_env_variable"),
                ToolCallDelta(arguments='{"name":'),
                ToolCallDelta(arguments=' "HOME"}'),
                ToolCallEnd(),
            ]
        )
        events = await _collect(stream(provider, _context()))

        tc_begins = [e for e in events if isinstance(e, ToolCallBegin)]
        assert len(tc_begins) == 3
        assert tc_begins[0].name == "get_system_info"
        assert tc_begins[1].name == "get_python_version"
        assert tc_begins[2].name == "get_env_variable"

        done = next(e for e in events if isinstance(e, Done))
        assert done.message.tool_calls is not None
        assert len(done.message.tool_calls) == 3
        assert done.message.tool_calls[0].id == "call_1"
        assert done.message.tool_calls[0].arguments == "{}"
        assert done.message.tool_calls[1].id == "call_2"
        assert done.message.tool_calls[2].id == "call_3"
        assert done.message.tool_calls[2].arguments == '{"name": "HOME"}'
        assert done.message.stop_reason == "tool_use"

    async def test_parallel_tool_calls_ordering(self) -> None:
        """ToolCallBegin/End events are forwarded in order."""
        provider = MockProvider(
            [
                ToolCallBegin(id="call_1", name="tool_a"),
                ToolCallEnd(),
                ToolCallBegin(id="call_2", name="tool_b"),
                ToolCallEnd(),
            ]
        )
        events = await _collect(stream(provider, _context()))

        tc_events = [e for e in events if isinstance(e, ToolCallBegin | ToolCallEnd)]
        assert len(tc_events) == 4
        assert isinstance(tc_events[0], ToolCallBegin)
        assert isinstance(tc_events[1], ToolCallEnd)
        assert isinstance(tc_events[2], ToolCallBegin)
        assert isinstance(tc_events[3], ToolCallEnd)

    async def test_text_then_parallel_tool_calls(self) -> None:
        """Text deltas appear before tool call events."""
        provider = MockProvider(
            [
                TextDelta(delta="Let me check."),
                ToolCallBegin(id="call_1", name="tool_a"),
                ToolCallDelta(arguments="{}"),
                ToolCallEnd(),
                ToolCallBegin(id="call_2", name="tool_b"),
                ToolCallDelta(arguments="{}"),
                ToolCallEnd(),
            ]
        )
        events = await _collect(stream(provider, _context()))

        text_idx = next(i for i, e in enumerate(events) if isinstance(e, TextDelta))
        first_tc_idx = next(i for i, e in enumerate(events) if isinstance(e, ToolCallBegin))
        assert text_idx < first_tc_idx

        done = next(e for e in events if isinstance(e, Done))
        assert done.message.extract_text() == "Let me check."
        assert done.message.tool_calls is not None
        assert len(done.message.tool_calls) == 2


class TestStreamErrors:
    async def test_provider_error_yields_error_event(self) -> None:
        provider = ErrorProvider(KaiError(ErrorKind.PROVIDER, "API down"))
        events = await _collect(stream(provider, _context()))

        error_events = [e for e in events if isinstance(e, Error)]
        assert len(error_events) == 1
        assert isinstance(error_events[0].error, KaiError)

    async def test_empty_response_yields_error(self) -> None:
        provider = MockProvider([])  # No events at all
        events = await _collect(stream(provider, _context()))

        error_events = [e for e in events if isinstance(e, Error)]
        assert len(error_events) == 1
        assert isinstance(error_events[0].error, KaiError)
        assert error_events[0].error.kind == ErrorKind.EMPTY_RESPONSE


class TestComplete:
    async def test_complete_returns_message(self) -> None:
        provider = MockProvider(
            [
                TextDelta(delta="Hello!"),
                Usage(usage=TokenUsage(input_tokens=5, output_tokens=3)),
            ]
        )
        msg = await complete(provider, _context())

        assert isinstance(msg, Message)
        assert msg.extract_text() == "Hello!"
        assert msg.role == "assistant"
        assert msg.usage is not None

    async def test_complete_raises_on_error(self) -> None:
        provider = ErrorProvider(KaiError(ErrorKind.PROVIDER, "API error"))
        with pytest.raises(KaiError, match="API error"):
            await complete(provider, _context())

    async def test_complete_raises_on_empty(self) -> None:
        provider = MockProvider([])
        with pytest.raises(KaiError) as exc_info:
            await complete(provider, _context())
        assert exc_info.value.kind == ErrorKind.EMPTY_RESPONSE
