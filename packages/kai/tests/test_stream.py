"""Tests for kai.stream module using a mock provider."""

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from kai.errors import EmptyResponseError, ProviderError
from kai.providers import LLMBase
from kai.stream import complete, stream
from kai.types.message import Context, Message
from kai.types.stream import (
    Chunk,
    DoneEvent,
    ErrorEvent,
    StartEvent,
    StreamEvent,
    TextChunk,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkChunk,
    ThinkDeltaEvent,
    ThinkEndEvent,
    ThinkSignatureChunk,
    ThinkStartEvent,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallEndEvent,
    ToolCallStart,
    ToolCallStartEvent,
    UsageChunk,
)
from kai.types.usage import TokenUsage


class MockProvider(LLMBase):
    """A mock provider that yields pre-configured chunks."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        provider: str = "mock",
        model: str = "mock-1",
    ) -> None:
        self._chunks = chunks
        self._provider = provider
        self._model = model

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    async def stream_raw(self, context: Context, **kwargs: Any) -> AsyncIterator[Chunk]:
        for chunk in self._chunks:
            yield chunk


class ErrorProvider(LLMBase):
    """A mock provider that raises an error."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    @property
    def provider(self) -> str:
        return "error"

    @property
    def model(self) -> str:
        return "error-1"

    async def stream_raw(self, context: Context, **kwargs: Any) -> AsyncIterator[Chunk]:
        raise self._error
        yield  # Make it a generator  # type: ignore[misc]


def _context() -> Context:
    return Context(messages=[Message(role="user", content="hello")])


async def _collect(events: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in events]


class TestStreamTextOnly:
    async def test_simple_text_stream(self) -> None:
        chunks = [
            TextChunk(text="Hello"),
            TextChunk(text=" world"),
            UsageChunk(usage=TokenUsage(input_tokens=10, output_tokens=5)),
        ]
        provider = MockProvider(chunks)
        events = await _collect(stream(provider, _context()))

        # StartEvent, TextStartEvent, TextDeltaEvent x2, TextEndEvent, DoneEvent
        assert isinstance(events[0], StartEvent)

        text_starts = [e for e in events if isinstance(e, TextStartEvent)]
        assert len(text_starts) == 1

        text_deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert len(text_deltas) == 2
        assert text_deltas[0].delta == "Hello"
        assert text_deltas[1].delta == " world"

        text_ends = [e for e in events if isinstance(e, TextEndEvent)]
        assert len(text_ends) == 1
        assert text_ends[0].text == "Hello world"

        done = [e for e in events if isinstance(e, DoneEvent)]
        assert len(done) == 1
        assert done[0].message.extract_text() == "Hello world"
        assert done[0].message.usage is not None
        assert done[0].message.usage.input_tokens == 10
        assert done[0].message.stop_reason == "stop"

    async def test_partial_message_in_events(self) -> None:
        chunks = [TextChunk(text="Hi")]
        provider = MockProvider(chunks)
        events = await _collect(stream(provider, _context()))

        text_delta = next(e for e in events if isinstance(e, TextDeltaEvent))
        # partial should contain the text accumulated so far
        assert text_delta.partial.extract_text() == "Hi"


class TestStreamThinking:
    async def test_think_then_text(self) -> None:
        chunks = [
            ThinkChunk(text="Let me think"),
            ThinkChunk(text="..."),
            ThinkSignatureChunk(signature="sig123"),
            TextChunk(text="Answer"),
        ]
        provider = MockProvider(chunks)
        events = await _collect(stream(provider, _context()))

        think_starts = [e for e in events if isinstance(e, ThinkStartEvent)]
        assert len(think_starts) == 1

        think_deltas = [e for e in events if isinstance(e, ThinkDeltaEvent)]
        assert len(think_deltas) == 2

        think_ends = [e for e in events if isinstance(e, ThinkEndEvent)]
        assert len(think_ends) == 1
        assert think_ends[0].text == "Let me think..."

        text_starts = [e for e in events if isinstance(e, TextStartEvent)]
        assert len(text_starts) == 1

        done = next(e for e in events if isinstance(e, DoneEvent))
        assert done.message.extract_text() == "Answer"
        # The message should have both think and text parts
        assert len(done.message.content) == 2


class TestStreamToolCalls:
    async def test_tool_call_stream(self) -> None:
        chunks = [
            ToolCallStart(id="call_1", name="search"),
            ToolCallDelta(arguments='{"q":'),
            ToolCallDelta(arguments=' "test"}'),
            ToolCallEnd(),
            UsageChunk(usage=TokenUsage(input_tokens=20, output_tokens=10)),
        ]
        provider = MockProvider(chunks)
        events = await _collect(stream(provider, _context()))

        tc_starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
        assert len(tc_starts) == 1
        assert tc_starts[0].name == "search"

        tc_ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(tc_ends) == 1
        assert tc_ends[0].tool_call.arguments == '{"q": "test"}'

        done = next(e for e in events if isinstance(e, DoneEvent))
        assert done.message.tool_calls is not None
        assert len(done.message.tool_calls) == 1
        assert done.message.stop_reason == "tool_use"

    async def test_text_then_tool_call(self) -> None:
        chunks = [
            TextChunk(text="I'll search for that."),
            ToolCallStart(id="call_1", name="search"),
            ToolCallDelta(arguments='{"q": "test"}'),
            ToolCallEnd(),
        ]
        provider = MockProvider(chunks)
        events = await _collect(stream(provider, _context()))

        # Text should end before tool call starts
        text_end_idx = next(i for i, e in enumerate(events) if isinstance(e, TextEndEvent))
        tc_start_idx = next(i for i, e in enumerate(events) if isinstance(e, ToolCallStartEvent))
        assert text_end_idx < tc_start_idx

        done = next(e for e in events if isinstance(e, DoneEvent))
        assert done.message.extract_text() == "I'll search for that."
        assert done.message.tool_calls is not None

    async def test_parallel_tool_calls(self) -> None:
        """Multiple tool calls should each produce Start/End events and appear in the message."""
        chunks = [
            ToolCallStart(id="call_1", name="get_system_info"),
            ToolCallDelta(arguments="{}"),
            ToolCallEnd(),
            ToolCallStart(id="call_2", name="get_python_version"),
            ToolCallDelta(arguments="{}"),
            ToolCallEnd(),
            ToolCallStart(id="call_3", name="get_env_variable"),
            ToolCallDelta(arguments='{"name":'),
            ToolCallDelta(arguments=' "HOME"}'),
            ToolCallEnd(),
        ]
        provider = MockProvider(chunks)
        events = await _collect(stream(provider, _context()))

        tc_starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
        assert len(tc_starts) == 3
        assert tc_starts[0].name == "get_system_info"
        assert tc_starts[1].name == "get_python_version"
        assert tc_starts[2].name == "get_env_variable"

        tc_ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(tc_ends) == 3
        assert tc_ends[0].tool_call.id == "call_1"
        assert tc_ends[0].tool_call.arguments == "{}"
        assert tc_ends[1].tool_call.id == "call_2"
        assert tc_ends[2].tool_call.id == "call_3"
        assert tc_ends[2].tool_call.arguments == '{"name": "HOME"}'

        done = next(e for e in events if isinstance(e, DoneEvent))
        assert done.message.tool_calls is not None
        assert len(done.message.tool_calls) == 3
        assert done.message.stop_reason == "tool_use"

    async def test_parallel_tool_calls_ordering(self) -> None:
        """Each ToolCallEnd must appear before the next ToolCallStart."""
        chunks = [
            ToolCallStart(id="call_1", name="tool_a"),
            ToolCallEnd(),
            ToolCallStart(id="call_2", name="tool_b"),
            ToolCallEnd(),
        ]
        provider = MockProvider(chunks)
        events = await _collect(stream(provider, _context()))

        tc_events = [e for e in events if isinstance(e, ToolCallStartEvent | ToolCallEndEvent)]
        assert len(tc_events) == 4
        assert isinstance(tc_events[0], ToolCallStartEvent)
        assert isinstance(tc_events[1], ToolCallEndEvent)
        assert isinstance(tc_events[2], ToolCallStartEvent)
        assert isinstance(tc_events[3], ToolCallEndEvent)

    async def test_text_then_parallel_tool_calls(self) -> None:
        """Text block should be flushed before parallel tool calls start."""
        chunks = [
            TextChunk(text="Let me check."),
            ToolCallStart(id="call_1", name="tool_a"),
            ToolCallDelta(arguments="{}"),
            ToolCallEnd(),
            ToolCallStart(id="call_2", name="tool_b"),
            ToolCallDelta(arguments="{}"),
            ToolCallEnd(),
        ]
        provider = MockProvider(chunks)
        events = await _collect(stream(provider, _context()))

        text_end_idx = next(i for i, e in enumerate(events) if isinstance(e, TextEndEvent))
        first_tc_start_idx = next(
            i for i, e in enumerate(events) if isinstance(e, ToolCallStartEvent)
        )
        assert text_end_idx < first_tc_start_idx

        done = next(e for e in events if isinstance(e, DoneEvent))
        assert done.message.extract_text() == "Let me check."
        assert done.message.tool_calls is not None
        assert len(done.message.tool_calls) == 2


class TestStreamErrors:
    async def test_provider_error_yields_error_event(self) -> None:
        provider = ErrorProvider(ProviderError("API down"))
        events = await _collect(stream(provider, _context()))

        assert isinstance(events[0], StartEvent)
        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert isinstance(error_events[0].error, ProviderError)

    async def test_empty_response_yields_error(self) -> None:
        provider = MockProvider([])  # No chunks at all
        events = await _collect(stream(provider, _context()))

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert isinstance(error_events[0].error, EmptyResponseError)


class TestComplete:
    async def test_complete_returns_message(self) -> None:
        chunks = [
            TextChunk(text="Hello!"),
            UsageChunk(usage=TokenUsage(input_tokens=5, output_tokens=3)),
        ]
        provider = MockProvider(chunks)
        msg = await complete(provider, _context())

        assert isinstance(msg, Message)
        assert msg.extract_text() == "Hello!"
        assert msg.role == "assistant"
        assert msg.usage is not None

    async def test_complete_raises_on_error(self) -> None:
        provider = ErrorProvider(ProviderError("API error"))
        with pytest.raises(ProviderError, match="API error"):
            await complete(provider, _context())

    async def test_complete_raises_on_empty(self) -> None:
        provider = MockProvider([])
        with pytest.raises(EmptyResponseError):
            await complete(provider, _context())
