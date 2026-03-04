"""Tests for Anthropic provider — message serialization and stream conversion.

Tests exercise the public ``stream()`` method with a mocked Anthropic client.
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

from anthropic.types import (
    MessageDeltaEvent,
    MessageStartEvent,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
)

from kai.providers.anthropic import AnthropicMessages
from kai.types.message import Context, ImagePart, Message, TextPart, ThinkPart, ToolCall
from kai.types.stream import (
    StreamEvent,
    TextDelta,
    ToolCallBegin,
    ToolCallDelta,
    ToolCallEnd,
    Usage,
)


def _ctx(*messages: Message, system: str | None = None) -> Context:
    return Context(system=system, messages=list(messages))


class _MockStream:
    """Minimal mock of ``anthropic.AsyncStream`` — async with + async for."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def __aenter__(self) -> "_MockStream":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Any]:
        for e in self._events:
            yield e


async def _stream_raw(
    events: list[Any] | None = None,
    context: Context | None = None,
) -> tuple[list[StreamEvent], dict[str, Any]]:
    """Call ``AnthropicMessages.stream()`` with a mocked client.

    Returns ``(output_chunks, captured_create_kwargs)``.
    """
    if context is None:
        context = _ctx(Message(role="user", content="hello"))

    captured: dict[str, Any] = {}

    async def _fake_create(**kwargs: Any) -> _MockStream:
        captured.update(kwargs)
        return _MockStream(events or [])

    with patch("kai.providers.anthropic.AsyncAnthropic") as mock_cls:
        mock_cls.return_value.messages.create = _fake_create
        provider = AnthropicMessages(model="test-model", api_key="test-key")

    output = [c async for c in provider.stream(context)]
    return output, captured


def _msg_start(
    *,
    input_tokens: int = 10,
    output_tokens: int = 0,
) -> MessageStartEvent:
    """Build a ``message_start`` event."""
    return MessageStartEvent.model_validate(
        {
            "type": "message_start",
            "message": {
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "test",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            },
        }
    )


def _block_start(
    block: dict[str, Any],
    *,
    index: int = 0,
) -> RawContentBlockStartEvent:
    """Build a ``content_block_start`` event."""
    return RawContentBlockStartEvent.model_validate(
        {
            "type": "content_block_start",
            "index": index,
            "content_block": block,
        }
    )


def _block_delta(
    delta: dict[str, Any],
    *,
    index: int = 0,
) -> RawContentBlockDeltaEvent:
    """Build a ``content_block_delta`` event."""
    return RawContentBlockDeltaEvent.model_validate(
        {
            "type": "content_block_delta",
            "index": index,
            "delta": delta,
        }
    )


def _msg_delta(*, output_tokens: int = 0) -> MessageDeltaEvent:
    """Build a ``message_delta`` event."""
    return MessageDeltaEvent.model_validate(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }
    )


class TestMessageSerialization:
    """Context → kwargs sent to the Anthropic Messages API."""

    async def test_system_prompt(self) -> None:
        """System prompt is passed via the separate ``system`` kwarg."""
        ctx = _ctx(
            Message(role="user", content="Hi"),
            system="Be helpful.",
        )
        _, kw = await _stream_raw(context=ctx)
        assert kw["system"] == "Be helpful."

    async def test_no_system(self) -> None:
        ctx = _ctx(Message(role="user", content="Hi"))
        _, kw = await _stream_raw(context=ctx)
        assert kw["system"] == []

    async def test_user_text(self) -> None:
        ctx = _ctx(Message(role="user", content="Hello"))
        _, kw = await _stream_raw(context=ctx)
        msgs = kw["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        blocks = msgs[0]["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Hello"

    async def test_user_image(self) -> None:
        ctx = _ctx(
            Message(
                role="user",
                content=[
                    TextPart(text="Look"),
                    ImagePart(data="abc", mime_type="image/png"),
                ],
            )
        )
        _, kw = await _stream_raw(context=ctx)
        blocks = kw["messages"][0]["content"]
        assert len(blocks) == 2
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "image"
        assert blocks[1]["source"]["type"] == "base64"

    async def test_tool_result(self) -> None:
        ctx = _ctx(
            Message(role="user", content="Hi"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(id="tc1", name="fn", arguments="{}"),
                ],
            ),
            Message.tool_result("tc1", "result"),
        )
        _, kw = await _stream_raw(context=ctx)
        tool_msg = kw["messages"][2]
        assert tool_msg["role"] == "user"
        block = tool_msg["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "tc1"
        assert block["content"] == "result"

    async def test_multiple_tool_results_merged_into_one_message(self) -> None:
        """Regression: consecutive tool-result messages must be merged into a
        single user message with multiple tool_result blocks.

        The Anthropic API rejects separate user messages for each tool result
        with: "tool_use ids were found without tool_result blocks immediately
        after".  Before the fix, each Message.tool_result() produced its own
        MessageParam, triggering that 400 error.
        """
        ctx = _ctx(
            Message(role="user", content="What's the weather in Tokyo and Paris?"),
            Message(
                role="assistant",
                content="I'll check both.",
                tool_calls=[
                    ToolCall(id="tc1", name="get_weather", arguments='{"city":"Tokyo"}'),
                    ToolCall(id="tc2", name="get_weather", arguments='{"city":"Paris"}'),
                ],
            ),
            Message.tool_result("tc1", "Sunny, 22°C in Tokyo"),
            Message.tool_result("tc2", "Cloudy, 15°C in Paris"),
        )
        _, kw = await _stream_raw(context=ctx)
        msgs = kw["messages"]

        # Must be exactly 3 messages: user, assistant, one merged tool-result message.
        # Before the fix this was 4 (two separate user messages for the tool results).
        assert len(msgs) == 3, (
            f"Expected 3 messages (user + assistant + merged tool results), got {len(msgs)}. "
            "Multiple tool results must be merged into a single user message."
        )

        merged = msgs[2]
        assert merged["role"] == "user"
        blocks = merged["content"]

        # Both tool results must live in the same message.
        assert len(blocks) == 2
        assert all(b["type"] == "tool_result" for b in blocks)

        by_id = {b["tool_use_id"]: b for b in blocks}
        assert by_id["tc1"]["content"] == "Sunny, 22°C in Tokyo"
        assert by_id["tc2"]["content"] == "Cloudy, 15°C in Paris"

    async def test_tool_results_between_turns_not_merged(self) -> None:
        """Tool results from different assistant turns must NOT be merged together.

        Each assistant message's tool results belong to a separate user message.
        """
        ctx = _ctx(
            Message(role="user", content="Hi"),
            # Turn 1: one tool call + result
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="tc1", name="fn", arguments="{}")],
            ),
            Message.tool_result("tc1", "result-1"),
            # Turn 2: another tool call + result
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="tc2", name="fn", arguments="{}")],
            ),
            Message.tool_result("tc2", "result-2"),
        )
        _, kw = await _stream_raw(context=ctx)
        msgs = kw["messages"]

        # user, assistant-1, tool-results-1, assistant-2, tool-results-2
        assert len(msgs) == 5

        tool_msg_1 = msgs[2]
        tool_msg_2 = msgs[4]
        assert tool_msg_1["content"][0]["tool_use_id"] == "tc1"
        assert tool_msg_2["content"][0]["tool_use_id"] == "tc2"

    async def test_assistant_text(self) -> None:
        ctx = _ctx(
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
        )
        _, kw = await _stream_raw(context=ctx)
        msg = kw["messages"][1]
        assert msg["role"] == "assistant"
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Hello!"

    async def test_assistant_with_tool_calls(self) -> None:
        ctx = _ctx(
            Message(role="user", content="Hi"),
            Message(
                role="assistant",
                content="I'll check.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="search",
                        arguments='{"q":"test"}',
                    ),
                ],
            ),
        )
        _, kw = await _stream_raw(context=ctx)
        blocks = kw["messages"][1]["content"]
        text_blocks = [b for b in blocks if b["type"] == "text"]
        tool_blocks = [b for b in blocks if b["type"] == "tool_use"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "I'll check."
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["id"] == "tc1"
        assert tool_blocks[0]["name"] == "search"

    async def test_thinking_with_signature_kept(self) -> None:
        """Think parts with signatures are preserved."""
        ctx = _ctx(
            Message(role="user", content="Hi"),
            Message(
                role="assistant",
                content=[
                    ThinkPart(text="reasoning", signature="sig123"),
                    TextPart(text="answer"),
                ],
            ),
        )
        _, kw = await _stream_raw(context=ctx)
        blocks = kw["messages"][1]["content"]
        think = [b for b in blocks if b.get("type") == "thinking"]
        assert len(think) == 1
        assert think[0]["thinking"] == "reasoning"
        assert think[0]["signature"] == "sig123"

    async def test_thinking_without_signature_stripped(self) -> None:
        """Think parts without signatures are stripped."""
        ctx = _ctx(
            Message(role="user", content="Hi"),
            Message(
                role="assistant",
                content=[
                    ThinkPart(text="reasoning"),  # no signature
                    TextPart(text="answer"),
                ],
            ),
        )
        _, kw = await _stream_raw(context=ctx)
        blocks = kw["messages"][1]["content"]
        think = [b for b in blocks if b.get("type") == "thinking"]
        assert len(think) == 0


class TestStreamConversion:
    """Anthropic stream events → kai Chunks."""

    async def test_text_streaming(self) -> None:
        events = [
            _msg_start(input_tokens=10),
            _block_start({"type": "text", "text": ""}),
            _block_delta({"type": "text_delta", "text": "Hello"}),
            _block_delta({"type": "text_delta", "text": " world"}),
            _msg_delta(output_tokens=5),
        ]
        chunks, _ = await _stream_raw(events)
        texts = [c for c in chunks if isinstance(c, TextDelta)]
        assert len(texts) == 2
        assert texts[0].delta == "Hello"
        assert texts[1].delta == " world"
        usage = next(c for c in chunks if isinstance(c, Usage))
        assert usage.usage.input_tokens == 10
        assert usage.usage.output_tokens == 5

    async def test_tool_call_streaming(self) -> None:
        events = [
            _msg_start(),
            _block_start(
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "search",
                    "input": {},
                }
            ),
            _block_delta(
                {
                    "type": "input_json_delta",
                    "partial_json": '{"q":',
                }
            ),
            _block_delta(
                {
                    "type": "input_json_delta",
                    "partial_json": ' "test"}',
                }
            ),
        ]
        chunks, _ = await _stream_raw(events)
        starts = [c for c in chunks if isinstance(c, ToolCallBegin)]
        deltas = [c for c in chunks if isinstance(c, ToolCallDelta)]
        ends = [c for c in chunks if isinstance(c, ToolCallEnd)]
        assert len(starts) == 1
        assert starts[0].id == "toolu_01"
        assert starts[0].name == "search"
        assert len(deltas) == 2
        assert len(ends) == 1

    async def test_parallel_tool_calls(self) -> None:
        """Multiple tool_use blocks get proper Start/End chunks."""
        events = [
            _msg_start(),
            _block_start(
                {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
                index=0,
            ),
            _block_start(
                {"type": "tool_use", "id": "t2", "name": "b", "input": {}},
                index=1,
            ),
        ]
        chunks, _ = await _stream_raw(events)
        starts = [c for c in chunks if isinstance(c, ToolCallBegin)]
        ends = [c for c in chunks if isinstance(c, ToolCallEnd)]
        assert len(starts) == 2
        assert len(ends) == 2
        si = [i for i, c in enumerate(chunks) if isinstance(c, ToolCallBegin)]
        ei = [i for i, c in enumerate(chunks) if isinstance(c, ToolCallEnd)]
        assert ei[0] < si[1]

    async def test_usage_always_emitted(self) -> None:
        """Usage event is emitted even for an empty stream."""
        chunks, _ = await _stream_raw([])
        usage = [c for c in chunks if isinstance(c, Usage)]
        assert len(usage) == 1
