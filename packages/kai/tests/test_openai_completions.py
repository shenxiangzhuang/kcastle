"""Tests for OpenAI Completions provider — message serialization and stream conversion.

Tests exercise the public ``stream()`` method with a mocked OpenAI client,
verifying both the message format sent to the API and the chunks produced from
the streamed response.
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

from openai.types.chat import ChatCompletionChunk

from kai.providers.openai import OpenAIChatCompletions
from kai.providers.openai import (
    _extract_reasoning_text as _extract_reasoning_text,
)
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
    """Build a Context from messages."""
    return Context(system=system, messages=list(messages))


def _chunk(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, int] | None = None,
) -> ChatCompletionChunk:
    """Build a ``ChatCompletionChunk`` from minimal test parameters.

    ``tool_calls`` entries use a flat dict with keys ``index``, ``id``,
    ``name``, ``arguments``; the helper restructures them into the nested
    ``function`` sub-dict expected by the API schema.
    """
    choices: list[dict[str, Any]] = []
    if content is not None or tool_calls is not None:
        delta: dict[str, Any] = {}
        if content is not None:
            delta["content"] = content
        if tool_calls is not None:
            entries: list[dict[str, Any]] = []
            for tc in tool_calls:
                entry: dict[str, Any] = {"index": tc.get("index", 0)}
                if "id" in tc:
                    entry["id"] = tc["id"]
                    entry["type"] = "function"
                func: dict[str, Any] = {}
                if "name" in tc:
                    func["name"] = tc["name"]
                if "arguments" in tc:
                    func["arguments"] = tc["arguments"]
                if func:
                    entry["function"] = func
                entries.append(entry)
            delta["tool_calls"] = entries
        choices.append({"index": 0, "delta": delta, "finish_reason": None})

    data: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "test",
        "choices": choices,
    }
    if usage is not None:
        data["usage"] = usage
    return ChatCompletionChunk.model_validate(data)


async def _stream_raw(
    api_chunks: list[ChatCompletionChunk],
    context: Context | None = None,
) -> tuple[list[StreamEvent], dict[str, Any]]:
    """Call ``OpenAIChatCompletions.stream()`` with a mocked client.

    Returns ``(output_events, captured_create_kwargs)``.
    """
    if context is None:
        context = _ctx(Message(role="user", content="hello"))

    captured: dict[str, Any] = {}

    async def _fake_create(
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        captured.update(kwargs)

        async def _gen() -> AsyncIterator[ChatCompletionChunk]:
            for c in api_chunks:
                yield c

        return _gen()

    with patch("kai.providers.openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create = _fake_create
        provider = OpenAIChatCompletions(model="test-model", api_key="test-key")

    output = [c async for c in provider.stream(context)]
    return output, captured


class TestMessageSerialization:
    """Context → ``messages`` kwarg sent to the Chat Completions API."""

    async def test_system_only(self) -> None:
        ctx = _ctx(system="Be helpful.")
        _, kw = await _stream_raw([], ctx)
        msgs = kw["messages"]
        assert msgs == [{"role": "system", "content": "Be helpful."}]

    async def test_no_system(self) -> None:
        ctx = _ctx(Message(role="user", content="Hello"))
        _, kw = await _stream_raw([], ctx)
        msgs = kw["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    async def test_system_plus_messages(self) -> None:
        ctx = _ctx(
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
            system="Be helpful.",
        )
        _, kw = await _stream_raw([], ctx)
        msgs = kw["messages"]
        assert len(msgs) == 3
        assert msgs[0] == {"role": "system", "content": "Be helpful."}
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    async def test_user_text(self) -> None:
        ctx = _ctx(Message(role="user", content="Hello"))
        _, kw = await _stream_raw([], ctx)
        msg = kw["messages"][0]
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"

    async def test_user_multimodal(self) -> None:
        ctx = _ctx(
            Message(
                role="user",
                content=[
                    TextPart(text="Look at this"),
                    ImagePart(data="abc123", mime_type="image/png"),
                ],
            )
        )
        _, kw = await _stream_raw([], ctx)
        content: list[Any] = kw["messages"][0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Look at this"}
        assert content[1]["type"] == "image_url"

    async def test_assistant_text(self) -> None:
        ctx = _ctx(
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
        )
        _, kw = await _stream_raw([], ctx)
        msg = kw["messages"][1]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello!"

    async def test_assistant_with_tool_calls(self) -> None:
        ctx = _ctx(
            Message(role="user", content="Hi"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="get_weather",
                        arguments='{"city":"Tokyo"}',
                    ),
                ],
            ),
        )
        _, kw = await _stream_raw([], ctx)
        msg = kw["messages"][1]
        assert msg["role"] == "assistant"
        tcs = msg.get("tool_calls")
        assert len(tcs) == 1
        assert tcs[0]["id"] == "tc1"
        assert tcs[0]["function"]["name"] == "get_weather"

    async def test_assistant_think_parts_stripped(self) -> None:
        """Think parts are stripped for the Completions API."""
        ctx = _ctx(
            Message(role="user", content="Hi"),
            Message(
                role="assistant",
                content=[
                    ThinkPart(text="reasoning..."),
                    TextPart(text="answer"),
                ],
            ),
        )
        _, kw = await _stream_raw([], ctx)
        assert kw["messages"][1]["content"] == "answer"

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
            Message.tool_result("tc1", "sunny 22°C"),
        )
        _, kw = await _stream_raw([], ctx)
        msg = kw["messages"][2]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "tc1"
        assert msg["content"] == "sunny 22°C"

    async def test_single_text_content_is_string(self) -> None:
        """A single TextPart should serialize as a plain string."""
        ctx = _ctx(Message(role="user", content=[TextPart(text="hi")]))
        _, kw = await _stream_raw([], ctx)
        assert kw["messages"][0]["content"] == "hi"

    async def test_empty_content_is_empty_string(self) -> None:
        ctx = _ctx(Message(role="user", content=[]))
        _, kw = await _stream_raw([], ctx)
        assert kw["messages"][0]["content"] == ""


class TestStreamConversion:
    """ChatCompletionChunk stream → kai StreamEvent sequence."""

    async def test_text_chunks(self) -> None:
        chunks, _ = await _stream_raw(
            [
                _chunk(content="Hello"),
                _chunk(content=" world"),
            ]
        )
        texts = [c for c in chunks if isinstance(c, TextDelta)]
        assert len(texts) == 2
        assert texts[0].delta == "Hello"
        assert texts[1].delta == " world"

    async def test_single_tool_call(self) -> None:
        chunks, _ = await _stream_raw(
            [
                _chunk(
                    tool_calls=[
                        {"index": 0, "id": "call_1", "name": "search"},
                    ]
                ),
                _chunk(tool_calls=[{"index": 0, "arguments": '{"q": "test"}'}]),
            ]
        )
        assert isinstance(chunks[0], ToolCallBegin)
        assert chunks[0].id == "call_1"
        assert chunks[0].name == "search"
        assert isinstance(chunks[1], ToolCallDelta)
        assert chunks[1].arguments == '{"q": "test"}'
        assert isinstance(chunks[2], ToolCallEnd)

    async def test_parallel_tool_calls_end_between_starts(self) -> None:
        """ToolCallEnd must appear between consecutive ToolCallStart chunks."""
        chunks, _ = await _stream_raw(
            [
                _chunk(
                    tool_calls=[
                        {"index": 0, "id": "call_1", "name": "get_system_info"},
                    ]
                ),
                _chunk(
                    tool_calls=[
                        {"index": 1, "id": "call_2", "name": "get_python_version"},
                    ]
                ),
                _chunk(
                    tool_calls=[
                        {"index": 2, "id": "call_3", "name": "get_env_variable"},
                    ]
                ),
                _chunk(
                    tool_calls=[
                        {"index": 2, "arguments": '{"name": "HOME"}'},
                    ]
                ),
            ]
        )
        starts = [c for c in chunks if isinstance(c, ToolCallBegin)]
        ends = [c for c in chunks if isinstance(c, ToolCallEnd)]
        assert len(starts) == 3
        assert len(ends) == 3
        si = [i for i, c in enumerate(chunks) if isinstance(c, ToolCallBegin)]
        ei = [i for i, c in enumerate(chunks) if isinstance(c, ToolCallEnd)]
        assert ei[0] < si[1]
        assert ei[1] < si[2]

    async def test_parallel_tool_calls_ids_preserved(self) -> None:
        chunks, _ = await _stream_raw(
            [
                _chunk(
                    tool_calls=[
                        {"index": 0, "id": "call_a", "name": "tool_a"},
                    ]
                ),
                _chunk(
                    tool_calls=[
                        {"index": 1, "id": "call_b", "name": "tool_b"},
                    ]
                ),
            ]
        )
        starts = [c for c in chunks if isinstance(c, ToolCallBegin)]
        assert starts[0].id == "call_a"
        assert starts[0].name == "tool_a"
        assert starts[1].id == "call_b"
        assert starts[1].name == "tool_b"

    async def test_interleaved_arguments(self) -> None:
        """Tool calls with argument deltas across multiple chunks."""
        chunks, _ = await _stream_raw(
            [
                _chunk(
                    tool_calls=[
                        {"index": 0, "id": "call_1", "name": "search"},
                    ]
                ),
                _chunk(tool_calls=[{"index": 0, "arguments": '{"q":'}]),
                _chunk(tool_calls=[{"index": 0, "arguments": ' "a"}'}]),
                _chunk(
                    tool_calls=[
                        {"index": 1, "id": "call_2", "name": "lookup"},
                    ]
                ),
                _chunk(tool_calls=[{"index": 1, "arguments": '{"id": 1}'}]),
            ]
        )
        starts = [c for c in chunks if isinstance(c, ToolCallBegin)]
        ends = [c for c in chunks if isinstance(c, ToolCallEnd)]
        deltas = [c for c in chunks if isinstance(c, ToolCallDelta)]
        assert len(starts) == 2
        assert len(ends) == 2
        assert len(deltas) == 3
        end0 = next(i for i, c in enumerate(chunks) if isinstance(c, ToolCallEnd))
        start1 = next(
            i for i, c in enumerate(chunks) if isinstance(c, ToolCallBegin) and c.name == "lookup"
        )
        assert end0 < start1

    async def test_single_chunk_multiple_tool_calls(self) -> None:
        """Multiple tool_calls packed in a single delta chunk."""
        chunks, _ = await _stream_raw(
            [
                _chunk(
                    tool_calls=[
                        {"index": 0, "id": "call_1", "name": "tool_a"},
                        {"index": 1, "id": "call_2", "name": "tool_b"},
                    ]
                ),
            ]
        )
        starts = [c for c in chunks if isinstance(c, ToolCallBegin)]
        ends = [c for c in chunks if isinstance(c, ToolCallEnd)]
        assert len(starts) == 2
        assert len(ends) == 2
        end0 = next(i for i, c in enumerate(chunks) if isinstance(c, ToolCallEnd))
        start1 = next(
            i for i, c in enumerate(chunks) if isinstance(c, ToolCallBegin) and c.id == "call_2"
        )
        assert end0 < start1

    async def test_usage(self) -> None:
        chunks, _ = await _stream_raw(
            [
                _chunk(
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    }
                ),
            ]
        )
        u = [c for c in chunks if isinstance(c, Usage)]
        assert len(u) == 1
        assert u[0].usage.input_tokens == 10
        assert u[0].usage.output_tokens == 5


class TestExtractReasoningText:
    """Tests for ``_extract_reasoning_text`` — extracting thinking content from delta."""

    def test_reasoning_content_field(self) -> None:
        class _Delta:
            reasoning_content = "thinking hard"

        assert _extract_reasoning_text(_Delta()) == "thinking hard"

    def test_reasoning_field(self) -> None:
        class _Delta:
            reasoning = "hmm"

        assert _extract_reasoning_text(_Delta()) == "hmm"

    def test_reasoning_text_field(self) -> None:
        class _Delta:
            reasoning_text = "let me see"

        assert _extract_reasoning_text(_Delta()) == "let me see"

    def test_first_nonempty_field_wins(self) -> None:
        """When multiple fields are present, the first non-empty one wins."""

        class _Delta:
            reasoning_content = "first"
            reasoning = "second"

        assert _extract_reasoning_text(_Delta()) == "first"

    def test_skips_empty_fields(self) -> None:
        class _Delta:
            reasoning_content = ""
            reasoning = "fallback"

        assert _extract_reasoning_text(_Delta()) == "fallback"

    def test_reasoning_details_minimax(self) -> None:
        """MiniMax-style reasoning_details with type=reasoning.text."""

        class _Delta:
            reasoning_details = [
                {"type": "reasoning.text", "id": "r1", "text": "let me think about this"}
            ]

        assert _extract_reasoning_text(_Delta()) == "let me think about this"

    def test_reasoning_details_ignores_non_text_types(self) -> None:
        class _Delta:
            reasoning_details = [{"type": "reasoning.encrypted", "id": "r1", "data": "..."}]

        assert _extract_reasoning_text(_Delta()) is None

    def test_no_reasoning_fields(self) -> None:
        class _Delta:
            content = "just text"

        assert _extract_reasoning_text(_Delta()) is None

    def test_reasoning_details_multiple_items(self) -> None:
        class _Delta:
            reasoning_details = [
                {"type": "reasoning.text", "id": "r1", "text": "part1"},
                {"type": "reasoning.text", "id": "r2", "text": "part2"},
            ]

        assert _extract_reasoning_text(_Delta()) == "part1part2"

    def test_scalar_field_takes_priority_over_details(self) -> None:
        """Scalar reasoning fields are checked before reasoning_details."""

        class _Delta:
            reasoning_content = "scalar"
            reasoning_details = [{"type": "reasoning.text", "id": "r1", "text": "from details"}]

        assert _extract_reasoning_text(_Delta()) == "scalar"
