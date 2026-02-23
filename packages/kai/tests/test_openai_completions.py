"""Tests for OpenAI Completions provider — message serialization and stream conversion.

Tests exercise the public ``stream_raw()`` method with a mocked OpenAI client,
verifying both the message format sent to the API and the chunks produced from
the streamed response.
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

from openai.types.chat import ChatCompletionChunk

from kai.chunk import (
    Chunk,
    TextChunk,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageChunk,
)
from kai.message import Context, ImagePart, Message, TextPart, ThinkPart, ToolCall
from kai.providers.openai import OpenAICompletions

# --- Helpers ---


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
) -> tuple[list[Chunk], dict[str, Any]]:
    """Call ``OpenAICompletions.stream_raw()`` with a mocked client.

    Returns ``(output_chunks, captured_create_kwargs)``.
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

    with patch("kai.providers.openai._completions.AsyncOpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create = _fake_create
        provider = OpenAICompletions(model="test-model", api_key="test-key")

    output = [c async for c in provider.stream_raw(context)]
    return output, captured


# --- Message serialization ---


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


# --- Stream conversion ---


class TestStreamConversion:
    """ChatCompletionChunk stream → kai Chunk sequence."""

    async def test_text_chunks(self) -> None:
        chunks, _ = await _stream_raw(
            [
                _chunk(content="Hello"),
                _chunk(content=" world"),
            ]
        )
        texts = [c for c in chunks if isinstance(c, TextChunk)]
        assert len(texts) == 2
        assert texts[0].text == "Hello"
        assert texts[1].text == " world"

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
        assert isinstance(chunks[0], ToolCallStart)
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
        starts = [c for c in chunks if isinstance(c, ToolCallStart)]
        ends = [c for c in chunks if isinstance(c, ToolCallEnd)]
        assert len(starts) == 3
        assert len(ends) == 3
        si = [i for i, c in enumerate(chunks) if isinstance(c, ToolCallStart)]
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
        starts = [c for c in chunks if isinstance(c, ToolCallStart)]
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
        starts = [c for c in chunks if isinstance(c, ToolCallStart)]
        ends = [c for c in chunks if isinstance(c, ToolCallEnd)]
        deltas = [c for c in chunks if isinstance(c, ToolCallDelta)]
        assert len(starts) == 2
        assert len(ends) == 2
        assert len(deltas) == 3
        end0 = next(i for i, c in enumerate(chunks) if isinstance(c, ToolCallEnd))
        start1 = next(
            i for i, c in enumerate(chunks) if isinstance(c, ToolCallStart) and c.name == "lookup"
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
        starts = [c for c in chunks if isinstance(c, ToolCallStart)]
        ends = [c for c in chunks if isinstance(c, ToolCallEnd)]
        assert len(starts) == 2
        assert len(ends) == 2
        end0 = next(i for i, c in enumerate(chunks) if isinstance(c, ToolCallEnd))
        start1 = next(
            i for i, c in enumerate(chunks) if isinstance(c, ToolCallStart) and c.id == "call_2"
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
        u = [c for c in chunks if isinstance(c, UsageChunk)]
        assert len(u) == 1
        assert u[0].usage.input_tokens == 10
        assert u[0].usage.output_tokens == 5
