"""Tests for OpenAI Responses provider — message serialization.

Tests exercise the public ``stream()`` method with a mocked OpenAI client,
verifying the ``input`` payload sent to the Responses API.
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

from kai.providers.openai import OpenAIResponses
from kai.types.message import Context, ImagePart, Message, TextPart, ToolCall
from kai.types.stream import Chunk


def _ctx(*messages: Message, system: str | None = None) -> Context:
    return Context(system=system, messages=list(messages))


async def _stream_raw(
    context: Context | None = None,
) -> tuple[list[Chunk], dict[str, Any]]:
    """Call ``OpenAIResponses.stream()`` with a mocked client.

    The stream is always empty; only the captured ``input`` kwarg matters.
    """
    if context is None:
        context = _ctx(Message(role="user", content="hello"))

    captured: dict[str, Any] = {}

    async def _fake_create(**kwargs: Any) -> AsyncIterator[Any]:
        captured.update(kwargs)

        async def _gen() -> AsyncIterator[Any]:
            for _ in ():
                yield

        return _gen()

    with patch("kai.providers.openai.AsyncOpenAI") as mock_cls:
        mock_cls.return_value.responses.create = _fake_create
        provider = OpenAIResponses(model="test-model", api_key="test-key")

    output = [c async for c in provider.stream(context)]
    return output, captured


class TestMessageSerialization:
    """Context → ``input`` kwarg sent to the Responses API."""

    async def test_system_uses_developer_role(self) -> None:
        ctx = _ctx(system="Be helpful.")
        _, kw = await _stream_raw(ctx)
        items = kw["input"]
        assert len(items) == 1
        assert items[0]["role"] == "developer"
        assert items[0]["content"] == "Be helpful."

    async def test_no_system(self) -> None:
        ctx = _ctx(Message(role="user", content="Hi"))
        _, kw = await _stream_raw(ctx)
        items = kw["input"]
        assert len(items) == 1
        assert items[0]["role"] == "user"

    async def test_system_plus_messages(self) -> None:
        ctx = _ctx(
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
            system="Be helpful.",
        )
        _, kw = await _stream_raw(ctx)
        items = kw["input"]
        assert items[0]["role"] == "developer"
        assert items[1]["role"] == "user"

    async def test_user_text(self) -> None:
        ctx = _ctx(Message(role="user", content="Hello"))
        _, kw = await _stream_raw(ctx)
        item = kw["input"][0]
        assert item["role"] == "user"
        assert item["content"] == "Hello"

    async def test_user_multimodal(self) -> None:
        ctx = _ctx(
            Message(
                role="user",
                content=[
                    TextPart(text="Look"),
                    ImagePart(data="abc123", mime_type="image/png"),
                ],
            )
        )
        _, kw = await _stream_raw(ctx)
        content: list[Any] = kw["input"][0].get("content")
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "input_text"
        assert content[1]["type"] == "input_image"

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
            Message.tool_result("tc1", "sunny"),
        )
        _, kw = await _stream_raw(ctx)
        fr = next(i for i in kw["input"] if i.get("type") == "function_call_output")
        assert fr["call_id"] == "tc1"
        assert fr["output"] == "sunny"

    async def test_assistant_text(self) -> None:
        ctx = _ctx(
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
        )
        _, kw = await _stream_raw(ctx)
        msg_items = [i for i in kw["input"] if i.get("type") == "message"]
        assert len(msg_items) == 1
        item = msg_items[0]
        assert item["role"] == "assistant"
        content = item["content"]
        assert len(content) == 1
        assert content[0]["type"] == "output_text"
        assert content[0]["text"] == "Hello!"

    async def test_assistant_with_tool_calls(self) -> None:
        ctx = _ctx(
            Message(role="user", content="Hi"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(id="tc1", name="fn", arguments='{"x":1}'),
                ],
            ),
        )
        _, kw = await _stream_raw(ctx)
        fcs = [i for i in kw["input"] if i.get("type") == "function_call"]
        assert len(fcs) == 1
        fc = fcs[0]
        assert fc["call_id"] == "tc1"
        assert fc["name"] == "fn"
        assert fc["arguments"] == '{"x":1}'

    async def test_assistant_text_and_tool_calls(self) -> None:
        ctx = _ctx(
            Message(role="user", content="Hi"),
            Message(
                role="assistant",
                content="thinking...",
                tool_calls=[
                    ToolCall(id="tc1", name="fn", arguments="{}"),
                ],
            ),
        )
        _, kw = await _stream_raw(ctx)
        types = [i.get("type") for i in kw["input"] if i.get("type")]
        assert "message" in types
        assert "function_call" in types

    async def test_single_text_content_is_string(self) -> None:
        ctx = _ctx(Message(role="user", content=[TextPart(text="hello")]))
        _, kw = await _stream_raw(ctx)
        assert kw["input"][0]["content"] == "hello"

    async def test_empty_content_is_empty_string(self) -> None:
        ctx = _ctx(Message(role="user", content=[]))
        _, kw = await _stream_raw(ctx)
        assert kw["input"][0]["content"] == ""
