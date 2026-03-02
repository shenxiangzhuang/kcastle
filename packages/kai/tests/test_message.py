"""Tests for kai.message module."""

import pytest
from pydantic import ValidationError

from kai.tool import Tool
from kai.types.message import (
    ContentPart,
    Context,
    ImagePart,
    Message,
    TextPart,
    ThinkPart,
    ToolCall,
)
from kai.types.usage import TokenUsage


class TestContentParts:
    def test_text_part(self) -> None:
        p = TextPart(text="hello")
        assert p.type == "text"
        assert p.text == "hello"

    def test_think_part(self) -> None:
        p = ThinkPart(text="reasoning", signature="sig123")
        assert p.type == "think"
        assert p.text == "reasoning"
        assert p.signature == "sig123"

    def test_think_part_no_signature(self) -> None:
        p = ThinkPart(text="thinking")
        assert p.signature is None

    def test_image_part(self) -> None:
        p = ImagePart(data="base64data", mime_type="image/png")
        assert p.type == "image"
        assert p.data == "base64data"
        assert p.mime_type == "image/png"

    def test_parts_are_frozen(self) -> None:
        p = TextPart(text="hello")
        with pytest.raises(ValidationError):
            p.text = "world"  # type: ignore[misc]


class TestToolCall:
    def test_tool_call(self) -> None:
        tc = ToolCall(id="call_1", name="get_weather", arguments='{"city": "NYC"}')
        assert tc.id == "call_1"
        assert tc.name == "get_weather"

    def test_tool_call_frozen(self) -> None:
        tc = ToolCall(id="x", name="y", arguments="{}")
        with pytest.raises(ValidationError):
            tc.id = "z"  # type: ignore[misc]


class TestMessage:
    def test_string_content_coercion(self) -> None:
        msg = Message(role="user", content="hello")
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextPart)
        assert msg.content[0].text == "hello"

    def test_none_content_coercion(self) -> None:
        msg = Message(role="assistant", content=None)
        assert msg.content == []

    def test_single_part_coercion(self) -> None:
        part = TextPart(text="world")
        msg = Message(role="user", content=part)
        assert len(msg.content) == 1
        assert msg.content[0] is part

    def test_list_content(self) -> None:
        parts: list[ContentPart] = [TextPart(text="hi"), ThinkPart(text="hmm")]
        msg = Message(role="assistant", content=parts)
        assert len(msg.content) == 2

    def test_extract_text(self) -> None:
        msg = Message(
            role="assistant",
            content=[ThinkPart(text="reasoning"), TextPart(text="answer")],
        )
        assert msg.extract_text() == "answer"

    def test_extract_text_with_sep(self) -> None:
        msg = Message(
            role="assistant",
            content=[TextPart(text="hello"), TextPart(text="world")],
        )
        assert msg.extract_text(sep=" ") == "hello world"

    def test_tool_result_factory(self) -> None:
        msg = Message.tool_result("call_1", "result text")
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_1"
        assert msg.extract_text() == "result text"

    def test_tool_result_error(self) -> None:
        msg = Message.tool_result("call_1", "bad input", is_error=True)
        assert msg.extract_text() == "Error: bad input"

    def test_assistant_with_tool_calls(self) -> None:
        tc = ToolCall(id="c1", name="search", arguments='{"q": "test"}')
        msg = Message(role="assistant", content="Let me search.", tool_calls=[tc])
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"

    def test_usage_and_stop_reason(self) -> None:
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        msg = Message(
            role="assistant",
            content="done",
            usage=usage,
            stop_reason="stop",
        )
        assert msg.usage is not None
        assert msg.usage.total_tokens == 150
        assert msg.stop_reason == "stop"

    def test_serialization_single_text(self) -> None:
        msg = Message(role="user", content="hello")
        data = msg.model_dump()
        assert data["content"] == "hello"

    def test_serialization_multi_parts(self) -> None:
        parts: list[ContentPart] = [TextPart(text="hi"), ThinkPart(text="hmm")]
        msg = Message(role="assistant", content=parts)
        data = msg.model_dump()
        content: object = data["content"]
        assert isinstance(content, list)
        assert len(content) == 2  # pyright: ignore[reportUnknownArgumentType]


class TestContext:
    def test_basic_context(self) -> None:
        ctx = Context(
            system="You are helpful.",
            messages=[Message(role="user", content="hi")],
        )
        assert ctx.system == "You are helpful."
        assert len(ctx.messages) == 1

    def test_context_with_tools(self) -> None:
        tool = Tool(name="calc", description="calculator", parameters={})
        ctx = Context(
            messages=[Message(role="user", content="hi")],
            tools=[tool],
        )
        assert len(ctx.tools) == 1

    def test_context_no_system(self) -> None:
        ctx = Context(messages=[Message(role="user", content="hi")])
        assert ctx.system is None
