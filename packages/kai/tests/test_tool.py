"""Tests for kai.tool module."""

import pytest
from pydantic import ValidationError

from kai.tool import Tool


def test_tool_creation() -> None:
    tool = Tool(
        name="get_weather",
        description="Get the current weather",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    assert tool.name == "get_weather"
    assert tool.description == "Get the current weather"
    assert "properties" in tool.parameters


def test_tool_empty_parameters() -> None:
    tool = Tool(name="noop", description="Do nothing", parameters={})
    assert tool.parameters == {}


def test_tool_frozen() -> None:
    tool = Tool(name="a", description="b", parameters={})
    with pytest.raises(ValidationError):
        tool.name = "c"  # type: ignore[misc]
