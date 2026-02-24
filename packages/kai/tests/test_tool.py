"""Tests for kai.tool module."""

from typing import Any

import pytest

from kai.tool import Tool, ToolResult

# --- ToolResult ---


class TestToolResult:
    def test_basic_result(self) -> None:
        result = ToolResult(output="hello")
        assert result.output == "hello"
        assert result.is_error is False

    def test_error_result(self) -> None:
        result = ToolResult.error("something failed")
        assert result.output == "something failed"
        assert result.is_error is True

    def test_frozen(self) -> None:
        result = ToolResult(output="hello")
        with pytest.raises(AttributeError):
            result.is_error = True  # type: ignore[misc]


# --- Tool ---


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


@pytest.mark.asyncio
async def test_base_tool_execute_raises() -> None:
    tool = Tool(name="schema_only", description="desc", parameters={})
    with pytest.raises(NotImplementedError, match="schema_only"):
        await tool.execute(call_id="c1", arguments={})


# --- Executable Tool subclass ---


class EchoTool(Tool):
    name: str = "echo"
    description: str = "Echo a message."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    async def execute(self, *, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output=f"echo: {arguments['message']}")


class TestExecutableTool:
    @pytest.mark.asyncio
    async def test_execute(self) -> None:
        tool = EchoTool()
        result = await tool.execute(call_id="c1", arguments={"message": "hi"})
        assert result.output == "echo: hi"
        assert result.is_error is False

    def test_is_tool(self) -> None:
        tool = EchoTool()
        assert isinstance(tool, Tool)
        assert tool.name == "echo"
        assert tool.description == "Echo a message."

    def test_schema_fields_accessible(self) -> None:
        tool = EchoTool()
        assert tool.parameters["type"] == "object"
        assert "message" in tool.parameters["properties"]
