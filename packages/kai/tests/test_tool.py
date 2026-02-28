"""Tests for kai.tool module."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from kai.tool import Tool, ToolResult


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


def test_tool_creation_with_raw_parameters() -> None:
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
    tool = Tool(name="noop", description="Do nothing")
    assert tool.parameters == {}


@pytest.mark.asyncio
async def test_base_tool_execute_raises() -> None:
    tool = Tool(name="schema_only", description="desc")
    with pytest.raises(NotImplementedError, match="schema_only"):
        await tool.execute({})


class EchoTool(Tool):
    name: str = "echo"
    description: str = "Echo a message."

    class Params(BaseModel):
        message: str = Field(description="The message to echo")

    async def execute(self, params: EchoTool.Params) -> ToolResult:
        return ToolResult(output=f"echo: {params.message}")


class TestExecutableTool:
    @pytest.mark.asyncio
    async def test_execute_with_typed_params(self) -> None:
        tool = EchoTool()
        params = EchoTool.Params(message="hi")
        result = await tool.execute(params)
        assert result.output == "echo: hi"
        assert result.is_error is False

    def test_is_tool(self) -> None:
        tool = EchoTool()
        assert isinstance(tool, Tool)
        assert tool.name == "echo"
        assert tool.description == "Echo a message."

    def test_auto_generated_parameters(self) -> None:
        tool = EchoTool()
        # Parameters should be auto-generated from Params
        assert tool.parameters["type"] == "object"
        assert "message" in tool.parameters["properties"]
        assert tool.parameters["properties"]["message"]["type"] == "string"
        assert "message" in tool.parameters.get("required", [])

    def test_auto_schema_includes_description(self) -> None:
        tool = EchoTool()
        assert tool.parameters["properties"]["message"]["description"] == "The message to echo"


class TestAutoSchema:
    def test_params_generates_schema(self) -> None:
        class MyTool(Tool):
            name: str = "my_tool"
            description: str = "A tool."

            class Params(BaseModel):
                x: int = Field(description="An integer")
                y: str = Field(default="hi", description="A string")

        tool = MyTool()
        assert tool.parameters["type"] == "object"
        assert "x" in tool.parameters["properties"]
        assert "y" in tool.parameters["properties"]
        assert tool.parameters["properties"]["x"]["type"] == "integer"
        assert "x" in tool.parameters["required"]
        # y has a default, so it should NOT be required
        assert "y" not in tool.parameters.get("required", [])

    def test_no_params_keeps_manual_parameters(self) -> None:
        """Tools without Params class keep their manually specified parameters."""
        tool = Tool(
            name="manual",
            description="Manual params",
            parameters={"type": "object", "properties": {"a": {"type": "string"}}},
        )
        assert tool.parameters["properties"]["a"]["type"] == "string"

    def test_params_overrides_manual_parameters(self) -> None:
        """If Params is defined, manual parameters are overridden."""

        class OverrideTool(Tool):
            name: str = "override"
            description: str = "Override test."
            parameters: dict[str, Any] = {"manually": "set"}

            class Params(BaseModel):
                value: str

        tool = OverrideTool()
        # Params should win over manual parameters
        assert "manually" not in tool.parameters
        assert "value" in tool.parameters["properties"]

    def test_no_title_in_schema(self) -> None:
        """Auto-generated schema should not contain 'title' fields."""

        class TitledTool(Tool):
            name: str = "titled"
            description: str = "Test."

            class Params(BaseModel):
                name: str

        tool = TitledTool()
        assert "title" not in tool.parameters
        # Check nested properties too
        for _prop_name, prop_schema in tool.parameters.get("properties", {}).items():
            assert "title" not in prop_schema
