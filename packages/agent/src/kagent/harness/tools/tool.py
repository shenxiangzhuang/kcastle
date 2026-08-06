"""Typed local functions exposed as OpenAI Responses tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from openai.types.responses import FunctionToolParam, ResponseFunctionToolCall
from pydantic import BaseModel, ValidationError

from kagent.events import ToolResult
from kagent.harness.env import Env


@dataclass(frozen=True, slots=True)
class Tool[ParamsT: BaseModel]:
    name: str
    description: str
    params: type[ParamsT]
    run: Callable[[ParamsT, Env], Awaitable[str]]
    requires_approval: bool = False

    def schema(self) -> FunctionToolParam:
        parameters = self.params.model_json_schema()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
            "strict": False,
        }

    async def execute(self, call: ResponseFunctionToolCall, env: Env) -> ToolResult:
        try:
            params = self.params.model_validate_json(call.arguments, extra="forbid")
        except ValidationError as error:
            return ToolResult(f"Invalid arguments: {error}", is_error=True)
        try:
            return ToolResult(await self.run(params, env))
        except Exception as error:
            return ToolResult(f"{type(error).__name__}: {error}", is_error=True)


type AnyTool = Tool[Any]


class ToolRuntime:
    """Resolve, approve, and execute local function tools."""

    def __init__(self, env: Env, tools: Sequence[AnyTool] = ()) -> None:
        self.env = env
        self.tools = tuple(tools)

    @property
    def schemas(self) -> tuple[FunctionToolParam, ...]:
        return tuple(tool.schema() for tool in self.tools)

    async def execute(
        self,
        call: ResponseFunctionToolCall,
        *,
        approve: Callable[[ResponseFunctionToolCall], Awaitable[bool]] | None = None,
    ) -> ToolResult:
        tool = next((candidate for candidate in self.tools if candidate.name == call.name), None)
        if tool is None:
            return ToolResult(f"Tool not found: {call.name}", is_error=True)
        if tool.requires_approval and (approve is None or not await approve(call)):
            return ToolResult("Tool call denied by user", is_error=True)
        return await tool.execute(call, self.env)
