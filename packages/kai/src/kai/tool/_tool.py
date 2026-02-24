"""Unified tool definition.

``Tool`` is a single class that serves as both a declarative schema
(name, description, parameters) and an executable tool (override ``execute()``).

- Schema-only usage: instantiate directly with ``Tool(name=..., ...)``
- Executable usage: subclass and override ``execute()``

Providers only read the schema fields; the ``execute()`` method is used
exclusively by the agent runtime.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from kai.tool._result import ToolResult


class Tool(BaseModel):
    """A tool that can be passed to an LLM and optionally executed.

    Schema-only example::

        tool = Tool(
            name="get_weather",
            description="Get the current weather for a location.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        )

    Executable example::

        class GetWeather(Tool):
            name: str = "get_weather"
            description: str = "Get the current weather."
            parameters: dict[str, Any] = {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            }

            async def execute(self, *, call_id: str, arguments: dict[str, Any]) -> ToolResult:
                location = arguments["location"]
                return ToolResult(output=f"Sunny in {location}")
    """

    name: str
    """The name of the tool."""

    description: str
    """A description of what the tool does."""

    parameters: dict[str, Any]
    """The parameters of the tool in JSON Schema format."""

    async def execute(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Execute the tool with the given arguments.

        Subclass and override this method to make the tool executable.
        The base implementation raises ``NotImplementedError``.

        Args:
            call_id: Unique identifier for this tool call.
            arguments: Pre-parsed arguments dict (JSON already decoded).

        Returns:
            The result of the execution.
        """
        raise NotImplementedError(
            f"Tool '{self.name}' does not implement execute(). "
            "Subclass Tool and override execute() to make it executable."
        )
