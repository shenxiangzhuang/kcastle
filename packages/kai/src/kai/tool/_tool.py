"""Unified tool definition and execution result.

``Tool`` is a single class that serves as both a declarative schema
(name, description, parameters) and an executable tool (override ``execute()``).

- Schema-only usage: instantiate directly with ``Tool(name=..., parameters=...)``
- Executable usage: subclass, define inner ``Params(BaseModel)``, override ``execute()``

The inner ``Params`` class enables **typed tool parameters**: JSON Schema is
auto-generated from the Pydantic model, and the ``execute()`` method receives
a validated ``Params`` instance instead of a raw dict.

Providers only read the schema fields; the ``execute()`` method is used
exclusively by the agent runtime.

``ToolResult`` holds the text output of a tool execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result of executing a tool.

    The ``output`` field contains the tool's text output.
    ``is_error`` indicates whether the execution failed.

    Example::

        result = ToolResult(output="The weather is sunny.")
        error = ToolResult.error("Something went wrong")
    """

    output: str
    """The text output of the tool execution."""

    is_error: bool = False
    """Whether this result represents an error."""

    @staticmethod
    def error(message: str) -> ToolResult:
        """Create an error result."""
        return ToolResult(output=message, is_error=True)


class Tool(BaseModel):
    """A tool that can be passed to an LLM and optionally executed.

    Schema-only example (raw JSON Schema)::

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

    Executable example (typed Params — recommended)::

        class GetWeather(Tool):
            name: str = "get_weather"
            description: str = "Get the current weather."

            class Params(BaseModel):
                location: str = Field(description="City name")

            async def execute(self, params: GetWeather.Params) -> ToolResult:
                return ToolResult(output=f"Sunny in {params.location}")
    """

    name: str
    """The name of the tool."""

    description: str
    """A description of what the tool does."""

    parameters: dict[str, Any] = {}
    """The parameters of the tool in JSON Schema format.

    If a ``Params`` inner class is defined, this field is auto-populated
    from the Pydantic model's JSON Schema during initialization. Manual
    ``parameters`` are ignored when ``Params`` is present.
    """

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        """Auto-generate parameters from inner Params class if present."""
        params_cls = get_params_class(type(self))
        if params_cls is not None:
            from kai.tool._schema import params_to_json_schema

            object.__setattr__(self, "parameters", params_to_json_schema(params_cls))

    async def execute(self, params: Any) -> ToolResult:  # noqa: ANN401
        """Execute the tool with validated parameters.

        Subclass and override this method to make the tool executable.
        The base implementation raises ``NotImplementedError``.

        When using a typed ``Params`` inner class, the runtime validates
        the raw arguments dict against the Params model and passes the
        resulting instance to this method. Override with a typed signature::

            async def execute(self, params: MyTool.Params) -> ToolResult: ...

        Args:
            params: Validated parameters instance (``Params``), or a raw
                ``dict[str, Any]`` if no ``Params`` class is defined.

        Returns:
            The result of the execution.
        """
        raise NotImplementedError(
            f"Tool '{self.name}' does not implement execute(). "
            "Subclass Tool and override execute() to make it executable."
        )


def get_params_class(tool_cls: type[Tool]) -> type[BaseModel] | None:
    """Get the inner Params class from a Tool subclass, if defined."""
    params_cls = getattr(tool_cls, "Params", None)
    if (
        params_cls is not None
        and isinstance(params_cls, type)
        and issubclass(params_cls, BaseModel)
    ):
        return params_cls
    return None
