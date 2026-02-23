"""Tool definition types.

Tools are declarative schemas that describe what functions the model can call.
Actual tool execution logic belongs in the kagent layer.
"""

from typing import Any

from pydantic import BaseModel


class Tool(BaseModel, frozen=True):
    """A tool definition that can be passed to an LLM.

    The parameters field should be a JSON Schema object describing the function's arguments.

    Example::

        Tool(
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
    """

    name: str
    """The name of the tool."""

    description: str
    """A description of what the tool does."""

    parameters: dict[str, Any]
    """The parameters of the tool in JSON Schema format."""
