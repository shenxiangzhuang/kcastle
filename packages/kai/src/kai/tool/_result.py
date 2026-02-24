"""Tool execution result."""

from __future__ import annotations

from dataclasses import dataclass


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
