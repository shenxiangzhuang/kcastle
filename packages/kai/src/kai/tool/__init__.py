"""Tool types for LLM tool calling.

Re-exports:

- ``Tool`` — unified tool class (schema + optional execution)
- ``ToolResult`` — result of executing a tool
"""

from kai.tool._result import ToolResult
from kai.tool._tool import Tool

__all__ = ["Tool", "ToolResult"]
