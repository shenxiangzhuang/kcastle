"""Tool types for LLM tool calling.

Re-exports:

- ``Tool`` — unified tool class (schema + optional execution)
- ``ToolResult`` — result of executing a tool
- ``get_params_class`` — extract inner Params from a Tool subclass
"""

from kai.tool._tool import Tool, ToolResult, get_params_class

__all__ = ["Tool", "ToolResult", "get_params_class"]
