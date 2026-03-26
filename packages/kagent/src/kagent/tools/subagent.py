"""Sub-agent tools for the agent runtime.

These tools are created by ``AgentRuntime`` and injected into the agent's
tool set. They capture runtime callbacks to spawn and query sub-agents.

Sub-agents run in the background and report results back to the parent
runtime's mailbox.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from kai import Tool, ToolResult


class SpawnSubAgentTool(Tool):
    """Spawn a sub-agent to work on a task in the background.

    The sub-agent runs independently and reports back when done.
    Use ``check_sub_agent`` to monitor progress.
    """

    name: str = "spawn_sub_agent"
    description: str = (
        "Spawn a sub-agent to work on a task in the background. "
        "Use this for complex, multi-step tasks that can run independently. "
        "Returns the sub-agent ID for tracking."
    )

    _spawn_fn: Callable[..., str] = PrivateAttr()

    class Params(BaseModel):
        task: str = Field(description="Clear description of the task for the sub-agent.")
        system: str | None = Field(
            default=None,
            description="Optional system prompt override for the sub-agent.",
        )

    @classmethod
    def for_runtime(cls, runtime: Any) -> SpawnSubAgentTool:
        """Create a tool bound to the given runtime's spawn method."""
        tool = cls.model_construct(
            name="spawn_sub_agent",
            description=cls.model_fields["description"].default,
        )
        tool._spawn_fn = runtime.spawn_child
        return tool

    async def execute(self, params: SpawnSubAgentTool.Params) -> ToolResult:
        child_id = self._spawn_fn(task=params.task, system=params.system)
        return ToolResult(
            output=f"Sub-agent spawned with ID: {child_id}. "
            f"Use check_sub_agent to monitor progress."
        )


class CheckSubAgentTool(Tool):
    """Check the status or result of a running sub-agent."""

    name: str = "check_sub_agent"
    description: str = (
        "Check the status of sub-agents. "
        "Call without arguments to see all sub-agents, "
        "or with a specific child_id for details."
    )

    _status_fn: Callable[..., str] = PrivateAttr()

    class Params(BaseModel):
        child_id: str | None = Field(
            default=None,
            description="Sub-agent ID to check. Omit to list all.",
        )

    @classmethod
    def for_runtime(cls, runtime: Any) -> CheckSubAgentTool:
        """Create a tool bound to the given runtime's status method."""
        tool = cls.model_construct(
            name="check_sub_agent",
            description=cls.model_fields["description"].default,
        )
        tool._status_fn = runtime.child_status
        return tool

    async def execute(self, params: CheckSubAgentTool.Params) -> ToolResult:
        status = self._status_fn(child_id=params.child_id)
        return ToolResult(output=status)
