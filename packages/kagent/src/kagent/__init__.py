"""kagent — Core agent runtime for the K agent framework.

A context-first, three-level agent SDK built on ``kai``:

- Level 0: ``agent_step()`` — single-turn primitive, full context control
- Level 1: ``agent_loop()`` — multi-turn loop with callbacks
- Level 2: ``Agent`` — stateful SDK with steering and follow-up

Example::

    from kagent import Agent

    agent = Agent(provider=my_provider, system="You are helpful.", tools=[my_tool])
    msg = await agent.complete("Hello!")
    print(msg.extract_text())
"""

# Level 2: Stateful agent SDK
# Tools (re-exported from kai for convenience)
from kai import Tool, ToolResult

from kagent.agent import Agent

# Events
from kagent.event import (
    AgentEnd,
    AgentError,
    AgentEvent,
    AgentStart,
    StreamChunk,
    ToolExecEnd,
    ToolExecStart,
    TurnEnd,
    TurnStart,
)

# Level 1: Multi-turn loop
from kagent.loop import (
    BuildContextFn,
    OnToolResultFn,
    ShouldContinueFn,
    agent_loop,
)

# State
from kagent.state import AgentState

# Level 0: Single-turn primitive
from kagent.step import agent_step

__all__ = [
    # Levels
    "agent_step",
    "agent_loop",
    "Agent",
    # State
    "AgentState",
    # Tools
    "Tool",
    "ToolResult",
    # Events
    "AgentEvent",
    "AgentStart",
    "AgentEnd",
    "AgentError",
    "TurnStart",
    "TurnEnd",
    "StreamChunk",
    "ToolExecStart",
    "ToolExecEnd",
    # Callback types
    "BuildContextFn",
    "OnToolResultFn",
    "ShouldContinueFn",
]
