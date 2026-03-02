"""kagent — Core agent runtime for the K agent framework.

A context-first, three-level agent SDK built on ``kai``:

- Level 0: ``agent_step()`` — single-turn primitive, full context control
- Level 1: ``agent_loop()`` — multi-turn loop with callbacks
- Level 2: ``Agent`` — stateful SDK with steering and follow-up

Example::

    from kagent import Agent

    agent = Agent(llm=my_llm, system=\"You are helpful.\", tools=[my_tool])
    msg = await agent.complete("Hello!")
    print(msg.extract_text())
"""

import logging as _logging

# Level 2: Stateful agent SDK
from kagent.agent import Agent

# Context builders
from kagent.context import (
    AdaptiveBuilder,
    CompactingBuilder,
    ContextBuilder,
    ContextSwitchTool,
    DefaultBuilder,
    SlidingWindowBuilder,
)

# Events
from kagent.event import (
    AgentAbort,
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

# Hooks (observability)
from kagent.hooks import Hooks, LoggingHooks, MultiHooks

# Level 1: Multi-turn loop
from kagent.loop import (
    ShouldContinueFn,
    agent_loop,
)

# State
from kagent.state import AgentState

# Level 0: Single-turn primitive
from kagent.step import OnToolResultFn, agent_step

# Trace
from kagent.trace import (
    Trace,
    TraceEntry,
    TraceManager,
    TraceStore,
)

__all__ = [
    # Levels
    "agent_step",
    "agent_loop",
    "Agent",
    # State
    "AgentState",
    # Context builders
    "ContextBuilder",
    "DefaultBuilder",
    "SlidingWindowBuilder",
    "CompactingBuilder",
    "AdaptiveBuilder",
    "ContextSwitchTool",
    # Events
    "AgentEvent",
    "AgentStart",
    "AgentEnd",
    "AgentAbort",
    "AgentError",
    "TurnStart",
    "TurnEnd",
    "StreamChunk",
    "ToolExecStart",
    "ToolExecEnd",
    # Hooks
    "Hooks",
    "LoggingHooks",
    "MultiHooks",
    # Callback types
    "OnToolResultFn",
    "ShouldContinueFn",
    # Trace
    "Trace",
    "TraceEntry",
    "TraceManager",
    "TraceStore",
]

_logging.getLogger("kagent").addHandler(_logging.NullHandler())
