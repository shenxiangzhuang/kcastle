"""kagent — Actor-based agent framework built on ``kai``.

Two core abstractions:

- ``Agent`` — pure handler: configuration + ``handle(state)``
- ``AgentRuntime`` — actor process: mailbox, state, lifecycle, sub-agents

Lower-level primitives:

- ``agent_step()`` — single-turn primitive, full context control
- ``agent_loop()`` — multi-turn loop with callbacks

Example::

    from kagent import Agent, AgentRuntime, UserInput, complete

    agent = Agent(llm=my_llm, system="You are helpful.", tools=[my_tool])

    # One-shot:
    msg = await complete(agent, "Hello!")

    # Actor runtime:
    runtime = AgentRuntime(agent)
    await runtime.start()
    async for event in runtime.send(UserInput("Hello!")):
        print(event)
"""

import logging as _logging

# Agent (pure handler)
from kagent.agent import Agent, complete

# Runtime (actor process)
from kagent.runtime import AgentRuntime

# Signals
from kagent.signal import ChildCompleted as ChildCompletedSignal
from kagent.signal import ChildError as ChildErrorSignal
from kagent.signal import Signal, UserInput

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
    ChildEvent,
    ChildSpawned,
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

# Tools
from kagent.tools import CheckSubAgentTool, SpawnSubAgentTool

# Trace
from kagent.trace import (
    Trace,
    TraceEntry,
    TraceManager,
    TraceStore,
)

__all__ = [
    # Core
    "Agent",
    "AgentRuntime",
    "complete",
    # Signals
    "Signal",
    "UserInput",
    "ChildCompletedSignal",
    "ChildErrorSignal",
    # Levels
    "agent_step",
    "agent_loop",
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
    "ChildSpawned",
    "ChildEvent",
    # Hooks
    "Hooks",
    "LoggingHooks",
    "MultiHooks",
    # Callback types
    "OnToolResultFn",
    "ShouldContinueFn",
    # Tools
    "SpawnSubAgentTool",
    "CheckSubAgentTool",
    # Trace
    "Trace",
    "TraceEntry",
    "TraceManager",
    "TraceStore",
]

_logging.getLogger("kagent").addHandler(_logging.NullHandler())
