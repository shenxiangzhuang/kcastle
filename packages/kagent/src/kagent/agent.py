"""Agent — pure handler with a single interface.

The ``Agent`` class is a configuration container with one method: ``handle()``.
It holds no mutable state, no queues, no flags. Given the same state, it
produces the same event stream.

Runtime concerns (mailbox, lifecycle, sub-agents, abort, steering) live in
``AgentRuntime``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from kai import Message, ProviderBase, Tool

from kagent.context import ContextBuilder
from kagent.event import AgentError, AgentEvent, TurnEnd
from kagent.hooks import Hooks
from kagent.loop import (
    ShouldContinueFn,
    agent_loop,
)
from kagent.state import AgentState
from kagent.step import OnToolResultFn
from kagent.trace.entry import TraceEntry

logger = logging.getLogger("kagent.agent")


class Agent:
    """A pure agent: configuration + handle.

    Agent is a message handler. It processes the current state and yields
    events. It does not own state, manage concurrency, or handle lifecycle.

    Example — with AgentRuntime::

        agent = Agent(llm=provider, system="You are helpful.", tools=[my_tool])
        runtime = AgentRuntime(agent)
        await runtime.start()
        async for event in runtime.send(UserInput("Hello")):
            print(event)

    Example — standalone one-shot::

        msg = await complete(agent, "What's 2+2?")
        print(msg.extract_text())
    """

    def __init__(
        self,
        *,
        llm: ProviderBase,
        system: str | None = None,
        tools: list[Tool] | None = None,
        context_builder: ContextBuilder | None = None,
        on_tool_result: OnToolResultFn | None = None,
        hooks: Hooks | None = None,
        max_turns: int = 100,
    ) -> None:
        self.llm = llm
        self.system = system
        self.tools = list(tools) if tools else []
        self.context_builder = context_builder
        self.on_tool_result = on_tool_result
        self.hooks = hooks
        self.max_turns = max_turns

    async def handle(
        self,
        state: AgentState,
        *,
        should_continue: ShouldContinueFn | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Process the current state and yield events.

        This is the single interface. Agent is a message handler — like a human,
        it receives context and responds.

        The optional ``should_continue`` parameter allows the runtime to inject
        abort/steering logic without the agent knowing about concurrency.

        Args:
            state: The mutable agent state (trace, tools, system).
            should_continue: Optional callback for loop control.
                If ``None``, continues while the assistant produces tool calls.

        Yields:
            ``AgentStart → (TurnStart → … → TurnEnd)* → AgentEnd``
        """
        async for event in agent_loop(
            llm=self.llm,
            state=state,
            context_builder=self.context_builder,
            on_tool_result=self.on_tool_result,
            should_continue=should_continue,
            hooks=self.hooks,
            max_turns=self.max_turns,
        ):
            yield event


async def complete(
    agent: Agent,
    user_input: str,
    state: AgentState | None = None,
) -> Message:
    """One-shot convenience: run agent and return the final assistant message.

    Creates a temporary state if none is provided.

    Args:
        agent: The agent to run.
        user_input: The user message.
        state: Optional existing state. If ``None``, a fresh one is created.

    Returns:
        The final assistant ``Message``.

    Raises:
        RuntimeError: If the agent ends without producing an assistant message.
    """
    if state is None:
        state = AgentState(system=agent.system, tools=list(agent.tools))
    state.trace.append(TraceEntry.user(Message(role="user", content=user_input)))

    last_assistant: Message | None = None
    last_error: BaseException | None = None
    async for event in agent.handle(state):
        if isinstance(event, TurnEnd):
            last_assistant = event.message
        elif isinstance(event, AgentError):
            last_error = event.error

    if last_assistant is None:
        if last_error is not None:
            raise RuntimeError(f"Agent failed: {last_error}") from last_error
        raise RuntimeError("Agent ended without producing an assistant message")
    return last_assistant
