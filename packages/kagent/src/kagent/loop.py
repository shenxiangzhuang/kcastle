"""Level 1: Multi-turn agent loop.

``agent_loop()`` wraps ``agent_step()`` with automatic message management
and loop control. Callbacks are **plain kwargs** — no Hooks class, no
Middleware base class.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

from kai import Context, Message, Provider

from kagent.event import (
    AgentEnd,
    AgentError,
    AgentEvent,
    AgentStart,
    TurnEnd,
)
from kagent.state import AgentState
from kagent.step import OnToolResultFn, agent_step

# --- Callback type aliases ---

type BuildContextFn = Callable[[AgentState], Awaitable[Context]]
"""Build a ``kai.Context`` from the current agent state.
The SINGLE POINT where all context customization happens."""

type ShouldContinueFn = Callable[[AgentState, Message], Awaitable[bool]]
"""After each turn, decide whether to continue looping.
Receives ``(state, assistant_message)``. Return ``False`` to stop."""


def _default_build_context(state: AgentState) -> Context:
    """Build a Context from state using the default strategy."""
    return Context(
        system=state.system,
        messages=state.messages,
        tools=state.tools,
    )


async def agent_loop(
    *,
    provider: Provider,
    state: AgentState,
    build_context: BuildContextFn | None = None,
    on_tool_result: OnToolResultFn | None = None,
    should_continue: ShouldContinueFn | None = None,
    max_turns: int = 100,
) -> AsyncIterator[AgentEvent]:
    """Run a multi-turn agent loop.

    Repeatedly calls ``agent_step()`` and manages state until the model stops
    requesting tools, a callback halts the loop, or ``max_turns`` is reached.

    The ``state.messages`` list is mutated in-place — assistant messages and
    tool results are appended after each turn.

    Args:
        provider: The kai LLM provider.
        state: Mutable agent state (system, messages, tools).
        build_context: Custom context builder. If ``None``, auto-builds from state.
        on_tool_result: Intercept tool results. If ``None``, results pass through.
        should_continue: Custom continuation logic. If ``None``, continues while tool_calls.
        max_turns: Safety limit. ``0`` = unlimited.

    Yields:
        ``AgentStart → (TurnStart → … → TurnEnd)* → AgentEnd``

    Example::

        state = AgentState(
            system="You are helpful.",
            messages=[Message(role="user", content="Hello!")],
            tools=[my_tool],
        )
        async for event in agent_loop(provider=provider, state=state):
            match event:
                case TurnEnd(message=msg):
                    print(msg.extract_text())
    """
    yield AgentStart()

    turn_count = 0
    while max_turns == 0 or turn_count < max_turns:
        # === Single context construction point ===
        if build_context is not None:
            context = await build_context(state)
        else:
            context = _default_build_context(state)

        # Delegate to agent_step — all LLM streaming + tool execution happens there
        assistant_msg: Message | None = None
        async for event in agent_step(
            provider=provider,
            context=context,
            tools=state.tools,
            on_tool_result=on_tool_result,
        ):
            # Intercept TurnEnd to update state
            if isinstance(event, TurnEnd):
                assistant_msg = event.message
                state.messages.append(event.message)
                for tool_msg in event.tool_results:
                    state.messages.append(tool_msg)

            # Intercept AgentError to terminate
            if isinstance(event, AgentError):
                yield event
                yield AgentEnd(messages=state.messages)
                return

            yield event

        if assistant_msg is None:
            # Should not happen — AgentError should have been yielded
            yield AgentEnd(messages=state.messages)
            return

        turn_count += 1

        # Decide whether to continue
        if should_continue is not None:
            if not await should_continue(state, assistant_msg):
                break
        elif not assistant_msg.tool_calls:
            break

    yield AgentEnd(messages=state.messages)
