"""Level 1: Multi-turn agent loop.

``agent_loop()`` wraps ``agent_step()`` with automatic trace management,
loop control, and lifecycle hooks.  Each turn's assistant message and tool
results are recorded as ``TraceEntry`` objects in ``state.trace``.

Hooks (optional) are called at each lifecycle point for observability.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from uuid import uuid4

from kai import Message, ProviderBase
from kai.types.usage import TokenUsage

from kagent.context import ContextBuilder, DefaultBuilder
from kagent.event import (
    AgentEnd,
    AgentError,
    AgentEvent,
    AgentStart,
    ToolExecEnd,
    ToolExecStart,
    TurnEnd,
    TurnStart,
)
from kagent.hooks import Hooks
from kagent.state import AgentState
from kagent.step import OnToolResultFn, agent_step
from kagent.trace.entry import TraceEntry

logger = logging.getLogger("kagent.loop")

# --- Callback type aliases ---

type ShouldContinueFn = Callable[[AgentState, Message], Awaitable[bool]]
"""After each turn, decide whether to continue looping.
Receives ``(state, assistant_message)``. Return ``False`` to stop."""


def _record_turn_end(
    state: AgentState,
    event: TurnEnd,
    *,
    run_id: str,
    turn_index: int,
) -> TokenUsage | None:
    """Record a completed turn in the trace and return token usage.

    Delegates trace-recording responsibility to this helper so that the main
    loop stays focused on orchestration (SRP / Delegation principles).
    """
    state.trace.append(
        TraceEntry.assistant(
            event.message,
            run_id=run_id,
            turn_index=turn_index,
            usage=event.message.usage,
        )
    )
    for tool_msg in event.tool_results:
        state.trace.append(
            TraceEntry.tool_result(
                tool_msg,
                run_id=run_id,
                turn_index=turn_index,
            )
        )
    return event.message.usage


async def agent_loop(
    *,
    llm: ProviderBase,
    state: AgentState,
    context_builder: ContextBuilder | None = None,
    on_tool_result: OnToolResultFn | None = None,
    should_continue: ShouldContinueFn | None = None,
    hooks: Hooks | None = None,
    max_turns: int = 100,
) -> AsyncIterator[AgentEvent]:
    """Run a multi-turn agent loop.

    Repeatedly calls ``agent_step()`` and manages state until the model stops
    requesting tools, a callback halts the loop, or ``max_turns`` is reached.

    Each turn's assistant message and tool results are recorded as
    ``TraceEntry`` objects in ``state.trace``.

    Args:
        llm: The kai LLM implementation.
        state: Mutable agent state (system, trace, tools).
        context_builder: Controls how ``AgentState`` becomes ``kai.Context``.
            If ``None``, uses ``DefaultBuilder()`` (pass-through).
        on_tool_result: Intercept tool results. If ``None``, results pass through.
        should_continue: Custom continuation logic. If ``None``, continues while tool_calls.
        hooks: Optional lifecycle hooks for observability (logging, tracing, metrics).
        max_turns: Safety limit. ``0`` = unlimited.

    Yields:
        ``AgentStart → (TurnStart → … → TurnEnd)* → AgentEnd``

    Example::

        state = AgentState(system="You are helpful.")
        state.trace.append(TraceEntry.user(Message(role="user", content="Hello!")))
        async for event in agent_loop(llm=provider, state=state):
            match event:
                case TurnEnd(message=msg):
                    print(msg.extract_text())
    """
    _hooks = hooks or Hooks()

    yield AgentStart()

    builder = context_builder or DefaultBuilder()
    run_id = uuid4().hex[:8]
    agent_t0 = time.perf_counter()
    total_usage: TokenUsage | None = None

    _hooks.on_agent_start(run_id=run_id, model=llm.model, provider=llm.provider)

    turn_count = 0
    while max_turns == 0 or turn_count < max_turns:
        # === Single context construction point ===
        context = await builder.build(state)

        turn_t0 = time.perf_counter()
        _hooks.on_turn_start(run_id=run_id, turn_index=turn_count)
        _hooks.on_llm_start(run_id=run_id, turn_index=turn_count)

        # Delegate to agent_step — all LLM streaming + tool execution happens there
        assistant_msg: Message | None = None
        async for event in agent_step(
            llm=llm,
            context=context,
            tools=state.tools,
            on_tool_result=on_tool_result,
        ):
            # --- Hook dispatch on intercepted events ---
            match event:
                case TurnStart():
                    # TurnStart from step means LLM call is about to begin.
                    # on_llm_start already called above.
                    pass

                case ToolExecStart():
                    _hooks.on_tool_start(
                        run_id=run_id,
                        turn_index=turn_count,
                        call_id=event.call_id,
                        tool_name=event.tool_name,
                        arguments=event.arguments,
                    )

                case ToolExecEnd():
                    _hooks.on_tool_end(
                        run_id=run_id,
                        turn_index=turn_count,
                        call_id=event.call_id,
                        tool_name=event.tool_name,
                        result=event.result,
                        duration_ms=event.duration_ms,
                        is_error=event.is_error,
                    )

                case TurnEnd():
                    assistant_msg = event.message

                    # LLM end hook
                    _hooks.on_llm_end(
                        run_id=run_id,
                        turn_index=turn_count,
                        message=event.message,
                        duration_ms=event.llm_duration_ms,
                    )

                    # Record turn in trace (delegated to helper)
                    turn_usage = _record_turn_end(
                        state, event, run_id=run_id, turn_index=turn_count
                    )

                    # Accumulate total usage
                    if turn_usage is not None:
                        total_usage = total_usage + turn_usage if total_usage else turn_usage

                    # Turn end hook
                    turn_duration_ms = (time.perf_counter() - turn_t0) * 1000
                    _hooks.on_turn_end(
                        run_id=run_id,
                        turn_index=turn_count,
                        message=event.message,
                        tool_results=event.tool_results,
                        llm_duration_ms=event.llm_duration_ms,
                        duration_ms=turn_duration_ms,
                    )

                case AgentError():
                    logger.error("[%s] Agent error at turn %d: %s", run_id, turn_count, event.error)
                    agent_duration_ms = (time.perf_counter() - agent_t0) * 1000
                    _hooks.on_agent_end(
                        run_id=run_id,
                        turn_count=turn_count,
                        duration_ms=agent_duration_ms,
                        usage=total_usage,
                    )
                    yield event
                    yield AgentEnd(messages=state.messages)
                    return

                case _:
                    pass

            yield event

        if assistant_msg is None:
            # Should not happen — AgentError should have been yielded
            agent_duration_ms = (time.perf_counter() - agent_t0) * 1000
            _hooks.on_agent_end(
                run_id=run_id,
                turn_count=turn_count,
                duration_ms=agent_duration_ms,
                usage=total_usage,
            )
            yield AgentEnd(messages=state.messages)
            return

        turn_count += 1

        # Decide whether to continue
        if should_continue is not None:
            if not await should_continue(state, assistant_msg):
                break
        elif not assistant_msg.tool_calls:
            break

    agent_duration_ms = (time.perf_counter() - agent_t0) * 1000
    _hooks.on_agent_end(
        run_id=run_id,
        turn_count=turn_count,
        duration_ms=agent_duration_ms,
        usage=total_usage,
    )
    yield AgentEnd(messages=state.messages)
