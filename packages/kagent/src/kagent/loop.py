"""Level 1: Multi-turn agent loop.

``agent_loop()`` wraps ``agent_step()`` with automatic message management
and loop control. Callbacks are **plain kwargs** — no Hooks class, no
Middleware base class.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable

from kai import Context, DoneEvent, ErrorEvent, Message, Provider, Tool, ToolResult, stream

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
from kagent.state import AgentState

# --- Callback type aliases ---

type BuildContextFn = Callable[[AgentState], Awaitable[Context]]
"""Build a ``kai.Context`` from the current agent state.
The SINGLE POINT where all context customization happens."""

type OnToolResultFn = Callable[[str, str, ToolResult], Awaitable[ToolResult]]
"""Called after tool execution. Receives ``(call_id, tool_name, result)``.
Return (possibly modified) result."""

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


def _find_tool(name: str, tools: list[Tool]) -> Tool | None:
    """Find a tool by name."""
    for tool in tools:
        if tool.name == name:
            return tool
    return None


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

    Repeatedly calls the LLM and executes tool calls until the model stops
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

        yield TurnStart()

        # Stream LLM response
        assistant_msg: Message | None = None
        async for stream_event in await stream(provider, context):
            yield StreamChunk(event=stream_event)

            match stream_event:
                case DoneEvent(message=msg):
                    assistant_msg = msg
                case ErrorEvent(error=err):
                    yield AgentError(error=err)
                    yield AgentEnd(messages=state.messages)
                    return
                case _:
                    pass

        if assistant_msg is None:
            yield AgentError(error=RuntimeError("Stream ended without DoneEvent or ErrorEvent"))
            yield AgentEnd(messages=state.messages)
            return

        # Append assistant message to state
        state.messages.append(assistant_msg)

        # Execute tool calls
        tool_result_messages: list[Message] = []
        if assistant_msg.tool_calls:
            for tool_call in assistant_msg.tool_calls:
                tool = _find_tool(tool_call.name, state.tools)

                try:
                    arguments: dict[str, object] = json.loads(tool_call.arguments)
                except json.JSONDecodeError as e:
                    arguments = {}
                    result = ToolResult.error(f"Invalid JSON arguments: {e}")
                    yield ToolExecStart(
                        call_id=tool_call.id,
                        tool_name=tool_call.name,
                        arguments=arguments,
                    )

                    if on_tool_result is not None:
                        result = await on_tool_result(tool_call.id, tool_call.name, result)

                    yield ToolExecEnd(
                        call_id=tool_call.id,
                        tool_name=tool_call.name,
                        result=result,
                        is_error=True,
                    )
                    tool_msg = Message.tool_result(tool_call.id, result.output, is_error=True)
                    state.messages.append(tool_msg)
                    tool_result_messages.append(tool_msg)
                    continue

                yield ToolExecStart(
                    call_id=tool_call.id,
                    tool_name=tool_call.name,
                    arguments=arguments,
                )

                if tool is None:
                    result = ToolResult.error(f"Tool not found: {tool_call.name}")
                else:
                    try:
                        result = await tool.execute(
                            call_id=tool_call.id,
                            arguments=arguments,
                        )
                    except Exception as e:
                        result = ToolResult.error(str(e))

                # on_tool_result interception
                if on_tool_result is not None:
                    result = await on_tool_result(tool_call.id, tool_call.name, result)

                yield ToolExecEnd(
                    call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=result,
                    is_error=result.is_error,
                )

                tool_msg = Message.tool_result(
                    tool_call.id, result.output, is_error=result.is_error
                )
                state.messages.append(tool_msg)
                tool_result_messages.append(tool_msg)

        yield TurnEnd(message=assistant_msg, tool_results=tool_result_messages)
        turn_count += 1

        # Decide whether to continue
        if should_continue is not None:
            if not await should_continue(state, assistant_msg):
                break
        elif not assistant_msg.tool_calls:
            break

    yield AgentEnd(messages=state.messages)
