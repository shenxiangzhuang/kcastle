"""Level 0: Single-turn primitive.

``agent_step()`` executes exactly one LLM call plus tool execution.
It accepts a ``kai.Context`` directly — the caller has full control
over what goes to the LLM. No state management, no loop, no continuation.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable

from kai import Context, DoneEvent, ErrorEvent, Message, Provider, Tool, ToolResult, stream
from kai.tool import get_params_class
from pydantic import ValidationError

from kagent.event import (
    AgentError,
    AgentEvent,
    StreamChunk,
    ToolExecEnd,
    ToolExecStart,
    TurnEnd,
    TurnStart,
)

logger = logging.getLogger("kagent.step")

type OnToolResultFn = Callable[[str, str, ToolResult], Awaitable[ToolResult]]
"""Called after tool execution. Receives ``(call_id, tool_name, result)``.
Return (possibly modified) result."""


def _build_tool_map(tools: list[Tool]) -> dict[str, Tool]:
    """Build a name → tool lookup dict."""
    return {tool.name: tool for tool in tools}


async def _execute_tool(tool: Tool, arguments: dict[str, object]) -> ToolResult:
    """Validate arguments and execute a tool.

    If the tool defines an inner ``Params(BaseModel)`` class, arguments
    are validated through Pydantic before calling ``execute(params)``.
    Otherwise, the raw arguments dict is passed directly.
    """
    params_cls = get_params_class(type(tool))
    if params_cls is not None:
        try:
            params = params_cls.model_validate(arguments)
        except ValidationError as e:
            return ToolResult.error(f"Invalid arguments: {e}")
        return await tool.execute(params)
    return await tool.execute(arguments)


async def agent_step(
    *,
    provider: Provider,
    context: Context,
    tools: list[Tool],
    on_tool_result: OnToolResultFn | None = None,
) -> AsyncIterator[AgentEvent]:
    """Execute a single LLM turn: one LLM call + tool execution.

    This is a pure function that does not manage any state. The caller
    builds the ``Context`` and decides what to do with the results.

    Args:
        provider: The kai LLM provider.
        context: The complete LLM context (system + messages + tool schemas).
        tools: Executable tools for dispatching tool calls from the LLM response.
        on_tool_result: Optional callback to intercept/modify tool results.

    Yields:
        ``TurnStart`` → ``StreamChunk``… → ``ToolExecStart/End``… → ``TurnEnd``

    Example::

        context = Context(
            system="You are helpful.",
            messages=[Message(role="user", content="Hello!")],
            tools=my_tools,
        )
        async for event in agent_step(provider=provider, context=context, tools=my_tools):
            match event:
                case TurnEnd(message=msg):
                    print(msg.extract_text())
    """
    yield TurnStart()

    # Stream LLM response
    llm_t0 = time.perf_counter()
    assistant_msg: Message | None = None
    async for stream_event in await stream(provider, context):
        yield StreamChunk(event=stream_event)

        match stream_event:
            case DoneEvent(message=msg):
                assistant_msg = msg
            case ErrorEvent(error=err):
                logger.error("LLM stream error in agent_step: %s", err)
                yield AgentError(error=err)
                return
            case _:
                pass

    llm_duration_ms = (time.perf_counter() - llm_t0) * 1000

    if assistant_msg is None:
        logger.error("Stream ended without DoneEvent or ErrorEvent")
        yield AgentError(error=RuntimeError("Stream ended without DoneEvent or ErrorEvent"))
        return

    # Execute tool calls (if any)
    tool_map = _build_tool_map(tools)
    tool_result_messages: list[Message] = []
    if assistant_msg.tool_calls:
        for tool_call in assistant_msg.tool_calls:
            # Parse JSON arguments
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
                logger.warning(
                    "Tool %s: invalid JSON arguments: %s",
                    tool_call.name,
                    e,
                )

                if on_tool_result is not None:
                    result = await on_tool_result(tool_call.id, tool_call.name, result)

                yield ToolExecEnd(
                    call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=result,
                    is_error=result.is_error,
                    duration_ms=0.0,
                )
                tool_result_messages.append(
                    Message.tool_result(tool_call.id, result.output, is_error=result.is_error)
                )
                continue

            yield ToolExecStart(
                call_id=tool_call.id,
                tool_name=tool_call.name,
                arguments=arguments,
            )

            # Look up and execute tool
            tool_t0 = time.perf_counter()
            tool = tool_map.get(tool_call.name)
            if tool is None:
                result = ToolResult.error(f"Tool not found: {tool_call.name}")
                logger.warning("Tool not found: %s", tool_call.name)
            else:
                try:
                    result = await _execute_tool(tool, arguments)
                except Exception as e:
                    result = ToolResult.error(str(e))
                    logger.error("Tool %s execution error: %s", tool_call.name, e)

            # on_tool_result interception
            if on_tool_result is not None:
                result = await on_tool_result(tool_call.id, tool_call.name, result)

            tool_duration_ms = (time.perf_counter() - tool_t0) * 1000

            logger.info(
                "Tool %s completed: is_error=%s duration=%.0fms",
                tool_call.name,
                result.is_error,
                tool_duration_ms,
            )

            yield ToolExecEnd(
                call_id=tool_call.id,
                tool_name=tool_call.name,
                result=result,
                is_error=result.is_error,
                duration_ms=tool_duration_ms,
            )

            tool_result_messages.append(
                Message.tool_result(tool_call.id, result.output, is_error=result.is_error)
            )

    yield TurnEnd(
        message=assistant_msg,
        tool_results=tool_result_messages,
        llm_duration_ms=llm_duration_ms,
    )
