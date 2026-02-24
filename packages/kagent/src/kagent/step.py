"""Level 0: Single-turn primitive.

``agent_step()`` executes exactly one LLM call plus tool execution.
It accepts a ``kai.Context`` directly — the caller has full control
over what goes to the LLM. No state management, no loop, no continuation.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from kai import Context, DoneEvent, ErrorEvent, Message, Provider, Tool, ToolResult, stream

from kagent.event import (
    AgentError,
    AgentEvent,
    StreamChunk,
    ToolExecEnd,
    ToolExecStart,
    TurnEnd,
    TurnStart,
)


def _find_tool(name: str, tools: list[Tool]) -> Tool | None:
    """Find a tool by name."""
    for tool in tools:
        if tool.name == name:
            return tool
    return None


async def agent_step(
    *,
    provider: Provider,
    context: Context,
    tools: list[Tool],
) -> AsyncIterator[AgentEvent]:
    """Execute a single LLM turn: one LLM call + tool execution.

    This is a pure function that does not manage any state. The caller
    builds the ``Context`` and decides what to do with the results.

    Args:
        provider: The kai LLM provider.
        context: The complete LLM context (system + messages + tool schemas).
        tools: Executable tools for dispatching tool calls from the LLM response.

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
    assistant_msg: Message | None = None
    async for stream_event in await stream(provider, context):
        yield StreamChunk(event=stream_event)

        match stream_event:
            case DoneEvent(message=msg):
                assistant_msg = msg
            case ErrorEvent(error=err):
                yield AgentError(error=err)
                return
            case _:
                pass

    if assistant_msg is None:
        yield AgentError(error=RuntimeError("Stream ended without DoneEvent or ErrorEvent"))
        return

    # Execute tool calls (if any)
    tool_result_messages: list[Message] = []
    if assistant_msg.tool_calls:
        for tool_call in assistant_msg.tool_calls:
            tool = _find_tool(tool_call.name, tools)

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
                yield ToolExecEnd(
                    call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=result,
                    is_error=True,
                )
                tool_result_messages.append(
                    Message.tool_result(tool_call.id, result.output, is_error=True)
                )
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

            yield ToolExecEnd(
                call_id=tool_call.id,
                tool_name=tool_call.name,
                result=result,
                is_error=result.is_error,
            )

            tool_result_messages.append(
                Message.tool_result(tool_call.id, result.output, is_error=result.is_error)
            )

    yield TurnEnd(message=assistant_msg, tool_results=tool_result_messages)
