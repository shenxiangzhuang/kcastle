"""Core streaming API — stream() and complete().

``stream()`` consumes events from a Provider, forwards them to the caller,
and accumulates an assistant Message that is yielded as the final ``Done`` event.

``complete()`` is a convenience wrapper that collects the full message.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Sequence
from typing import Unpack

from kai.errors import ErrorKind, KaiError
from kai.providers import ProviderBase
from kai.providers.base import GenerationKwargs
from kai.types.message import ContentPart, Context, Message, TextPart, ThinkPart, ToolCall
from kai.types.stream import (
    Done,
    Error,
    StreamEvent,
    TextDelta,
    ThinkDelta,
    ThinkSignature,
    ToolCallBegin,
    ToolCallDelta,
    ToolCallEnd,
    Usage,
)
from kai.types.usage import TokenUsage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def stream(
    llm: ProviderBase,
    context: Context,
    **kwargs: Unpack[GenerationKwargs],
) -> AsyncIterator[StreamEvent]:
    """Stream events from an LLM.

    Provider events are forwarded directly to callers.  In addition, a final
    ``Done`` or ``Error`` event is appended carrying the accumulated assistant
    message.

    Example::

        async for event in stream(llm, context):
            match event:
                case TextDelta(delta=text):
                    print(text, end="")
                case Done(message=msg):
                    print(f"\\nDone. Tokens: {msg.usage}")
    """
    msg_count = len(context.messages) if context.messages else 0
    tool_count = len(context.tools) if context.tools else 0
    logger.debug(
        "LLM stream start: provider=%s model=%s messages=%d tools=%d",
        llm.provider,
        llm.model,
        msg_count,
        tool_count,
    )
    t0 = time.perf_counter()

    events: list[StreamEvent] = []

    try:
        async for event in llm.stream(context, **kwargs):
            events.append(event)
            yield event
    except KaiError as e:
        logger.error("LLM stream error: %s/%s %s %.0fms", llm.provider, llm.model, e, _ms(t0))
        yield Error(error=e)
        return
    except Exception as e:
        logger.error(
            "LLM stream error: %s/%s %s %.0fms",
            llm.provider,
            llm.model,
            e,
            _ms(t0),
            exc_info=True,
        )
        yield Error(error=e)
        return

    message = _build_message(events)

    if not message.content and not message.tool_calls:
        err = KaiError(ErrorKind.EMPTY_RESPONSE, "The LLM returned an empty response.")
        logger.error("LLM stream error: %s/%s %s %.0fms", llm.provider, llm.model, err, _ms(t0))
        yield Error(error=err)
        return

    usage = message.usage
    stop = "tool_use" if message.tool_calls else "stop"
    logger.info(
        "LLM stream done: %s/%s in=%d out=%d stop=%s %.0fms",
        llm.provider,
        llm.model,
        usage.input_tokens if usage else 0,
        usage.output_tokens if usage else 0,
        stop,
        _ms(t0),
    )

    yield Done(message=message)


async def complete(
    llm: ProviderBase,
    context: Context,
    **kwargs: Unpack[GenerationKwargs],
) -> Message:
    """Get a complete response from an LLM.

    Convenience wrapper around ``stream()`` that returns the final message.

    Raises:
        KaiError: If the LLM call encounters an error or returns no content.
    """
    async for event in stream(llm, context, **kwargs):
        match event:
            case Done(message=message):
                return message
            case Error(error=error):
                raise error
            case _:
                pass

    raise KaiError(ErrorKind.EMPTY_RESPONSE, "Stream ended without a done or error event.")


# ---------------------------------------------------------------------------
# Build message from events — pure function, no class
# ---------------------------------------------------------------------------


def _build_message(events: Sequence[StreamEvent]) -> Message:  # noqa: C901
    """Build an assistant ``Message`` from a sequence of stream events."""
    parts: list[ContentPart] = []
    tool_calls: list[ToolCall] = []
    usage: TokenUsage | None = None

    text_buf: list[str] = []
    think_buf: list[str] = []
    think_sig: str | None = None

    tool_id: str | None = None
    tool_name: str | None = None
    tool_args: list[str] = []

    def flush_text() -> None:
        if text_buf:
            parts.append(TextPart(text="".join(text_buf)))
            text_buf.clear()

    def flush_think() -> None:
        nonlocal think_sig
        if think_buf:
            parts.append(ThinkPart(text="".join(think_buf), signature=think_sig))
            think_buf.clear()
            think_sig = None

    def flush_tool() -> None:
        nonlocal tool_id, tool_name
        if tool_id is not None and tool_name is not None:
            tool_calls.append(ToolCall(id=tool_id, name=tool_name, arguments="".join(tool_args)))
            tool_id = None
            tool_name = None
            tool_args.clear()

    for event in events:
        match event:
            case TextDelta(delta=text):
                flush_think()
                text_buf.append(text)
            case ThinkDelta(delta=text):
                flush_text()
                think_buf.append(text)
            case ThinkSignature(signature=sig):
                think_sig = sig
            case ToolCallBegin(id=tid, name=name):
                flush_text()
                flush_think()
                tool_id = tid
                tool_name = name
                tool_args.clear()
            case ToolCallDelta(arguments=args):
                tool_args.append(args)
            case ToolCallEnd():
                flush_tool()
            case Usage(usage=u):
                usage = u
            case Done() | Error():
                pass

    flush_text()
    flush_think()
    flush_tool()

    return Message(
        role="assistant",
        content=parts or None,
        tool_calls=tool_calls or None,
        usage=usage,
        stop_reason="tool_use" if tool_calls else "stop",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000
