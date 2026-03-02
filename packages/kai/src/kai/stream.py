"""Core streaming API — stream() and complete().

stream() consumes raw Chunks from an LLM implementation, accumulates an assistant Message,
and yields rich StreamEvent objects with partial message snapshots.

complete() is a convenience wrapper that collects the full message.
"""

import logging
import time
from collections.abc import AsyncIterator
from typing import Any, cast

from kai.errors import EmptyResponseError, ProviderError
from kai.providers import ProviderBase
from kai.types.message import ContentPart, Context, Message, TextPart, ThinkPart, ToolCall
from kai.types.stream import (
    Chunk,
    DoneEvent,
    ErrorEvent,
    StartEvent,
    StreamEvent,
    TextChunk,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkChunk,
    ThinkDeltaEvent,
    ThinkEndEvent,
    ThinkSignatureChunk,
    ThinkStartEvent,
    ToolCallDelta,
    ToolCallDeltaEvent,
    ToolCallEnd,
    ToolCallEndEvent,
    ToolCallStart,
    ToolCallStartEvent,
    UsageChunk,
)
from kai.types.usage import TokenUsage

logger = logging.getLogger("kai.stream")


async def stream(
    llm: ProviderBase,
    context: Context,
    **kwargs: Any,
) -> AsyncIterator[StreamEvent]:
    """Stream events from an LLM.

    Consumes raw chunks from the LLM implementation and yields rich stream events,
    each carrying an accumulated partial assistant message.

    Args:
        llm: The LLM implementation to use.
        context: Conversation context (system prompt, messages, tools).
        **kwargs: API-specific options (temperature, max_tokens, etc.).

    Yields:
        StreamEvent objects. The final event is always DoneEvent or ErrorEvent.

    Example::

        async for event in stream(llm, context):
            match event:
                case TextDeltaEvent(delta=text):
                    print(text, end="")
                case DoneEvent(message=msg):
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

    state = _StreamState()

    yield StartEvent()

    try:
        async for chunk in llm.stream(context, **kwargs):
            for event in state.process_chunk(chunk):
                yield event
    except ProviderError as e:
        duration_ms = (time.perf_counter() - t0) * 1000
        logger.error(
            "LLM stream error: provider=%s model=%s error=%s duration=%.0fms",
            llm.provider,
            llm.model,
            e,
            duration_ms,
        )
        yield ErrorEvent(error=e, partial=state.build_partial())
        return
    except Exception as e:
        # Intentional recovery boundary:
        # unknown exceptions are converted into ErrorEvent so callers
        # receive a unified stream contract instead of raised exceptions.
        duration_ms = (time.perf_counter() - t0) * 1000
        logger.exception(
            "LLM stream error (unexpected): provider=%s model=%s error=%s duration=%.0fms",
            llm.provider,
            llm.model,
            e,
            duration_ms,
        )
        yield ErrorEvent(error=e, partial=state.build_partial())
        return

    # Flush any pending block
    for event in state.flush_pending():
        yield event

    # Determine stop reason
    stop_reason = "tool_use" if state.tool_calls else "stop"
    final = state.build_final(stop_reason=stop_reason)

    if not final.content and not final.tool_calls:
        duration_ms = (time.perf_counter() - t0) * 1000
        logger.error(
            "LLM stream empty response: provider=%s model=%s duration=%.0fms",
            llm.provider,
            llm.model,
            duration_ms,
        )
        yield ErrorEvent(
            error=EmptyResponseError("The LLM returned an empty response."),
            partial=final,
        )
        return

    duration_ms = (time.perf_counter() - t0) * 1000
    usage = final.usage
    logger.info(
        "LLM stream complete: provider=%s model=%s in=%d out=%d stop=%s duration=%.0fms",
        llm.provider,
        llm.model,
        usage.input_tokens if usage else 0,
        usage.output_tokens if usage else 0,
        stop_reason,
        duration_ms,
    )

    yield DoneEvent(message=final)


async def complete(
    llm: ProviderBase,
    context: Context,
    **kwargs: Any,
) -> Message:
    """Get a complete response from an LLM.

    Convenience wrapper around stream() that collects all events and returns
    the final assistant message.

    Args:
        llm: The LLM implementation to use.
        context: Conversation context (system prompt, messages, tools).
        **kwargs: API-specific options (temperature, max_tokens, etc.).

    Returns:
        The complete assistant Message with content, tool_calls, and usage.

    Raises:
        ProviderError: If the LLM call encounters an error.
        EmptyResponseError: If the LLM returns no content.
    """
    async for event in stream(llm, context, **kwargs):
        match event:
            case DoneEvent(message=message):
                return message
            case ErrorEvent(error=error):
                raise error
            case _:
                pass

    raise EmptyResponseError("Stream ended without a done or error event.")


class _StreamState:
    """Mutable state for the stream accumulation state machine.

    All chunk processing and flushing logic lives inside this class
    to keep protected attributes internal.
    """

    __slots__ = (
        "content_parts",
        "tool_calls",
        "usage",
        "content_index",
        "_text",
        "_think",
        "_think_sig",
        "_tool_id",
        "_tool_name",
        "_tool_args",
    )

    def __init__(self) -> None:
        self.content_parts: list[ContentPart] = []
        self.tool_calls: list[ToolCall] = []
        self.usage: TokenUsage | None = None
        self.content_index: int = 0

        # Accumulation buffers for the current block
        self._text: str | None = None
        self._think: str | None = None
        self._think_sig: str | None = None
        self._tool_id: str | None = None
        self._tool_name: str | None = None
        self._tool_args: str = ""

    def build_partial(self) -> Message:
        """Build the current partial message snapshot."""
        parts: list[ContentPart] = list(self.content_parts)

        # Include in-progress blocks
        if self._think is not None:
            parts.append(
                ThinkPart(
                    text=self._think,
                    signature=self._think_sig,
                )
            )
        if self._text is not None:
            parts.append(TextPart(text=self._text))

        tool_calls = list(self.tool_calls)
        if self._tool_id is not None and self._tool_name is not None:
            tool_calls.append(
                ToolCall(
                    id=self._tool_id,
                    name=self._tool_name,
                    arguments=self._tool_args,
                )
            )

        return Message(
            role="assistant",
            content=parts or None,
            tool_calls=tool_calls or None,
            usage=self.usage,
        )

    def build_final(self, *, stop_reason: str = "stop") -> Message:
        """Build the final complete message."""
        parts: list[ContentPart] = list(self.content_parts)
        sr = cast(
            Any,
            stop_reason if stop_reason in ("stop", "length", "tool_use", "error") else "stop",
        )

        return Message(
            role="assistant",
            content=parts or None,
            tool_calls=self.tool_calls or None,
            usage=self.usage,
            stop_reason=sr,
        )

    def process_chunk(self, chunk: Chunk) -> list[StreamEvent]:
        """Process a single chunk and return events to emit."""
        events: list[StreamEvent] = []

        match chunk:
            case TextChunk(text=text):
                if self._think is not None:
                    events.extend(self.flush_think())
                if self._text is None:
                    self._text = text
                    self.content_index = len(self.content_parts)
                    partial = self.build_partial()
                    events.append(
                        TextStartEvent(
                            content_index=self.content_index,
                            partial=partial,
                        )
                    )
                else:
                    self._text += text
                partial = self.build_partial()
                events.append(
                    TextDeltaEvent(
                        content_index=self.content_index,
                        delta=text,
                        partial=partial,
                    )
                )

            case ThinkChunk(text=text):
                if self._text is not None:
                    events.extend(self.flush_text())
                if self._think is None:
                    self._think = text
                    self.content_index = len(self.content_parts)
                    partial = self.build_partial()
                    events.append(
                        ThinkStartEvent(
                            content_index=self.content_index,
                            partial=partial,
                        )
                    )
                else:
                    self._think += text
                partial = self.build_partial()
                events.append(
                    ThinkDeltaEvent(
                        content_index=self.content_index,
                        delta=text,
                        partial=partial,
                    )
                )

            case ThinkSignatureChunk(signature=sig):
                self._think_sig = sig

            case ToolCallStart(id=tool_id, name=name):
                if self._text is not None:
                    events.extend(self.flush_text())
                if self._think is not None:
                    events.extend(self.flush_think())

                self._tool_id = tool_id
                self._tool_name = name
                self._tool_args = ""

                partial = self.build_partial()
                events.append(
                    ToolCallStartEvent(
                        content_index=len(self.content_parts),
                        id=tool_id,
                        name=name,
                        partial=partial,
                    )
                )

            case ToolCallDelta(arguments=args):
                self._tool_args += args
                partial = self.build_partial()
                events.append(
                    ToolCallDeltaEvent(
                        content_index=len(self.content_parts),
                        arguments_delta=args,
                        partial=partial,
                    )
                )

            case ToolCallEnd():
                if self._tool_id is not None and self._tool_name is not None:
                    tool_call = ToolCall(
                        id=self._tool_id,
                        name=self._tool_name,
                        arguments=self._tool_args,
                    )
                    self.tool_calls.append(tool_call)
                    partial = self.build_partial()
                    events.append(
                        ToolCallEndEvent(
                            content_index=len(self.content_parts),
                            tool_call=tool_call,
                            partial=partial,
                        )
                    )
                    self._tool_id = None
                    self._tool_name = None
                    self._tool_args = ""

            case UsageChunk(usage=usage):
                self.usage = usage

        return events

    def flush_text(self) -> list[StreamEvent]:
        """Flush the current text block into content_parts."""
        events: list[StreamEvent] = []
        if self._text is not None:
            text = self._text
            part = TextPart(text=text)
            self.content_parts.append(part)
            self._text = None
            partial = self.build_partial()
            events.append(
                TextEndEvent(
                    content_index=len(self.content_parts) - 1,
                    text=text,
                    partial=partial,
                )
            )
        return events

    def flush_think(self) -> list[StreamEvent]:
        """Flush the current think block into content_parts."""
        events: list[StreamEvent] = []
        if self._think is not None:
            text = self._think
            part = ThinkPart(text=text, signature=self._think_sig)
            self.content_parts.append(part)
            self._think = None
            self._think_sig = None
            partial = self.build_partial()
            events.append(
                ThinkEndEvent(
                    content_index=len(self.content_parts) - 1,
                    text=text,
                    partial=partial,
                )
            )
        return events

    def flush_pending(self) -> list[StreamEvent]:
        """Flush any pending blocks."""
        events: list[StreamEvent] = []
        events.extend(self.flush_think())
        events.extend(self.flush_text())

        if self._tool_id is not None and self._tool_name is not None:
            tool_call = ToolCall(
                id=self._tool_id,
                name=self._tool_name,
                arguments=self._tool_args,
            )
            self.tool_calls.append(tool_call)
            partial = self.build_partial()
            events.append(
                ToolCallEndEvent(
                    content_index=len(self.content_parts),
                    tool_call=tool_call,
                    partial=partial,
                )
            )
            self._tool_id = None
            self._tool_name = None
            self._tool_args = ""

        return events
