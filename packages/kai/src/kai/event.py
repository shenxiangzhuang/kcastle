"""High-level stream events with accumulated partial message snapshots.

These events are emitted by stream() and carry a `partial` field containing
the accumulated assistant message built so far, following the pi-ai design.
Consumers can inspect `partial` at any point to see the full state.
"""

from dataclasses import dataclass

from kai.message import Message, ToolCall


@dataclass(frozen=True, slots=True)
class StartEvent:
    """Emitted when the stream begins."""

    type: str = "start"


@dataclass(frozen=True, slots=True)
class TextStartEvent:
    """Emitted when a new text content block begins."""

    content_index: int
    partial: Message
    type: str = "text_start"


@dataclass(frozen=True, slots=True)
class TextDeltaEvent:
    """Emitted for each text fragment."""

    content_index: int
    delta: str
    partial: Message
    type: str = "text_delta"


@dataclass(frozen=True, slots=True)
class TextEndEvent:
    """Emitted when a text content block is complete."""

    content_index: int
    text: str
    partial: Message
    type: str = "text_end"


@dataclass(frozen=True, slots=True)
class ThinkStartEvent:
    """Emitted when a new thinking block begins."""

    content_index: int
    partial: Message
    type: str = "think_start"


@dataclass(frozen=True, slots=True)
class ThinkDeltaEvent:
    """Emitted for each thinking fragment."""

    content_index: int
    delta: str
    partial: Message
    type: str = "think_delta"


@dataclass(frozen=True, slots=True)
class ThinkEndEvent:
    """Emitted when a thinking block is complete."""

    content_index: int
    text: str
    partial: Message
    type: str = "think_end"


@dataclass(frozen=True, slots=True)
class ToolCallStartEvent:
    """Emitted when a tool call begins."""

    content_index: int
    id: str
    name: str
    partial: Message
    type: str = "toolcall_start"


@dataclass(frozen=True, slots=True)
class ToolCallDeltaEvent:
    """Emitted for each tool call arguments fragment."""

    content_index: int
    arguments_delta: str
    partial: Message
    type: str = "toolcall_delta"


@dataclass(frozen=True, slots=True)
class ToolCallEndEvent:
    """Emitted when a tool call is complete."""

    content_index: int
    tool_call: ToolCall
    partial: Message
    type: str = "toolcall_end"


@dataclass(frozen=True, slots=True)
class DoneEvent:
    """Emitted when the stream completes successfully."""

    message: Message
    type: str = "done"


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    """Emitted when the stream encounters an error."""

    error: Exception
    partial: Message
    type: str = "error"


type StreamEvent = (
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkStartEvent
    | ThinkDeltaEvent
    | ThinkEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | DoneEvent
    | ErrorEvent
)
"""Union of all stream event types."""
