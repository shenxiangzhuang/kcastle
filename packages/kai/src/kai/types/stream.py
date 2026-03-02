"""Stream types — raw chunks and high-level events.

Raw chunks are the low-level data fragments that providers yield during streaming.
The stream() function consumes these and accumulates them into higher-level
StreamEvent objects with partial message snapshots.

Stream events are emitted by stream() and carry a ``partial`` field containing
the accumulated assistant message built so far. Consumers can inspect ``partial``
at any point to see the full state.
"""

from dataclasses import dataclass

from kai.types.message import Message, ToolCall
from kai.types.usage import TokenUsage

# ---------------------------------------------------------------------------
# Raw chunks (produced by providers)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A fragment of text content."""

    text: str


@dataclass(frozen=True, slots=True)
class ThinkChunk:
    """A fragment of thinking/reasoning content."""

    text: str


@dataclass(frozen=True, slots=True)
class ThinkSignatureChunk:
    """An encrypted thinking signature (emitted at end of a think block)."""

    signature: str


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    """Marks the beginning of a tool call."""

    id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """A fragment of tool call arguments (JSON string)."""

    arguments: str


@dataclass(frozen=True, slots=True)
class ToolCallEnd:
    """Marks the end of a tool call."""


@dataclass(frozen=True, slots=True)
class UsageChunk:
    """Token usage statistics (typically emitted at end of stream)."""

    usage: TokenUsage


type Chunk = (
    TextChunk
    | ThinkChunk
    | ThinkSignatureChunk
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
    | UsageChunk
)
"""Union of all raw chunk types that a provider can yield."""


# ---------------------------------------------------------------------------
# High-level stream events (produced by stream())
# ---------------------------------------------------------------------------


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
