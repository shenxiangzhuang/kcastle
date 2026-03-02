"""kai.types — Core type definitions for kai.

Re-exports all types from submodules for convenience:
- ``message``: Message, Context, ContentPart, ToolCall, etc.
- ``stream``: Raw Chunk types and high-level StreamEvent types.
- ``usage``: TokenUsage statistics.
"""

from kai.types.message import (
    ContentPart,
    Context,
    ImagePart,
    Message,
    TextPart,
    ThinkPart,
    ToolCall,
)
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

__all__ = [
    # Message types
    "Message",
    "ContentPart",
    "TextPart",
    "ThinkPart",
    "ImagePart",
    "ToolCall",
    "Context",
    # Stream chunks
    "Chunk",
    "TextChunk",
    "ThinkChunk",
    "ThinkSignatureChunk",
    "ToolCallStart",
    "ToolCallDelta",
    "ToolCallEnd",
    "UsageChunk",
    # Stream events
    "StreamEvent",
    "StartEvent",
    "TextStartEvent",
    "TextDeltaEvent",
    "TextEndEvent",
    "ThinkStartEvent",
    "ThinkDeltaEvent",
    "ThinkEndEvent",
    "ToolCallStartEvent",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "DoneEvent",
    "ErrorEvent",
    # Usage
    "TokenUsage",
]
