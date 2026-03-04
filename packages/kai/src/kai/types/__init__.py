"""kai.types — Core type definitions for kai.

Re-exports all types from submodules for convenience:
- ``message``: Message, Context, ContentPart, ToolCall, etc.
- ``stream``: StreamEvent union and individual event types.
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

__all__ = [
    # Message types
    "Message",
    "ContentPart",
    "TextPart",
    "ThinkPart",
    "ImagePart",
    "ToolCall",
    "Context",
    # Stream events
    "StreamEvent",
    "TextDelta",
    "ThinkDelta",
    "ThinkSignature",
    "ToolCallBegin",
    "ToolCallDelta",
    "ToolCallEnd",
    "Usage",
    "Done",
    "Error",
    # Usage
    "TokenUsage",
]
