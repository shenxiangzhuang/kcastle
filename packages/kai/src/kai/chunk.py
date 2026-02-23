"""Raw stream chunks produced by providers.

These are the low-level data fragments that providers yield during streaming.
The stream() function in stream.py consumes these and accumulates them into
higher-level StreamEvent objects with partial message snapshots.
"""

from dataclasses import dataclass

from kai.usage import TokenUsage


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
