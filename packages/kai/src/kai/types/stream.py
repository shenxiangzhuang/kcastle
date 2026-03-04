"""Stream event types for kai.

A single set of event types flows through the entire pipeline:
providers yield them, ``stream()`` forwards and accumulates them,
consumers match on them.  No intermediate "chunk" layer.
"""

from dataclasses import dataclass

from kai.types.message import Message
from kai.types.usage import TokenUsage

# ---------------------------------------------------------------------------
# Stream events — produced by providers *and* consumed by callers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A fragment of text content."""

    delta: str


@dataclass(frozen=True, slots=True)
class ThinkDelta:
    """A fragment of thinking/reasoning content."""

    delta: str


@dataclass(frozen=True, slots=True)
class ThinkSignature:
    """Encrypted thinking signature (for cross-turn replay)."""

    signature: str


@dataclass(frozen=True, slots=True)
class ToolCallBegin:
    """A tool call is starting."""

    id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """A fragment of tool call arguments (JSON string)."""

    arguments: str


@dataclass(frozen=True, slots=True)
class ToolCallEnd:
    """A tool call has finished."""


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage statistics (typically at end of stream)."""

    usage: TokenUsage


@dataclass(frozen=True, slots=True)
class Done:
    """Stream completed successfully. Carries the final accumulated message."""

    message: Message


@dataclass(frozen=True, slots=True)
class Error:
    """Stream encountered an error."""

    error: Exception


type StreamEvent = (
    TextDelta
    | ThinkDelta
    | ThinkSignature
    | ToolCallBegin
    | ToolCallDelta
    | ToolCallEnd
    | Usage
    | Done
    | Error
)
"""Union of all stream event types."""
