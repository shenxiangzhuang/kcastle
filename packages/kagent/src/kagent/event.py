"""Agent lifecycle events.

Discriminated union of all events emitted during agent execution.
Every event has a ``type`` literal field for pattern matching.

Events follow the nesting: ``AgentStart → (TurnStart → StreamChunk… →
ToolExecStart/End… → TurnEnd)* → AgentEnd``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from kai import Message, StreamEvent, ToolResult

# --- Turn-level events ---


@dataclass(frozen=True, slots=True)
class TurnStart:
    """A new LLM turn is starting."""

    type: Literal["turn_start"] = "turn_start"


@dataclass(frozen=True, slots=True)
class TurnEnd:
    """An LLM turn completed (assistant message + any tool results)."""

    message: Message
    tool_results: list[Message]
    type: Literal["turn_end"] = "turn_end"


# --- Streaming events (forwarded from kai) ---


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """A streaming event from the LLM, forwarded from kai."""

    event: StreamEvent
    type: Literal["stream_chunk"] = "stream_chunk"


# --- Tool execution events ---


@dataclass(frozen=True, slots=True)
class ToolExecStart:
    """A tool execution is starting."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    type: Literal["tool_exec_start"] = "tool_exec_start"


@dataclass(frozen=True, slots=True)
class ToolExecEnd:
    """A tool execution completed."""

    call_id: str
    tool_name: str
    result: ToolResult
    is_error: bool
    type: Literal["tool_exec_end"] = "tool_exec_end"


# --- Agent lifecycle ---


@dataclass(frozen=True, slots=True)
class AgentStart:
    """The agent loop is starting."""

    type: Literal["agent_start"] = "agent_start"


@dataclass(frozen=True, slots=True)
class AgentEnd:
    """The agent loop has ended."""

    messages: list[Message]
    type: Literal["agent_end"] = "agent_end"


# --- Abort ---


@dataclass(frozen=True, slots=True)
class AgentAbort:
    """The agent loop was aborted by the user."""

    messages: list[Message]
    type: Literal["agent_abort"] = "agent_abort"


# --- Error ---


@dataclass(frozen=True, slots=True)
class AgentError:
    """An error occurred during the agent loop."""

    error: Exception
    type: Literal["agent_error"] = "agent_error"


type AgentEvent = (
    AgentStart
    | AgentEnd
    | AgentAbort
    | TurnStart
    | TurnEnd
    | StreamChunk
    | ToolExecStart
    | ToolExecEnd
    | AgentError
)
"""Union of all agent lifecycle events."""
