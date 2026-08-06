"""Events emitted by an agent run."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from openai.types.responses import ResponseFunctionToolCall, ResponseUsage


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result returned through the Agent's environment-effect port."""

    output: str
    is_error: bool = False


type ToolExecutor = Callable[[ResponseFunctionToolCall], Awaitable[ToolResult]]


@dataclass(frozen=True, slots=True)
class ModelStarted:
    """A model turn is starting."""

    turn: int


@dataclass(frozen=True, slots=True)
class TextDelta:
    """Incremental assistant text."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolStarted:
    """A tool call is about to execute."""

    call: ResponseFunctionToolCall


@dataclass(frozen=True, slots=True)
class ToolFinished:
    """A tool call completed or failed recoverably."""

    call: ResponseFunctionToolCall
    output: str
    is_error: bool


@dataclass(frozen=True, slots=True)
class CompactionStarted:
    """Automatic or manual context compaction started."""

    tokens_before: int


@dataclass(frozen=True, slots=True)
class CompactionFinished:
    """Context compaction completed."""

    tokens_before: int
    first_kept_id: int
    summary: str


@dataclass(frozen=True, slots=True)
class RunFinished:
    """The agent is settled: no tools or queued messages remain."""

    output: str
    response_id: str
    usage: ResponseUsage | None


type AgentEvent = (
    ModelStarted
    | TextDelta
    | ToolStarted
    | ToolFinished
    | CompactionStarted
    | CompactionFinished
    | RunFinished
)
