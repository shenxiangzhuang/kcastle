"""Trace entry primitives.

``TraceEntry`` is the single append-only record type for all trace data.
The ``kind`` field discriminates between entry types; ``message`` preserves
the type-safe link to ``kai.Message``; ``data`` carries kind-specific
structured data; ``meta`` holds cross-cutting metadata.

Factory classmethods create entries with ``id=0`` (unassigned).  The real
auto-incrementing ID is assigned by ``Trace.append()``.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from kai import Message
from kai.usage import TokenUsage

# ---------------------------------------------------------------------------
# TraceKind
# ---------------------------------------------------------------------------


class TraceKind(StrEnum):
    """Discriminator for trace entry types."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ANCHOR = "anchor"
    EVENT = "event"


# ---------------------------------------------------------------------------
# TraceMeta
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TraceMeta:
    """Structured metadata attached to every trace entry.

    All fields are optional — the ``Trace`` auto-fills ``timestamp``
    on append if it is zero.
    """

    timestamp: float = 0.0
    """Unix timestamp (``time.time()``).  Zero means 'auto-fill on append'."""

    run_id: str | None = None
    """Identifier for the agent run that produced this entry."""

    turn_index: int | None = None
    """Zero-based turn counter within the run."""

    usage: TokenUsage | None = None
    """Token usage from the LLM response (assistant entries only)."""

    def with_timestamp(self) -> TraceMeta:
        """Return a copy with ``timestamp`` set to *now* if it was zero."""
        if self.timestamp != 0.0:
            return self
        return TraceMeta(
            timestamp=time.time(),
            run_id=self.run_id,
            turn_index=self.turn_index,
            usage=self.usage,
        )

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.  Omits default/None fields."""
        d: dict[str, Any] = {}
        if self.timestamp:
            d["timestamp"] = self.timestamp
        if self.run_id is not None:
            d["run_id"] = self.run_id
        if self.turn_index is not None:
            d["turn_index"] = self.turn_index
        if self.usage is not None:
            d["usage"] = dataclasses.asdict(self.usage)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraceMeta:
        """Deserialize from a dict produced by ``to_dict()``."""
        usage: TokenUsage | None = None
        if "usage" in d:
            usage = TokenUsage(**d["usage"])
        return cls(
            timestamp=d.get("timestamp", 0.0),
            run_id=d.get("run_id"),
            turn_index=d.get("turn_index"),
            usage=usage,
        )

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> TraceMeta:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# TraceEntry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TraceEntry:
    """A single append-only entry in the agent trace.

    ``id`` is ``0`` when created via a factory classmethod.
    ``Trace.append()`` reassigns a real auto-incrementing ID.
    """

    id: int
    """Sequence number assigned by ``Trace`` (0 = unassigned)."""

    kind: TraceKind
    """Entry type discriminator."""

    message: Message | None = None
    """The ``kai.Message`` for message-bearing entries."""

    data: dict[str, Any] = field(default_factory=lambda: {})
    """Kind-specific structured data (e.g. anchor name, event payload)."""

    meta: TraceMeta = field(default_factory=TraceMeta)
    """Cross-cutting metadata (timing, run info, usage)."""

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind.value,
        }
        if self.message is not None:
            d["message"] = self.message.model_dump()
        if self.data:
            d["data"] = self.data
        meta = self.meta.to_dict()
        if meta:
            d["meta"] = meta
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraceEntry:
        """Deserialize from a dict produced by ``to_dict()``."""
        message: Message | None = None
        if "message" in d:
            message = Message.model_validate(d["message"])
        return cls(
            id=d["id"],
            kind=TraceKind(d["kind"]),
            message=message,
            data=d.get("data", {}),
            meta=TraceMeta.from_dict(d.get("meta", {})),
        )

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> TraceEntry:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(s))

    # --- Factory classmethods ---

    @classmethod
    def user(cls, message: Message, **meta_kw: Any) -> TraceEntry:
        """Create a user message entry."""
        return cls(id=0, kind=TraceKind.USER, message=message, meta=TraceMeta(**meta_kw))

    @classmethod
    def assistant(cls, message: Message, **meta_kw: Any) -> TraceEntry:
        """Create an assistant message entry."""
        return cls(id=0, kind=TraceKind.ASSISTANT, message=message, meta=TraceMeta(**meta_kw))

    @classmethod
    def system(cls, content: str, **meta_kw: Any) -> TraceEntry:
        """Create a system prompt entry."""
        msg = Message(role="user", content=f"[System] {content}")
        return cls(
            id=0,
            kind=TraceKind.SYSTEM,
            message=msg,
            data={"content": content},
            meta=TraceMeta(**meta_kw),
        )

    @classmethod
    def tool_call(cls, calls: list[dict[str, Any]], **meta_kw: Any) -> TraceEntry:
        """Create a tool call entry (records the raw call data)."""
        return cls(
            id=0,
            kind=TraceKind.TOOL_CALL,
            data={"calls": calls},
            meta=TraceMeta(**meta_kw),
        )

    @classmethod
    def tool_result(cls, message: Message, **meta_kw: Any) -> TraceEntry:
        """Create a tool result entry."""
        return cls(id=0, kind=TraceKind.TOOL_RESULT, message=message, meta=TraceMeta(**meta_kw))

    @classmethod
    def anchor(cls, name: str, **meta_kw: Any) -> TraceEntry:
        """Create an anchor (context boundary marker)."""
        return cls(
            id=0,
            kind=TraceKind.ANCHOR,
            data={"name": name},
            meta=TraceMeta(**meta_kw),
        )

    @classmethod
    def event(cls, name: str, data: dict[str, Any] | None = None, **meta_kw: Any) -> TraceEntry:
        """Create a lifecycle event entry."""
        payload: dict[str, Any] = {"name": name}
        if data is not None:
            payload["data"] = data
        return cls(id=0, kind=TraceKind.EVENT, data=payload, meta=TraceMeta(**meta_kw))
