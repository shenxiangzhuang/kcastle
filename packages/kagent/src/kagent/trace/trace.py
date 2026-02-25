"""Append-only execution trace.

``Trace`` is a named, append-only container of ``TraceEntry`` objects.
Each trace has a unique ``id`` (UUID hex) and a human-readable ``name``.
It is the single source of truth for an agent session's execution history.

``Trace`` is a pure in-memory data structure — it does **not** hold a
``TraceStore``.  Persistence is orchestrated externally by ``TraceManager``.
"""

from __future__ import annotations

import time
from uuid import uuid4

from kai import Message

from kagent.trace.entry import TraceEntry, TraceKind


class Trace:
    """An append-only execution trace.  One per agent session.

    Example::

        trace = Trace(name="my-session")
        entry = trace.append(TraceEntry.user(Message(role="user", content="Hi")))
        print(entry.id)  # 1

        # Derive the classic message list
        msgs = trace.messages()
    """

    def __init__(self, *, id: str | None = None, name: str = "") -> None:
        self._id = id or uuid4().hex
        self._name = name
        self._created_at = time.time()
        self._entries: list[TraceEntry] = []
        self._next_id: int = 1

    # --- Identity ---

    @property
    def id(self) -> str:
        """Globally unique trace identifier."""
        return self._id

    @property
    def name(self) -> str:
        """Human-readable session name."""
        return self._name

    @property
    def created_at(self) -> float:
        """Unix timestamp when this trace was created."""
        return self._created_at

    # --- Core operations ---

    def append(self, entry: TraceEntry) -> TraceEntry:
        """Append an entry, assigning an auto-increment ID and timestamp.

        The original ``entry`` (with ``id=0``) is not modified — a new
        ``TraceEntry`` is created with the assigned ID and (if needed)
        a filled-in timestamp.

        Returns the stored entry with its assigned ID.
        """
        meta = entry.meta.with_timestamp()
        stored = TraceEntry(
            id=self._next_id,
            kind=entry.kind,
            message=entry.message,
            data=dict(entry.data),
            meta=meta,
        )
        self._next_id += 1
        self._entries.append(stored)
        return stored

    @property
    def entries(self) -> list[TraceEntry]:
        """All entries in append order (defensive copy)."""
        return list(self._entries)

    def __len__(self) -> int:
        """Number of entries in the trace."""
        return len(self._entries)

    # --- Derived views ---

    def messages(self) -> list[Message]:
        """Extract ordered ``Message`` list from message-bearing entries.

        Returns messages from entries with ``kind`` in
        ``{USER, ASSISTANT, TOOL_RESULT}`` — the kinds that carry a
        ``kai.Message`` relevant to the LLM conversation.
        """
        _message_kinds = {TraceKind.USER, TraceKind.ASSISTANT, TraceKind.TOOL_RESULT}
        result: list[Message] = []
        for entry in self._entries:
            if entry.kind in _message_kinds and entry.message is not None:
                result.append(entry.message)
        return result

    def by_kind(self, *kinds: TraceKind) -> list[TraceEntry]:
        """Filter entries by one or more kinds."""
        kind_set = set(kinds)
        return [e for e in self._entries if e.kind in kind_set]

    # --- Lifecycle ---

    def reset(self) -> None:
        """Clear all entries and reset the ID counter."""
        self._entries.clear()
        self._next_id = 1

    # --- Reconstruction ---

    @classmethod
    def from_records(
        cls,
        *,
        id: str,
        name: str,
        created_at: float,
        entries: list[TraceEntry],
    ) -> Trace:
        """Reconstruct a ``Trace`` from persisted data.

        Used by ``TraceStore.load()`` to restore a trace from storage.
        """
        trace = cls(id=id, name=name)
        trace._created_at = created_at
        trace._entries = list(entries)
        trace._next_id = max((e.id for e in entries), default=0) + 1
        return trace
