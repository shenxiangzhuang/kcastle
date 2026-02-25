"""Trace manager for multi-trace orchestration.

``TraceManager`` manages the lifecycle of multiple named traces and
optionally orchestrates persistence through a ``TraceStore``.  It is the
counterpart of Republic's ``TapeManager``.

When no ``TraceStore`` is provided, traces are purely in-memory.
When a store is provided, entries are persisted incrementally on
every ``append()``, and existing traces can be loaded for session resume.
"""

from __future__ import annotations

from kagent.trace.entry import TraceEntry
from kagent.trace.store import TraceStore
from kagent.trace.trace import Trace


class TraceManager:
    """Manages multiple traces with optional persistent storage.

    Example — in-memory only::

        manager = TraceManager()
        trace = manager.create("my-session")
        entry = manager.append(trace.id, TraceEntry.user(msg))

    Example — with persistence::

        store = JsonlTraceStore(Path("~/.kagent/traces"))
        manager = TraceManager(store=store)
        trace = manager.create("my-session")  # persists header
        entry = manager.append(trace.id, some_entry)  # persists entry

    Example — session resume::

        manager = TraceManager(store=store)
        trace = manager.load("abc123")  # loads from store
        # Continue appending …
    """

    def __init__(self, *, store: TraceStore | None = None) -> None:
        self._store = store
        self._traces: dict[str, Trace] = {}

    # --- Lifecycle ---

    def create(self, name: str = "") -> Trace:
        """Create a new trace and register it.

        If a store is configured, the trace header is persisted immediately.
        """
        trace = Trace(name=name)
        self._traces[trace.id] = trace
        if self._store is not None:
            self._store.create(trace.id, trace.name, trace.created_at)
        return trace

    def register(self, trace: Trace) -> None:
        """Register an externally-created trace (no store write)."""
        self._traces[trace.id] = trace

    def get(self, trace_id: str) -> Trace:
        """Get a trace by ID.  Raises ``KeyError`` if not found."""
        return self._traces[trace_id]

    def load(self, trace_id: str) -> Trace:
        """Load a trace from the store (session resume).

        The loaded trace is registered in the manager for subsequent use.
        Requires a store to be configured.

        Raises:
            RuntimeError: If no store is configured.
            KeyError: If the trace does not exist in the store.
        """
        if self._store is None:
            raise RuntimeError("Cannot load: no TraceStore configured")
        header, entries = self._store.load(trace_id)
        trace = Trace.from_records(
            id=header["id"],
            name=header.get("name", ""),
            created_at=header.get("created_at", 0.0),
            entries=entries,
        )
        self._traces[trace.id] = trace
        return trace

    def list_traces(self) -> list[str]:
        """List all trace IDs (in-memory).

        To list persisted traces, use ``store.list_traces()`` directly.
        """
        return sorted(self._traces.keys())

    # --- Entry operations ---

    def append(self, trace_id: str, entry: TraceEntry) -> TraceEntry:
        """Append an entry to the specified trace.

        If a store is configured, the entry is also persisted.
        Returns the stored entry with its assigned ID.
        """
        trace = self._traces[trace_id]
        stored = trace.append(entry)
        if self._store is not None:
            self._store.append(trace_id, stored)
        return stored

    def reset(self, trace_id: str) -> None:
        """Clear all entries in the specified trace."""
        self._traces[trace_id].reset()
