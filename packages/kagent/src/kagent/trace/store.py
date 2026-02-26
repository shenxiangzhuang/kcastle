"""Trace persistence.

``TraceStore`` is a Protocol for append-only trace storage.
Two implementations are provided:

- ``InMemoryTraceStore`` — dict-backed, for testing.
- ``JsonlTraceStore`` — JSONL file-backed, one file per trace.

JSONL file format::

    Line 1:  {"type": "header", "id": "...", "name": "...", "created_at": 1740...}
    Line 2+: {"id": 1, "kind": "user", "message": {...}, "data": {}, "meta": {...}}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from kagent.trace.entry import TraceEntry

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class TraceStore(Protocol):
    """Append-only trace storage interface."""

    def create(self, trace_id: str, name: str, created_at: float) -> None:
        """Initialize a new trace in the store."""
        ...

    def append(self, trace_id: str, entry: TraceEntry) -> None:
        """Append a single entry to an existing trace."""
        ...

    def load(self, trace_id: str) -> tuple[dict[str, Any], list[TraceEntry]]:
        """Load a trace.  Returns ``(header_dict, entries)``."""
        ...

    def list_traces(self) -> list[str]:
        """List all trace IDs in the store."""
        ...


# ---------------------------------------------------------------------------
# InMemoryTraceStore
# ---------------------------------------------------------------------------


class InMemoryTraceStore:
    """Dict-backed in-memory store.  For testing and ephemeral usage."""

    def __init__(self) -> None:
        self._headers: dict[str, dict[str, Any]] = {}
        self._entries: dict[str, list[TraceEntry]] = {}

    def create(self, trace_id: str, name: str, created_at: float) -> None:
        self._headers[trace_id] = {
            "id": trace_id,
            "name": name,
            "created_at": created_at,
        }
        self._entries[trace_id] = []

    def append(self, trace_id: str, entry: TraceEntry) -> None:
        if trace_id not in self._entries:
            raise KeyError(f"Trace '{trace_id}' does not exist")
        self._entries[trace_id].append(entry)

    def load(self, trace_id: str) -> tuple[dict[str, Any], list[TraceEntry]]:
        if trace_id not in self._headers:
            raise KeyError(f"Trace '{trace_id}' does not exist")
        header = dict(self._headers[trace_id])
        entries = list(self._entries[trace_id])
        return header, entries

    def list_traces(self) -> list[str]:
        return sorted(self._headers.keys())


# ---------------------------------------------------------------------------
# JsonlTraceStore
# ---------------------------------------------------------------------------


class JsonlTraceStore:
    """JSONL file-backed trace store.  One file per trace.

    Directory layout::

        {directory} / {trace_id}.jsonl

    Each file's first line is a header, subsequent lines are entries.

    Example::

        store = JsonlTraceStore(Path("~/.kagent/traces"))
        store.create("abc123", "my-session", time.time())
        store.append("abc123", entry)
        header, entries = store.load("abc123")
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)

    def _path(self, trace_id: str) -> Path:
        return self._directory / f"{trace_id}.jsonl"

    def create(self, trace_id: str, name: str, created_at: float) -> None:
        path = self._path(trace_id)
        header = {"type": "header", "id": trace_id, "name": name, "created_at": created_at}
        path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")

    def append(self, trace_id: str, entry: TraceEntry) -> None:
        path = self._path(trace_id)
        if not path.exists():
            raise KeyError(f"Trace '{trace_id}' does not exist")
        line = entry.to_json()
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def load(self, trace_id: str) -> tuple[dict[str, Any], list[TraceEntry]]:
        path = self._path(trace_id)
        if not path.exists():
            raise KeyError(f"Trace '{trace_id}' does not exist")

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            raise ValueError(f"Trace file '{path}' is empty")

        header: dict[str, Any] = json.loads(lines[0])
        entries = [TraceEntry.from_json(line) for line in lines[1:]]
        return header, entries

    def list_traces(self) -> list[str]:
        return sorted(p.stem for p in self._directory.glob("*.jsonl"))
