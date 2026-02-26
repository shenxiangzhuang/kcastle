"""Per-session trace persistence.

``SessionTraceStore`` implements ``kagent.TraceStore`` and writes
``trace.jsonl`` inside the session's own directory, keeping each session
fully self-contained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kagent.trace.entry import TraceEntry

_TRACE_FILENAME = "trace.jsonl"


class SessionTraceStore:
    """Trace store that writes to a session directory.

    File layout::

        <session_dir>/trace.jsonl

    Follows the kagent ``JsonlTraceStore`` format: first line is a
    JSON header, subsequent lines are serialized ``TraceEntry`` objects.

    Implements the ``kagent.TraceStore`` protocol.
    """

    def __init__(self, session_dir: Path) -> None:
        self._session_dir = session_dir
        self._path = session_dir / _TRACE_FILENAME

    @property
    def path(self) -> Path:
        """Path to the trace file."""
        return self._path

    def create(self, trace_id: str, name: str, created_at: float) -> None:
        """Initialize a new trace file with a header."""
        self._session_dir.mkdir(parents=True, exist_ok=True)
        header = {"type": "header", "id": trace_id, "name": name, "created_at": created_at}
        self._path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")

    def append(self, trace_id: str, entry: TraceEntry) -> None:
        """Append a serialized trace entry to the file."""
        if not self._path.exists():
            raise KeyError(f"Trace '{trace_id}' does not exist at {self._path}")
        line = entry.to_json()
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def load(self, trace_id: str) -> tuple[dict[str, Any], list[TraceEntry]]:
        """Load the trace header and entries from ``trace.jsonl``."""
        if not self._path.exists():
            raise KeyError(f"Trace '{trace_id}' does not exist at {self._path}")

        lines = self._path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            raise ValueError(f"Trace file '{self._path}' is empty")

        header: dict[str, Any] = json.loads(lines[0])
        entries = [TraceEntry.from_json(line) for line in lines[1:]]
        return header, entries

    def list_traces(self) -> list[str]:
        """List trace IDs.  A session directory has at most one trace."""
        if self._path.exists():
            lines = self._path.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                header: dict[str, Any] = json.loads(lines[0])
                trace_id = header.get("id", "")
                if trace_id:
                    return [str(trace_id)]
        return []

    def exists(self) -> bool:
        """Whether the trace file exists."""
        return self._path.is_file()
