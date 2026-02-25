"""Trace primitives for kagent.

The trace subpackage provides append-only execution tracing:

- ``TraceEntry`` — immutable record (the single data primitive)
- ``TraceKind`` — entry type discriminator (StrEnum)
- ``TraceMeta`` — structured metadata (timing, run info, usage)
- ``Trace`` — append-only container with identity
- ``TraceManager`` — multi-trace orchestration with optional persistence
- ``TraceStore`` — persistence Protocol
- ``InMemoryTraceStore`` — dict-backed store (testing)
- ``JsonlTraceStore`` — JSONL file-backed store (production)
"""

from kagent.trace.entry import TraceEntry, TraceKind, TraceMeta
from kagent.trace.manager import TraceManager
from kagent.trace.store import InMemoryTraceStore, JsonlTraceStore, TraceStore
from kagent.trace.trace import Trace

__all__ = [
    "InMemoryTraceStore",
    "JsonlTraceStore",
    "Trace",
    "TraceEntry",
    "TraceKind",
    "TraceManager",
    "TraceMeta",
    "TraceStore",
]
