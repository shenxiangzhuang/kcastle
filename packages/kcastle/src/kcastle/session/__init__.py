"""kcastle.session — Session management for kcastle.

Provides ``Session``, ``SessionManager``, ``SessionTraceStore``, and
related types for session lifecycle and persistence.
"""

from kcastle.session.manager import SessionInfo, SessionManager
from kcastle.session.session import Session, SessionMeta
from kcastle.session.store import SessionTraceStore

__all__ = [
    "Session",
    "SessionInfo",
    "SessionManager",
    "SessionMeta",
    "SessionTraceStore",
]
