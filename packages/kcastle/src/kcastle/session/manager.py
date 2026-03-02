"""Session manager — session lifecycle CRUD.

``SessionManager`` is the central API channels use to create, resume,
list, and drop sessions.  It uses the filesystem as the source of truth —
no separate registry file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from kcastle.log import logger
from kcastle.session.session import META_FILENAME, Session, SessionMeta

type AgentFactory = Any  # Callable[[Trace], Agent]


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """Lightweight session info for listing (no agent in memory)."""

    id: str
    name: str
    created_at: int
    last_active_at: int


class SessionManager:
    """Manages session lifecycle.  Filesystem is the registry.

    Channels interact with sessions exclusively through this manager:
    - ``create()`` — new session
    - ``get_or_create()`` — resume if exists, create if not
    - ``resume()`` — load from disk
    - ``get()`` — from memory cache only
    - ``suspend()`` — drop from memory
    - ``list()`` — scan filesystem
    - ``latest()`` — most recently active session
    """

    def __init__(
        self,
        *,
        sessions_dir: Path,
        agent_factory: AgentFactory,
    ) -> None:
        self._sessions_dir = sessions_dir
        self._agent_factory = agent_factory
        self._sessions: dict[str, Session] = {}

        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    @property
    def sessions_dir(self) -> Path:
        return self._sessions_dir

    def create(
        self,
        session_id: str | None = None,
        name: str = "",
    ) -> Session:
        """Create a new session.  Auto-generates ID if not provided."""
        sid = session_id or uuid4().hex[:8]
        session_dir = self._sessions_dir / sid

        if session_dir.exists():
            raise ValueError(f"Session '{sid}' already exists")

        session = Session.create(
            session_dir=session_dir,
            session_id=sid,
            name=name,
            agent_factory=self._agent_factory,
        )
        self._sessions[sid] = session
        logger.info("Created session %s", sid)
        return session

    def get_or_create(self, session_id: str, name: str = "") -> Session:
        """Resume if exists, create if not.  Primary API for channels."""
        if session_id in self._sessions:
            return self._sessions[session_id]

        session_dir = self._sessions_dir / session_id
        if session_dir.is_dir() and (session_dir / META_FILENAME).is_file():
            return self.resume(session_id)

        return self.create(session_id=session_id, name=name)

    def resume(self, session_id: str) -> Session:
        """Resume a session from disk."""
        if session_id in self._sessions:
            return self._sessions[session_id]

        session_dir = self._sessions_dir / session_id
        if not (session_dir / META_FILENAME).is_file():
            raise KeyError(f"Session '{session_id}' not found on disk")

        session = Session.resume(
            session_dir=session_dir,
            agent_factory=self._agent_factory,
        )
        self._sessions[session_id] = session
        logger.info("Resumed session %s", session_id)
        return session

    def get(self, session_id: str) -> Session | None:
        """Get a session from memory cache.  Returns None if not loaded."""
        return self._sessions.get(session_id)

    def suspend(self, session_id: str) -> None:
        """Suspend a session — drop from memory (trace already persisted)."""
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.suspend()
            logger.info("Suspended session %s", session_id)

    def suspend_all(self) -> None:
        """Suspend all in-memory sessions."""
        for sid in list(self._sessions.keys()):
            self.suspend(sid)

    def list(self) -> list[SessionInfo]:
        """Scan the sessions directory and return session info.

        Reads ``meta.json`` from each session directory.
        """
        results: list[SessionInfo] = []
        if not self._sessions_dir.is_dir():
            return results

        for child in sorted(self._sessions_dir.iterdir()):
            meta_path = child / META_FILENAME
            if not child.is_dir() or not meta_path.is_file():
                continue
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                meta = SessionMeta.from_dict(data)
                results.append(
                    SessionInfo(
                        id=meta.id,
                        name=meta.name,
                        created_at=meta.created_at,
                        last_active_at=meta.last_active_at,
                    )
                )
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
                logger.warning("Skipping invalid session directory: %s", child)
                logger.debug("Invalid session metadata at %s", meta_path, exc_info=True)

        results.sort(key=lambda s: s.last_active_at, reverse=True)
        return results

    def latest(self) -> Session | None:
        """Resume the most recently active session.  Returns None if empty."""
        sessions = self.list()
        if not sessions:
            return None
        return self.get_or_create(sessions[0].id)
