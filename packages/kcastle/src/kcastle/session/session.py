"""Session — wraps a ``kagent.Agent`` with metadata and persistence.

A ``Session`` is the atomic unit of interaction in kcastle.  It binds an
agent instance to a session directory on disk, manages metadata, and
exposes the ``run()`` / ``suspend()`` lifecycle.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kagent import Agent, AgentEvent, Trace, TraceManager

from kcastle.log import logger
from kcastle.session.store import SessionTraceStore

_META_FILENAME = "meta.json"


@dataclass(slots=True)
class SessionMeta:
    """Mutable session metadata, persisted in ``meta.json``."""

    id: str
    name: str
    created_at: int  # ms timestamp
    created_at_iso: str
    last_active_at: int  # ms timestamp
    last_active_at_iso: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "created_at_iso": self.created_at_iso,
            "last_active_at": self.last_active_at,
            "last_active_at_iso": self.last_active_at_iso,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SessionMeta:
        return cls(
            id=str(d["id"]),
            name=str(d.get("name", "")),
            created_at=int(d["created_at"]),
            created_at_iso=str(d.get("created_at_iso", "")),
            last_active_at=int(d.get("last_active_at", d["created_at"])),
            last_active_at_iso=str(d.get("last_active_at_iso", d.get("created_at_iso", ""))),
        )


def _now_ms() -> int:
    """Current time as milliseconds since epoch."""
    return int(time.time() * 1000)


def _now_iso() -> str:
    """Current time as ISO 8601 string with timezone."""
    return datetime.now(UTC).astimezone().isoformat()


def _save_meta(session_dir: Path, meta: SessionMeta) -> None:
    """Atomically write meta.json to the session directory."""
    path = session_dir / _META_FILENAME
    session_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta.to_dict(), indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_meta(session_dir: Path) -> SessionMeta:
    """Read meta.json from a session directory."""
    path = session_dir / _META_FILENAME
    data = json.loads(path.read_text(encoding="utf-8"))
    return SessionMeta.from_dict(data)


type AgentFactory = Any  # Callable[[Trace], Agent] — use Any to avoid Protocol overhead


class Session:
    """A live agent session with on-disk persistence.

    Lifecycle: ``create()`` → ``IDLE`` ↔ ``RUNNING`` → ``suspend()``
    """

    def __init__(
        self,
        *,
        session_dir: Path,
        meta: SessionMeta,
        agent: Agent,
        trace: Trace,
        trace_manager: TraceManager,
    ) -> None:
        self._session_dir = session_dir
        self._meta = meta
        self._agent = agent
        self._trace = trace
        self._trace_manager = trace_manager
        self._running = False

    @property
    def id(self) -> str:
        return self._meta.id

    @property
    def name(self) -> str:
        return self._meta.name

    @name.setter
    def name(self, value: str) -> None:
        self._meta.name = value
        _save_meta(self._session_dir, self._meta)

    @property
    def meta(self) -> SessionMeta:
        return self._meta

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def trace(self) -> Trace:
        return self._trace

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def session_dir(self) -> Path:
        return self._session_dir

    async def run(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """Run the agent with user input, streaming events.

        Updates ``last_active_at`` in meta.json on completion.
        """
        if self._running:
            raise RuntimeError(f"Session {self.id} is already running")

        self._running = True
        try:
            async for event in self._agent.run(user_input):
                yield event
        finally:
            self._running = False
            self._touch()

    async def complete(self, user_input: str) -> str:
        """Run the agent and return the final text response."""
        from kai import Message

        last_text = ""
        async for event in self.run(user_input):
            from kagent import TurnEnd

            if isinstance(event, TurnEnd):
                msg: Message = event.message
                last_text = msg.extract_text()
        return last_text

    def steer(self, message: str) -> None:
        """Inject a steering message into the running agent."""
        from kai import Message

        self._agent.steer(Message(role="user", content=message))

    def abort(self) -> None:
        """Abort the currently running agent."""
        self._agent.abort()

    def suspend(self) -> None:
        """Suspend the session — drop agent from memory.

        Trace is already persisted incrementally; just write final meta.
        """
        self._touch()
        logger.info("Session %s suspended", self.id)

    def _touch(self) -> None:
        """Update last_active_at in meta.json."""
        self._meta.last_active_at = _now_ms()
        self._meta.last_active_at_iso = _now_iso()
        _save_meta(self._session_dir, self._meta)

    @classmethod
    def create(
        cls,
        *,
        session_dir: Path,
        session_id: str,
        name: str,
        agent_factory: Any,  # Callable[[Trace], Agent]
    ) -> Session:
        """Create a brand-new session with a fresh trace."""
        session_dir.mkdir(parents=True, exist_ok=True)

        now = _now_ms()
        now_iso = _now_iso()
        meta = SessionMeta(
            id=session_id,
            name=name,
            created_at=now,
            created_at_iso=now_iso,
            last_active_at=now,
            last_active_at_iso=now_iso,
        )
        _save_meta(session_dir, meta)

        trace_store = SessionTraceStore(session_dir)
        trace_manager = TraceManager(store=trace_store)
        trace = trace_manager.create(name=name)

        agent: Agent = agent_factory(trace)

        logger.info("Created session %s", session_id)
        return cls(
            session_dir=session_dir,
            meta=meta,
            agent=agent,
            trace=trace,
            trace_manager=trace_manager,
        )

    @classmethod
    def resume(
        cls,
        *,
        session_dir: Path,
        agent_factory: Any,  # Callable[[Trace], Agent]
    ) -> Session:
        """Resume a session from disk (reload trace + rebuild agent)."""
        meta = _load_meta(session_dir)

        trace_store = SessionTraceStore(session_dir)
        trace_manager = TraceManager(store=trace_store)

        trace_ids = trace_store.list_traces()
        if not trace_ids:
            raise ValueError(f"No trace found in session {meta.id}")
        trace = trace_manager.load(trace_ids[0])

        agent: Agent = agent_factory(trace)

        logger.info("Resumed session %s (%d trace entries)", meta.id, len(trace))
        return cls(
            session_dir=session_dir,
            meta=meta,
            agent=agent,
            trace=trace,
            trace_manager=trace_manager,
        )
