"""Session — wraps an ``AgentRuntime`` with metadata and persistence.

A ``Session`` is the atomic unit of interaction in kcastle.  It binds an
agent runtime to a session directory on disk, manages metadata, and
exposes the ``run()`` / ``suspend()`` lifecycle.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kagent import Agent, AgentEvent, AgentRuntime, AgentState, Trace, TraceManager, UserInput

from kcastle.log import logger
from kcastle.session.store import SessionTraceStore

META_FILENAME = "meta.json"
type AgentFactory = Callable[[], Agent]


@dataclass(slots=True)
class SessionMeta:
    """Mutable session metadata, persisted in ``meta.json``."""

    id: str
    name: str
    created_at: int  # ms timestamp
    created_at_iso: str
    last_active_at: int  # ms timestamp
    last_active_at_iso: str
    provider_name: str | None = None
    model_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SessionMeta:
        return cls(
            id=str(d["id"]),
            name=str(d.get("name", "")),
            created_at=int(d["created_at"]),
            created_at_iso=str(d.get("created_at_iso", "")),
            last_active_at=int(d.get("last_active_at", d["created_at"])),
            last_active_at_iso=str(d.get("last_active_at_iso", d.get("created_at_iso", ""))),
            provider_name=(str(v) if (v := d.get("provider_name")) else None),
            model_id=(str(v) if (v := d.get("model_id")) else None),
        )


def _now_ms() -> int:
    """Current time as milliseconds since epoch."""
    return int(time.time() * 1000)


def _now_iso() -> str:
    """Current time as ISO 8601 string with timezone."""
    return datetime.now(UTC).astimezone().isoformat()


def _save_meta(session_dir: Path, meta: SessionMeta) -> None:
    """Atomically write meta.json to the session directory."""
    path = session_dir / META_FILENAME
    session_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta.to_dict(), indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_meta(session_dir: Path) -> SessionMeta:
    """Read meta.json from a session directory."""
    path = session_dir / META_FILENAME
    data = json.loads(path.read_text(encoding="utf-8"))
    return SessionMeta.from_dict(data)


class Session:
    """A live agent session with on-disk persistence.

    Lifecycle: ``create()`` → ``IDLE`` ↔ ``RUNNING`` → ``suspend()``
    """

    def __init__(
        self,
        *,
        session_dir: Path,
        meta: SessionMeta,
        runtime: AgentRuntime,
        trace: Trace,
        trace_manager: TraceManager,
    ) -> None:
        self._session_dir = session_dir
        self._meta = meta
        self._runtime = runtime
        self._trace = trace
        self._trace_manager = trace_manager
        self._running = False
        self._runtime_started = False

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
        return self._runtime.agent

    @property
    def runtime(self) -> AgentRuntime:
        return self._runtime

    @property
    def trace(self) -> Trace:
        return self._trace

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def model_override(self) -> tuple[str, str] | None:
        if not self._meta.provider_name or not self._meta.model_id:
            return None
        return (self._meta.provider_name, self._meta.model_id)

    def set_model_override(self, provider_name: str, model_id: str) -> None:
        self._meta.provider_name = provider_name
        self._meta.model_id = model_id
        _save_meta(self._session_dir, self._meta)

    async def run(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """Run the agent with user input, streaming events.

        Updates ``last_active_at`` in meta.json on completion.
        """
        if self._running:
            raise RuntimeError(f"Session {self.id} is already running")

        # Start runtime on first run
        if not self._runtime_started:
            await self._runtime.start()
            self._runtime_started = True

        self._running = True
        try:
            async for event in self._runtime.send(UserInput(text=user_input)):
                yield event
        finally:
            self._running = False
            self._touch()

    def abort(self) -> None:
        """Abort the currently running agent handle."""
        self._runtime.abort()

    def suspend(self) -> None:
        """Suspend the session — drop agent from memory.

        Trace is already persisted incrementally; just write final meta.
        """
        # Stop runtime if started
        if self._runtime_started:
            # Cancel the runtime's loop task directly
            if hasattr(self._runtime, '_loop_task') and self._runtime._loop_task:
                self._runtime._loop_task.cancel()
            self._runtime_started = False

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
        agent_factory: AgentFactory,
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

        return cls._assemble(
            session_dir=session_dir,
            meta=meta,
            trace=trace,
            trace_manager=trace_manager,
            agent_factory=agent_factory,
        )

    @classmethod
    def resume(
        cls,
        *,
        session_dir: Path,
        agent_factory: AgentFactory,
    ) -> Session:
        """Resume a session from disk (reload trace + rebuild agent)."""
        meta = _load_meta(session_dir)

        trace_store = SessionTraceStore(session_dir)
        trace_manager = TraceManager(store=trace_store)

        trace_ids = trace_store.list_traces()
        if not trace_ids:
            raise ValueError(f"No trace found in session {meta.id}")
        trace = trace_manager.load(trace_ids[0])

        logger.info("Resumed session %s (%d trace entries)", meta.id, len(trace))
        return cls._assemble(
            session_dir=session_dir,
            meta=meta,
            trace=trace,
            trace_manager=trace_manager,
            agent_factory=agent_factory,
        )

    @classmethod
    def _assemble(
        cls,
        *,
        session_dir: Path,
        meta: SessionMeta,
        trace: Trace,
        trace_manager: TraceManager,
        agent_factory: AgentFactory,
    ) -> Session:
        """Assemble a Session from its components (shared by create and resume)."""
        agent = agent_factory()
        state = AgentState(
            system=agent.system,
            trace=trace,
            tools=list(agent.tools),
        )
        runtime = AgentRuntime(agent, state=state)
        return cls(
            session_dir=session_dir,
            meta=meta,
            runtime=runtime,
            trace=trace,
            trace_manager=trace_manager,
        )
