"""Append-only JSONL persistence for Agent sessions."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

from kagent.state import ItemEntry, State


class SessionError(Exception):
    """Raised when a persistent session cannot be loaded or committed."""


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """Metadata used to identify a persisted session."""

    path: Path
    title: str
    created_at: datetime


class Session:
    """One Agent state persisted as an append-only JSONL stream."""

    def __init__(
        self,
        state: State | None = None,
        *,
        info: SessionInfo,
        persisted_entries: int = 0,
    ) -> None:
        self.state = state if state is not None else State()
        self.info = info
        self._persisted_entries = persisted_entries
        self._lock = asyncio.Lock()

    @classmethod
    def create(cls, directory: Path) -> Session:
        created_at = datetime.now(UTC)
        name = f"{created_at:%Y%m%dT%H%M%S}-{uuid4().hex[:8]}.jsonl"
        return cls(info=SessionInfo(directory / name, "Untitled session", created_at))

    @classmethod
    def open(cls, path: Path) -> Session:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            if not lines:
                raise ValueError("session is empty")
            info = cls._parse_info(path, cast(object, json.loads(lines[0])))
            records = [cast(object, json.loads(line)) for line in lines[1:]]
            state = State.from_records(records)
        except (json.JSONDecodeError, OSError, ValueError) as error:
            raise SessionError(f"Cannot open session {path}: {error}") from error
        return cls(state, info=info, persisted_entries=len(records))

    @classmethod
    def list(cls, directory: Path) -> tuple[SessionInfo, ...]:
        """List persisted sessions newest first."""

        try:
            infos = [
                cls._parse_info(
                    path,
                    cast(object, json.loads(path.read_text(encoding="utf-8").splitlines()[0])),
                )
                for path in directory.glob("*.jsonl")
            ]
        except (IndexError, json.JSONDecodeError, OSError, ValueError) as error:
            raise SessionError(f"Cannot list sessions in {directory}: {error}") from error
        return tuple(sorted(infos, key=lambda info: info.created_at, reverse=True))

    async def commit(self) -> None:
        path = self.info.path
        async with self._lock:
            records = self.state.records()
            if len(records) < self._persisted_entries:
                raise SessionError("Cannot commit state whose persisted history was removed")
            pending = records[self._persisted_entries :]
            if not pending:
                return
            if self._persisted_entries == 0:
                self.info = replace(self.info, title=self._initial_title())
                pending = [self._info_record(), *pending]
            value = "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                for record in pending
            )
            try:
                await asyncio.to_thread(self._append, path, value)
            except OSError as error:
                raise SessionError(f"Cannot commit session {path}: {error}") from error
            self._persisted_entries = len(records)

    def _initial_title(self) -> str:
        for entry in self.state.entries:
            if not isinstance(entry, ItemEntry):
                continue
            for item in entry.items:
                content = item.get("content")
                if item.get("role") == "user" and isinstance(content, str):
                    return " ".join(content.split())[:80] or "Untitled session"
        return "Untitled session"

    def _info_record(self) -> dict[str, object]:
        return {
            "type": "session",
            "version": 1,
            "title": self.info.title,
            "created_at": self.info.created_at.isoformat(),
        }

    @staticmethod
    def _parse_info(path: Path, record: object) -> SessionInfo:
        if not isinstance(record, dict) or record.get("type") != "session":
            raise ValueError("first record must be session metadata")
        title = record.get("title")
        created = record.get("created_at")
        if record.get("version") != 1 or not isinstance(title, str) or not title.strip():
            raise ValueError("invalid session metadata")
        if not isinstance(created, str):
            raise ValueError("invalid session creation time")
        created_at = datetime.fromisoformat(created)
        if created_at.tzinfo is None:
            raise ValueError("session creation time must include a timezone")
        return SessionInfo(path, title, created_at)

    @staticmethod
    def _append(path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(value)
