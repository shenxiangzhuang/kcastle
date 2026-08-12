"""Append-only JSONL persistence for Agent sessions."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Self, cast
from uuid import uuid4

from openai.types.responses import ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails

from kagent.state import (
    CompactionEntry,
    Item,
    ItemEntry,
    ResponseMetadata,
    State,
    StateEntry,
)

_HEADER_LIMIT = 64 * 1024
_INTERRUPTED_TOOL = (
    "Tool execution was interrupted; its side effects are unknown. Do not retry automatically."
)


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
        initialized: bool = False,
    ) -> None:
        self.state = state if state is not None else State()
        self.info = info
        self._initialized = initialized
        self._last_id = len(self.state)
        self._lock = asyncio.Lock()
        self._failed = False

    @classmethod
    def create(cls, directory: Path) -> Self:
        created_at = datetime.now(UTC)
        name = f"{created_at:%Y%m%dT%H%M%S}-{uuid4().hex[:8]}.jsonl"
        return cls(info=SessionInfo(directory / name, "Untitled session", created_at))

    @classmethod
    def open(cls, path: Path) -> Self:
        try:
            info, entries, tail_offset, missing_newline = cls._read(path)
            if tail_offset is not None:
                cls._repair_tail(path, tail_offset)
            elif missing_newline:
                cls._append(path, "\n")
            session = cls(State.restore(entries), info=info, initialized=True)
            session._close_interrupted_tools()
            return session
        except (OSError, UnicodeDecodeError, ValueError) as error:
            raise SessionError(f"Cannot open session {path}: {error}") from error

    @classmethod
    def list(cls, directory: Path) -> tuple[SessionInfo, ...]:
        """List valid persisted sessions newest first using only their headers."""

        infos: list[SessionInfo] = []
        try:
            paths = tuple(directory.glob("*.jsonl"))
        except OSError as error:
            raise SessionError(f"Cannot list sessions in {directory}: {error}") from error
        for path in paths:
            try:
                with path.open("rb") as file:
                    raw = file.readline(_HEADER_LIMIT + 1)
                if len(raw) > _HEADER_LIMIT or not raw:
                    continue
                infos.append(cls._parse_info(path, cast(object, json.loads(raw))))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
                continue
        return tuple(sorted(infos, key=lambda info: info.created_at, reverse=True))

    async def commit(self, entry: StateEntry) -> None:
        """Persist exactly one newly appended State entry."""

        async with self._lock:
            if self._failed:
                raise SessionError(f"Session {self.info.path} is unavailable after a write failure")
            if entry.id != self._last_id + 1 or len(self.state) != entry.id:
                raise SessionError(f"Session entry {entry.id} is not the next State entry")

            title = self._initial_title(entry) if not self._initialized else self.info.title
            records = [self._entry_record(entry)]
            if not self._initialized:
                records.insert(0, self._info_record(title))
            value = "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                for record in records
            )
            write = asyncio.create_task(
                asyncio.to_thread(
                    self._write_new if not self._initialized else self._append,
                    self.info.path,
                    value,
                )
            )
            cancelled = False
            try:
                try:
                    await asyncio.shield(write)
                except asyncio.CancelledError:
                    cancelled = True
                    await write
            except OSError as error:
                self._failed = True
                raise SessionError(f"Cannot commit session {self.info.path}: {error}") from error
            self.info = replace(self.info, title=title)
            self._initialized = True
            self._last_id = entry.id
            if cancelled:
                raise asyncio.CancelledError

    def _close_interrupted_tools(self) -> None:
        call_ids = self.state.unresolved_tool_call_ids()
        if not call_ids:
            return
        entry = self.state.append_items(
            [
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _INTERRUPTED_TOOL,
                }
                for call_id in call_ids
            ]
        )
        try:
            self._append(
                self.info.path,
                json.dumps(
                    self._entry_record(entry),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n",
            )
        except OSError:
            self.state.rollback(entry.id)
            raise
        self._last_id = entry.id

    @classmethod
    def _read(cls, path: Path) -> tuple[SessionInfo, Sequence[StateEntry], int | None, bool]:
        entries: list[StateEntry] = []
        with path.open("rb") as file:
            header = file.readline(_HEADER_LIMIT + 1)
            if not header:
                raise ValueError("session is empty")
            if len(header) > _HEADER_LIMIT:
                raise ValueError("session header is too large")
            info = cls._parse_info(path, cast(object, json.loads(header)))
            offset = len(header)
            raw = file.readline()
            line_number = 2
            tail_offset: int | None = None
            missing_newline = not header.endswith(b"\n")

            while raw:
                next_raw = file.readline()
                is_last = not next_raw
                try:
                    record = cast(object, json.loads(raw))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if not is_last:
                        raise ValueError(f"invalid session record at line {line_number}") from None
                    tail_offset = offset
                    break
                try:
                    entries.append(cls._parse_entry(record))
                except ValueError as error:
                    raise ValueError(
                        f"invalid session record at line {line_number}: {error}"
                    ) from None
                missing_newline = is_last and not raw.endswith(b"\n")
                offset += len(raw)
                raw = next_raw
                line_number += 1

        return info, entries, tail_offset, missing_newline

    @staticmethod
    def _parse_entry(record: object) -> StateEntry:
        if not isinstance(record, dict):
            raise ValueError("session entries must be objects")

        def required_int(key: str) -> int:
            value = record.get(key)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{key} must be an integer")
            return value

        def optional_response() -> ResponseMetadata | None:
            raw = record.get("response")
            if raw is None:
                return None
            if not isinstance(raw, dict):
                raise ValueError("response metadata must be an object")
            response_id = raw.get("id")
            model = raw.get("model")
            if not isinstance(response_id, str) or not isinstance(model, str):
                raise ValueError("response metadata is invalid")
            usage = raw.get("usage")
            if isinstance(usage, dict):
                details = usage.get("input_tokens_details")
                if isinstance(details, dict) and details.get("cache_write_tokens") is None:
                    cached_tokens = details.get("cached_tokens")
                    if not isinstance(cached_tokens, int) or isinstance(cached_tokens, bool):
                        raise ValueError("cached_tokens must be an integer")
                    usage = dict(usage)
                    usage["input_tokens_details"] = InputTokensDetails.model_construct(
                        cached_tokens=cached_tokens
                    )
            return ResponseMetadata(
                id=response_id,
                model=model,
                usage=None if usage is None else ResponseUsage.model_validate(usage),
            )

        match record.get("type"):
            case "items":
                raw_items = record.get("items")
                if not isinstance(raw_items, list) or not all(
                    isinstance(item, dict) for item in raw_items
                ):
                    raise ValueError("items entry contains invalid items")
                return ItemEntry(
                    id=required_int("id"),
                    items=tuple(cast(Item, item) for item in raw_items),
                    response=optional_response(),
                )
            case "compaction":
                summary = record.get("summary")
                if not isinstance(summary, str):
                    raise ValueError("compaction summary must be a string")
                return CompactionEntry(
                    id=required_int("id"),
                    summary=summary,
                    first_kept_id=required_int("first_kept_id"),
                    tokens_before=required_int("tokens_before"),
                    response=optional_response(),
                )
            case other:
                raise ValueError(f"unknown state entry type: {other!r}")

    @staticmethod
    def _entry_record(entry: StateEntry) -> dict[str, object]:
        def response_record(response: ResponseMetadata) -> dict[str, object]:
            return {
                "id": response.id,
                "model": response.model,
                "usage": (
                    None if response.usage is None else response.usage.model_dump(mode="json")
                ),
            }

        match entry:
            case ItemEntry(id=entry_id, items=items, response=response):
                record: dict[str, object] = {
                    "type": "items",
                    "id": entry_id,
                    "items": list(items),
                }
                if response is not None:
                    record["response"] = response_record(response)
                return record
            case CompactionEntry(
                id=entry_id,
                summary=summary,
                first_kept_id=first_kept_id,
                tokens_before=tokens_before,
                response=response,
            ):
                record = {
                    "type": "compaction",
                    "id": entry_id,
                    "summary": summary,
                    "first_kept_id": first_kept_id,
                    "tokens_before": tokens_before,
                }
                if response is not None:
                    record["response"] = response_record(response)
                return record

    def _info_record(self, title: str) -> dict[str, object]:
        return {
            "type": "session",
            "version": 1,
            "title": title,
            "created_at": self.info.created_at.isoformat(),
        }

    @staticmethod
    def _initial_title(entry: StateEntry) -> str:
        if isinstance(entry, ItemEntry):
            for item in entry.items:
                content = item.get("content")
                if item.get("role") == "user" and isinstance(content, str):
                    return " ".join(content.split())[:80] or "Untitled session"
        return "Untitled session"

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
    def _write_new(path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as file:
                file.write(value)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _append(path: Path, value: str) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(value)

    @staticmethod
    def _repair_tail(path: Path, offset: int) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with path.open("rb") as source, temporary.open("xb") as target:
                remaining = offset
                while remaining:
                    chunk = source.read(min(64 * 1024, remaining))
                    if not chunk:
                        raise OSError("session ended before the repair boundary")
                    target.write(chunk)
                    remaining -= len(chunk)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
