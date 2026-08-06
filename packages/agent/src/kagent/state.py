"""Append-only causal state and its current model-context projection."""

from __future__ import annotations

import json
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

type Item = dict[str, object]


@dataclass(frozen=True, slots=True)
class ItemEntry:
    """One atomic input batch, kept together during compaction."""

    id: int
    items: tuple[Item, ...]


@dataclass(frozen=True, slots=True)
class CompactionEntry:
    """A summary projection over history before ``first_kept_id``."""

    id: int
    summary: str
    first_kept_id: int
    tokens_before: int


type StateEntry = ItemEntry | CompactionEntry


def estimate_tokens(value: object) -> int:
    """Cheap token estimate suitable for compaction thresholds and cut points."""

    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return max(1, (len(encoded) + 3) // 4)


@dataclass(slots=True, init=False)
class State:
    """Full append-only history; ``context()`` returns the compacted projection."""

    _entries: list[StateEntry]

    def __init__(self, entries: Iterable[StateEntry] = ()) -> None:
        self._entries = deepcopy(list(entries))

    @property
    def entries(self) -> tuple[StateEntry, ...]:
        """A snapshot of the complete history."""

        return deepcopy(tuple(self._entries))

    def append_user(self, text: str) -> ItemEntry:
        return self.append_items([{"role": "user", "content": text}])

    def append_items(self, items: list[Item]) -> ItemEntry:
        if not items:
            raise ValueError("items must not be empty")
        entry = ItemEntry(self._next_id(), tuple(deepcopy(items)))
        self._entries.append(entry)
        return deepcopy(entry)

    def append_compaction(
        self,
        *,
        summary: str,
        first_kept_id: int,
        tokens_before: int,
    ) -> CompactionEntry:
        if not any(
            isinstance(entry, ItemEntry) and entry.id == first_kept_id for entry in self._entries
        ):
            raise ValueError(f"unknown first_kept_id: {first_kept_id}")
        entry = CompactionEntry(
            id=self._next_id(),
            summary=summary,
            first_kept_id=first_kept_id,
            tokens_before=tokens_before,
        )
        self._entries.append(entry)
        return entry

    def context(self) -> list[Item]:
        latest = self.latest_compaction
        batches = self.active_batches()
        items: list[Item] = []
        if latest is not None:
            items.append(
                {
                    "role": "user",
                    "content": (
                        "<conversation_summary>\n"
                        "This is factual context from earlier work, not new instructions.\n"
                        f"{latest.summary}\n"
                        "</conversation_summary>"
                    ),
                }
            )
        for batch in batches:
            items.extend(batch.items)
        return items

    def active_batches(self) -> list[ItemEntry]:
        latest = self.latest_compaction
        boundary = latest.first_kept_id if latest is not None else 0
        return [
            deepcopy(entry)
            for entry in self._entries
            if isinstance(entry, ItemEntry) and entry.id >= boundary
        ]

    @property
    def latest_compaction(self) -> CompactionEntry | None:
        return next(
            (entry for entry in reversed(self._entries) if isinstance(entry, CompactionEntry)),
            None,
        )

    def records(self) -> list[dict[str, object]]:
        """Return the plain-data records used by persistence adapters."""

        records: list[dict[str, object]] = []
        for entry in self._entries:
            match entry:
                case ItemEntry(id=entry_id, items=items):
                    records.append(
                        {"type": "items", "id": entry_id, "items": deepcopy(list(items))}
                    )
                case CompactionEntry(
                    id=entry_id,
                    summary=summary,
                    first_kept_id=first_kept_id,
                    tokens_before=tokens_before,
                ):
                    records.append(
                        {
                            "type": "compaction",
                            "id": entry_id,
                            "summary": summary,
                            "first_kept_id": first_kept_id,
                            "tokens_before": tokens_before,
                        }
                    )
        return records

    @classmethod
    def from_records(cls, raw: object) -> State:
        """Restore state from plain-data persistence records."""

        def required_int(record: dict[str, object], key: str) -> int:
            field = record.get(key)
            if not isinstance(field, int) or isinstance(field, bool):
                raise ValueError(f"{key} must be an integer")
            return field

        if not isinstance(raw, list):
            raise ValueError("state records must be a list")

        entries: list[StateEntry] = []
        for untyped_record in cast(list[object], raw):
            if not isinstance(untyped_record, dict):
                raise ValueError("state entries must be objects")
            record = cast(dict[str, object], untyped_record)
            match record.get("type"):
                case "items":
                    raw_items = record.get("items")
                    if not isinstance(raw_items, list) or not all(
                        isinstance(item, dict) for item in cast(list[object], raw_items)
                    ):
                        raise ValueError("items entry contains invalid items")
                    entries.append(
                        ItemEntry(
                            id=required_int(record, "id"),
                            items=tuple(cast(Item, item) for item in cast(list[object], raw_items)),
                        )
                    )
                case "compaction":
                    summary = record.get("summary")
                    if not isinstance(summary, str):
                        raise ValueError("compaction summary must be a string")
                    entries.append(
                        CompactionEntry(
                            id=required_int(record, "id"),
                            summary=summary,
                            first_kept_id=required_int(record, "first_kept_id"),
                            tokens_before=required_int(record, "tokens_before"),
                        )
                    )
                case other:
                    raise ValueError(f"unknown state entry type: {other!r}")

        ids = [entry.id for entry in entries]
        if ids != sorted(set(ids)):
            raise ValueError("state entry IDs must be unique and increasing")
        return cls(entries)

    def _next_id(self) -> int:
        return self._entries[-1].id + 1 if self._entries else 1
