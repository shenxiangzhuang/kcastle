"""Append-only causal state and its current model-context projection."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass

from openai.types.responses import ResponseUsage

type Item = dict[str, object]


@dataclass(frozen=True, slots=True)
class ResponseMetadata:
    """Responses API metadata attached to one assistant output batch."""

    id: str
    model: str
    usage: ResponseUsage | None


@dataclass(frozen=True, slots=True)
class ItemEntry:
    """One atomic input batch, kept together during compaction."""

    id: int
    items: tuple[Item, ...]
    response: ResponseMetadata | None = None


@dataclass(frozen=True, slots=True)
class CompactionEntry:
    """A summary projection over history before ``first_kept_id``."""

    id: int
    summary: str
    first_kept_id: int
    tokens_before: int
    response: ResponseMetadata | None = None


type StateEntry = ItemEntry | CompactionEntry


def estimate_tokens(value: object) -> int:
    """Cheap token estimate suitable for compaction thresholds and cut points."""

    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return max(1, (len(encoded) + 3) // 4)


@dataclass(slots=True, init=False)
class State:
    """Full append-only history; ``context()`` returns the compacted projection."""

    _entries: list[StateEntry]
    _active_start: int
    _item_indexes: dict[int, int]
    _latest_compaction: CompactionEntry | None
    _latest_response: ResponseMetadata | None

    def __init__(self, entries: Iterable[StateEntry] = ()) -> None:
        self._entries = deepcopy(list(entries))
        self._validate_history()
        self._reindex()

    @classmethod
    def restore(cls, entries: Sequence[StateEntry]) -> State:
        """Restore freshly decoded entries without copying their payloads again."""

        state = cls()
        state._entries = list(entries)
        state._validate_history()
        state._reindex()
        return state

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[StateEntry, ...]:
        """A snapshot of the complete history."""

        return deepcopy(tuple(self._entries))

    def append_user(self, text: str) -> ItemEntry:
        return self.append_items([{"role": "user", "content": text}])

    def append_items(
        self,
        items: list[Item],
        *,
        response: ResponseMetadata | None = None,
    ) -> ItemEntry:
        if not items:
            raise ValueError("items must not be empty")
        entry = ItemEntry(self._next_id(), tuple(deepcopy(items)), deepcopy(response))
        self._item_indexes[entry.id] = len(self._entries)
        self._entries.append(entry)
        if entry.response is not None:
            self._latest_response = entry.response
        return deepcopy(entry)

    def append_compaction(
        self,
        *,
        summary: str,
        first_kept_id: int,
        tokens_before: int,
        response: ResponseMetadata | None = None,
    ) -> CompactionEntry:
        active_start = self._item_indexes.get(first_kept_id)
        if active_start is None:
            raise ValueError(f"unknown first_kept_id: {first_kept_id}")
        entry = CompactionEntry(
            id=self._next_id(),
            summary=summary,
            first_kept_id=first_kept_id,
            tokens_before=tokens_before,
            response=deepcopy(response),
        )
        self._entries.append(entry)
        self._latest_compaction = entry
        self._active_start = active_start
        return entry

    def rollback(self, entry_id: int) -> None:
        """Remove only an uncommitted tail entry after persistence fails."""

        if not self._entries or self._entries[-1].id != entry_id:
            raise ValueError(f"cannot roll back non-tail entry: {entry_id}")
        self._entries.pop()
        self._reindex()

    def context(self) -> list[Item]:
        latest = self._latest_compaction
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
        for entry in self._entries[self._active_start :]:
            if isinstance(entry, ItemEntry):
                items.extend(deepcopy(entry.items))
        return items

    def active_batches(self) -> list[ItemEntry]:
        return [
            deepcopy(entry)
            for entry in self._entries[self._active_start :]
            if isinstance(entry, ItemEntry)
        ]

    def active_items(self) -> Iterator[Item]:
        """Yield defensive item snapshots from the current context suffix."""

        for entry in self._entries[self._active_start :]:
            if isinstance(entry, ItemEntry):
                yield from deepcopy(entry.items)

    def items(self) -> Iterator[Item]:
        """Yield defensive item snapshots without copying the complete history at once."""

        for entry in self._entries:
            if isinstance(entry, ItemEntry):
                yield from deepcopy(entry.items)

    def unresolved_tool_call_ids(self) -> tuple[str, ...]:
        """Return active function calls that have no corresponding output."""

        pending: dict[str, None] = {}
        for entry in self._entries[self._active_start :]:
            if not isinstance(entry, ItemEntry):
                continue
            for item in entry.items:
                call_id = item.get("call_id")
                if not isinstance(call_id, str):
                    continue
                if item.get("type") == "function_call":
                    pending[call_id] = None
                elif item.get("type") == "function_call_output":
                    pending.pop(call_id, None)
        return tuple(pending)

    @property
    def latest_compaction(self) -> CompactionEntry | None:
        return self._latest_compaction

    @property
    def latest_response(self) -> ResponseMetadata | None:
        return deepcopy(self._latest_response)

    def _validate_history(self) -> None:
        if [entry.id for entry in self._entries] != list(range(1, len(self._entries) + 1)):
            raise ValueError("state entry IDs must be consecutive from 1")
        seen_items: set[int] = set()
        for entry in self._entries:
            if isinstance(entry, ItemEntry):
                seen_items.add(entry.id)
            elif entry.first_kept_id not in seen_items:
                raise ValueError(f"unknown first_kept_id: {entry.first_kept_id}")

    def _reindex(self) -> None:
        self._active_start = 0
        self._item_indexes = {}
        self._latest_compaction = None
        self._latest_response = None
        for index, entry in enumerate(self._entries):
            if isinstance(entry, ItemEntry):
                self._item_indexes[entry.id] = index
                if entry.response is not None:
                    self._latest_response = entry.response
            else:
                self._latest_compaction = entry
                self._active_start = self._item_indexes[entry.first_kept_id]

    def _next_id(self) -> int:
        return self._entries[-1].id + 1 if self._entries else 1
