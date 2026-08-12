from __future__ import annotations

import pytest
from kagent import CompactionEntry, ItemEntry, ResponseMetadata, State
from openai.types.responses import ResponseUsage


def test_state_round_trip_and_compacted_context() -> None:
    state = State()
    old = state.append_items([{"role": "user", "content": "old"}])
    kept = state.append_items([{"role": "user", "content": "kept"}])
    state.append_compaction(summary="what happened", first_kept_id=kept.id, tokens_before=42)
    state.append_items([{"role": "user", "content": "new"}])

    restored = State(state.entries)

    assert restored.entries == state.entries
    assert old.items[0] not in restored.context()
    assert kept.items[0] in restored.context()
    assert restored.context()[-1]["content"] == "new"
    assert "what happened" in str(restored.context()[0]["content"])
    assert list(restored.active_items()) == [kept.items[0], {"role": "user", "content": "new"}]


def test_state_restore_rejects_nonconsecutive_ids() -> None:
    with pytest.raises(ValueError):
        State.restore([ItemEntry(2, ({"role": "user", "content": "bad"},))])


def test_state_restore_rejects_unknown_compaction_boundary() -> None:
    with pytest.raises(ValueError):
        State.restore([CompactionEntry(1, "bad", 99, 1)])


def test_state_items_are_incremental_defensive_snapshots() -> None:
    state = State()
    state.append_items([{"role": "user", "content": "你好"}])

    item = next(state.items())
    item["content"] = "changed"

    assert next(state.items())["content"] == "你好"


def test_state_history_is_append_only() -> None:
    source: dict[str, object] = {"role": "user", "content": "original"}
    state = State()
    appended = state.append_items([source])

    source["content"] = "changed at source"
    appended.items[0]["content"] = "changed through return value"
    snapshot = state.entries[0]
    assert isinstance(snapshot, ItemEntry)
    snapshot.items[0]["content"] = "changed through entries"
    context = state.context()
    context[0]["content"] = "changed through context"

    assert state.context()[0]["content"] == "original"


def test_compaction_state_stays_defensive_and_monotonic() -> None:
    usage = ResponseUsage(
        input_tokens=1,
        input_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0},
        output_tokens=1,
        output_tokens_details={"reasoning_tokens": 0},
        total_tokens=2,
    )
    state = State()
    old = state.append_user("old")
    kept = state.append_user("kept")
    compaction = state.append_compaction(
        summary="summary",
        first_kept_id=kept.id,
        tokens_before=2,
        response=ResponseMetadata("response", "test", usage),
    )

    assert compaction.response is not None and compaction.response.usage is not None
    compaction.response.usage.total_tokens = 0
    latest = state.latest_compaction
    assert latest is not None and latest.response is not None and latest.response.usage is not None
    assert latest.response.usage.total_tokens == 2

    latest.response.usage.total_tokens = 0
    latest = state.latest_compaction
    assert latest is not None and latest.response is not None and latest.response.usage is not None
    assert latest.response.usage.total_tokens == 2
    with pytest.raises(ValueError, match="active history"):
        state.append_compaction(summary="invalid", first_kept_id=old.id, tokens_before=2)
    with pytest.raises(ValueError, match="must not restore"):
        State.restore([*state.entries, CompactionEntry(len(state) + 1, "invalid", old.id, 2)])
