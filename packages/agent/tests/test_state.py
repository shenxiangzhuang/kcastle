from typing import cast

import pytest
from kagent import ItemEntry, State


def test_state_round_trip_and_compacted_context() -> None:
    state = State()
    old = state.append_items([{"role": "user", "content": "old"}])
    kept = state.append_items([{"role": "user", "content": "kept"}])
    state.append_compaction(summary="what happened", first_kept_id=kept.id, tokens_before=42)
    state.append_items([{"role": "user", "content": "new"}])

    restored = State.from_records(state.records())

    assert restored.entries == state.entries
    assert old.items[0] not in restored.context()
    assert kept.items[0] in restored.context()
    assert restored.context()[-1]["content"] == "new"
    assert "what happened" in str(restored.context()[0]["content"])


@pytest.mark.parametrize(
    "value",
    [{}, [{"type": "unknown", "id": 1}], [{"type": "items", "id": "1"}]],
)
def test_state_rejects_invalid_records(value: object) -> None:
    with pytest.raises(ValueError):
        State.from_records(value)


def test_state_records_are_plain_data() -> None:
    state = State()
    state.append_items([{"role": "user", "content": "你好"}])

    items = cast(list[dict[str, object]], state.records()[0]["items"])
    assert items[0]["content"] == "你好"


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
