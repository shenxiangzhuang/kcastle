import json
from pathlib import Path

import pytest
from kagent import CompactionEntry, ResponseMetadata, Session, SessionError
from openai.types.responses import ResponseUsage


async def test_session_is_a_titled_append_only_jsonl_stream(tmp_path: Path) -> None:
    session = Session.create(tmp_path)
    path = session.info.path
    assert path.suffix == ".jsonl"
    assert not path.exists()

    await session.commit(session.state.append_user("hello"))
    first_commit = path.read_text()
    lines = first_commit.splitlines()

    assert json.loads(lines[0])["title"] == "hello"
    assert json.loads(lines[0])["created_at"]
    assert json.loads(lines[1])["type"] == "items"

    await session.commit(session.state.append_user("again"))
    restored = Session.open(path)

    assert path.read_text().startswith(first_commit)
    assert len(path.read_text().splitlines()) == 3
    assert restored.info.title == "hello"
    assert [item["content"] for item in restored.state.context()] == ["hello", "again"]
    assert restored.state.latest_response is None
    assert Session.list(tmp_path)[0].title == "hello"


async def test_response_usage_round_trips_with_its_item_entry(tmp_path: Path) -> None:
    usage = ResponseUsage(
        input_tokens=120,
        input_tokens_details={"cached_tokens": 80, "cache_write_tokens": 0},
        output_tokens=30,
        output_tokens_details={"reasoning_tokens": 10},
        total_tokens=150,
    )
    session = Session.create(tmp_path)
    entry = session.state.append_items(
        [{"role": "assistant", "content": []}],
        response=ResponseMetadata(id="resp-1", model="test", usage=usage),
    )

    await session.commit(entry)
    kept = session.state.append_user("keep")
    await session.commit(kept)
    compaction = session.state.append_compaction(
        summary="summary",
        first_kept_id=kept.id,
        tokens_before=120,
        response=ResponseMetadata(id="compact-1", model="test", usage=usage),
    )
    await session.commit(compaction)
    restored = Session.open(session.info.path)

    record = json.loads(session.info.path.read_text().splitlines()[1])
    assert record["response"]["usage"] == usage.model_dump(mode="json")
    assert restored.state.latest_response == entry.response
    restored_compaction = restored.state.entries[-1]
    assert isinstance(restored_compaction, CompactionEntry)
    assert restored_compaction.response == compaction.response


async def test_open_keeps_missing_cache_write_tokens_unknown(tmp_path: Path) -> None:
    usage = ResponseUsage(
        input_tokens=120,
        input_tokens_details={"cached_tokens": 80, "cache_write_tokens": 0},
        output_tokens=30,
        output_tokens_details={"reasoning_tokens": 10},
        total_tokens=150,
    )
    session = Session.create(tmp_path)
    await session.commit(
        session.state.append_items(
            [{"role": "assistant", "content": []}],
            response=ResponseMetadata(id="response", model="test", usage=usage),
        )
    )
    records = [json.loads(line) for line in session.info.path.read_text().splitlines()]
    del records[1]["response"]["usage"]["input_tokens_details"]["cache_write_tokens"]
    session.info.path.write_text("".join(json.dumps(record) + "\n" for record in records))

    restored = Session.open(session.info.path)

    response = restored.state.latest_response
    assert response is not None and response.usage is not None
    assert response.usage.input_tokens_details.cached_tokens == 80
    assert response.usage.input_tokens_details.cache_write_tokens is None
    assert "cache_write_tokens" not in response.usage.input_tokens_details.model_dump(
        exclude_none=True
    )


async def test_each_created_session_gets_its_own_file(tmp_path: Path) -> None:
    first = Session.create(tmp_path)
    second = Session.create(tmp_path)
    await first.commit(first.state.append_user("first session"))
    await second.commit(second.state.append_user("second session"))

    assert first.info.path != second.info.path
    assert {info.title for info in Session.list(tmp_path)} == {
        "first session",
        "second session",
    }


async def test_open_repairs_only_a_malformed_final_line(tmp_path: Path) -> None:
    session = Session.create(tmp_path)
    await session.commit(session.state.append_user("hello"))
    valid = session.info.path.read_bytes()
    session.info.path.write_bytes(valid + b'{"type":"items"')

    restored = Session.open(session.info.path)

    assert session.info.path.read_bytes() == valid
    assert restored.state.context()[0]["content"] == "hello"


async def test_open_rejects_a_malformed_middle_line_without_rewriting(tmp_path: Path) -> None:
    session = Session.create(tmp_path)
    await session.commit(session.state.append_user("hello"))
    await session.commit(session.state.append_user("again"))
    lines = session.info.path.read_bytes().splitlines(keepends=True)
    damaged = b"".join([lines[0], lines[1], b"not-json\n", lines[2]])
    session.info.path.write_bytes(damaged)

    with pytest.raises(SessionError, match="line 3"):
        Session.open(session.info.path)

    assert session.info.path.read_bytes() == damaged


async def test_open_rejects_a_semantically_invalid_final_line(tmp_path: Path) -> None:
    session = Session.create(tmp_path)
    await session.commit(session.state.append_user("hello"))
    with session.info.path.open("ab") as file:
        file.write(b'{"type":"items","id":2}\n')
    damaged = session.info.path.read_bytes()

    with pytest.raises(SessionError, match="line 3"):
        Session.open(session.info.path)

    assert session.info.path.read_bytes() == damaged


async def test_open_repairs_a_missing_final_newline(tmp_path: Path) -> None:
    session = Session.create(tmp_path)
    await session.commit(session.state.append_user("hello"))
    session.info.path.write_bytes(session.info.path.read_bytes().rstrip(b"\n"))

    Session.open(session.info.path)

    assert session.info.path.read_bytes().endswith(b"\n")


async def test_open_closes_unresolved_tool_calls_without_replaying(tmp_path: Path) -> None:
    session = Session.create(tmp_path)
    await session.commit(session.state.append_user("run it"))
    await session.commit(
        session.state.append_items(
            [{"type": "function_call", "call_id": "call-1", "name": "shell"}]
        )
    )

    restored = Session.open(session.info.path)

    assert restored.state.unresolved_tool_call_ids() == ()
    assert "side effects are unknown" in str(restored.state.context())


async def test_list_reads_valid_headers_without_parsing_damaged_history(tmp_path: Path) -> None:
    session = Session.create(tmp_path)
    await session.commit(session.state.append_user("hello"))
    with session.info.path.open("ab") as file:
        file.write(b"not-json\n")

    assert Session.list(tmp_path) == (session.info,)
