import json
from pathlib import Path

from kagent import Session


async def test_session_is_a_titled_append_only_jsonl_stream(tmp_path: Path) -> None:
    session = Session.create(tmp_path)
    path = session.info.path
    assert path.suffix == ".jsonl"
    assert not path.exists()

    session.state.append_user("hello")
    await session.commit()
    first_commit = path.read_text()
    lines = first_commit.splitlines()

    assert json.loads(lines[0])["title"] == "hello"
    assert json.loads(lines[0])["created_at"]
    assert json.loads(lines[1])["type"] == "items"

    session.state.append_user("again")
    await session.commit()
    restored = Session.open(path)

    assert path.read_text().startswith(first_commit)
    assert len(path.read_text().splitlines()) == 3
    assert restored.info.title == "hello"
    assert [item["content"] for item in restored.state.context()] == ["hello", "again"]
    assert Session.list(tmp_path)[0].title == "hello"


async def test_each_created_session_gets_its_own_file(tmp_path: Path) -> None:
    first = Session.create(tmp_path)
    second = Session.create(tmp_path)
    first.state.append_user("first session")
    second.state.append_user("second session")

    await first.commit()
    await second.commit()

    assert first.info.path != second.info.path
    assert {info.title for info in Session.list(tmp_path)} == {
        "first session",
        "second session",
    }
