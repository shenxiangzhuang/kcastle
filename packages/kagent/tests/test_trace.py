"""Tests for kagent.trace — TraceEntry, Trace, TraceManager, stores."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from kai import Message
from kai.usage import TokenUsage

from kagent.trace import (
    InMemoryTraceStore,
    JsonlTraceStore,
    Trace,
    TraceEntry,
    TraceKind,
    TraceManager,
    TraceMeta,
)


class TestTraceEntry:
    def test_user_factory(self) -> None:
        msg = Message(role="user", content="hello")
        entry = TraceEntry.user(msg, run_id="r1")
        assert entry.id == 0
        assert entry.kind == TraceKind.USER
        assert entry.message is msg
        assert entry.meta.run_id == "r1"
        assert entry.data == {}

    def test_assistant_factory(self) -> None:
        msg = Message(role="assistant", content="hi")
        usage = TokenUsage(input_tokens=10, output_tokens=5)
        entry = TraceEntry.assistant(msg, usage=usage)
        assert entry.kind == TraceKind.ASSISTANT
        assert entry.meta.usage is usage

    def test_tool_result_factory(self) -> None:
        msg = Message(role="tool", content="result", tool_call_id="c1")
        entry = TraceEntry.tool_result(msg, turn_index=2)
        assert entry.kind == TraceKind.TOOL_RESULT
        assert entry.message is msg
        assert entry.meta.turn_index == 2

    def test_frozen(self) -> None:
        entry = TraceEntry.user(Message(role="user", content="hi"))
        with pytest.raises(AttributeError):
            entry.id = 99  # type: ignore[misc]


class TestTraceMetaSerialization:
    def test_empty_meta_roundtrip(self) -> None:
        meta = TraceMeta()
        d = meta.to_dict()
        assert d == {}
        assert TraceMeta.from_dict(d) == meta

    def test_full_meta_roundtrip(self) -> None:
        usage = TokenUsage(input_tokens=100, output_tokens=50, cache_read_tokens=10)
        meta = TraceMeta(timestamp=42.0, run_id="r1", turn_index=3, usage=usage)
        d = meta.to_dict()
        restored = TraceMeta.from_dict(d)
        assert restored.timestamp == 42.0
        assert restored.run_id == "r1"
        assert restored.turn_index == 3
        assert restored.usage == usage

    def test_usage_fields_in_dict(self) -> None:
        usage = TokenUsage(input_tokens=10, output_tokens=5, cache_write_tokens=2)
        meta = TraceMeta(usage=usage)
        d = meta.to_dict()
        assert d["usage"]["input_tokens"] == 10
        assert d["usage"]["cache_write_tokens"] == 2


class TestTraceEntrySerialization:
    def test_user_roundtrip(self) -> None:
        entry = TraceEntry(
            id=1,
            kind=TraceKind.USER,
            message=Message(role="user", content="hello"),
            meta=TraceMeta(timestamp=1.0, run_id="r1"),
        )
        d = entry.to_dict()
        restored = TraceEntry.from_dict(d)
        assert restored.id == 1
        assert restored.kind == TraceKind.USER
        assert restored.message is not None
        assert restored.message.extract_text() == "hello"
        assert restored.meta.run_id == "r1"

    def test_assistant_with_usage_roundtrip(self) -> None:
        usage = TokenUsage(input_tokens=100, output_tokens=50, cache_read_tokens=10)
        entry = TraceEntry(
            id=2,
            kind=TraceKind.ASSISTANT,
            message=Message(role="assistant", content="reply", usage=usage),
            meta=TraceMeta(timestamp=2.0, usage=usage),
        )
        restored = TraceEntry.from_dict(entry.to_dict())
        assert restored.meta.usage is not None
        assert restored.meta.usage.input_tokens == 100
        assert restored.meta.usage.cache_read_tokens == 10

    def test_data_roundtrip(self) -> None:
        """Entries with arbitrary data survive serialization."""
        entry = TraceEntry(
            id=3,
            kind=TraceKind.USER,
            data={"key": "value"},
            meta=TraceMeta(timestamp=3.0),
        )
        restored = TraceEntry.from_dict(entry.to_dict())
        assert restored.data["key"] == "value"
        assert restored.message is None

    def test_is_json_serializable(self) -> None:
        """to_dict() output can be passed through json.dumps/loads."""
        entry = TraceEntry(
            id=1,
            kind=TraceKind.USER,
            message=Message(role="user", content="hi"),
            meta=TraceMeta(timestamp=1.0),
        )
        roundtripped = json.loads(json.dumps(entry.to_dict()))
        restored = TraceEntry.from_dict(roundtripped)
        assert restored.message is not None
        assert restored.message.extract_text() == "hi"


class TestTraceMeta:
    def test_defaults(self) -> None:
        meta = TraceMeta()
        assert meta.timestamp == 0.0
        assert meta.run_id is None
        assert meta.turn_index is None
        assert meta.usage is None

    def test_with_timestamp_auto_fills(self) -> None:
        meta = TraceMeta()
        filled = meta.with_timestamp()
        assert filled.timestamp > 0.0

    def test_with_timestamp_preserves_existing(self) -> None:
        meta = TraceMeta(timestamp=42.0)
        same = meta.with_timestamp()
        assert same.timestamp == 42.0


class TestTrace:
    def test_create_with_defaults(self) -> None:
        trace = Trace()
        assert len(trace.id) == 32  # uuid4 hex
        assert trace.name == ""
        assert trace.created_at > 0
        assert len(trace) == 0

    def test_create_with_id_and_name(self) -> None:
        trace = Trace(id="abc", name="session-1")
        assert trace.id == "abc"
        assert trace.name == "session-1"

    def test_append_assigns_id(self) -> None:
        trace = Trace()
        entry = TraceEntry.user(Message(role="user", content="hi"))
        stored = trace.append(entry)
        assert stored.id == 1
        assert stored.meta.timestamp > 0

    def test_auto_increment_ids(self) -> None:
        trace = Trace()
        e1 = trace.append(TraceEntry.user(Message(role="user", content="a")))
        e2 = trace.append(TraceEntry.assistant(Message(role="assistant", content="b")))
        assert e1.id == 1
        assert e2.id == 2

    def test_entries_returns_copy(self) -> None:
        trace = Trace()
        trace.append(TraceEntry.user(Message(role="user", content="hi")))
        entries = trace.entries
        assert len(entries) == 1
        entries.clear()
        assert len(trace) == 1  # original unaffected

    def test_messages(self) -> None:
        trace = Trace()
        trace.append(TraceEntry.user(Message(role="user", content="hi")))
        trace.append(TraceEntry.assistant(Message(role="assistant", content="hello")))
        trace.append(TraceEntry.tool_result(Message(role="tool", content="r", tool_call_id="c")))

        msgs = trace.messages()
        assert len(msgs) == 3
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"
        assert msgs[2].role == "tool"

    def test_reset(self) -> None:
        trace = Trace()
        trace.append(TraceEntry.user(Message(role="user", content="hi")))
        trace.reset()
        assert len(trace) == 0

        # IDs restart after reset
        e = trace.append(TraceEntry.user(Message(role="user", content="new")))
        assert e.id == 1

    def test_from_records(self) -> None:
        entries = [
            TraceEntry(
                id=1,
                kind=TraceKind.USER,
                message=Message(role="user", content="hello"),
                meta=TraceMeta(timestamp=100.0),
            ),
            TraceEntry(
                id=2,
                kind=TraceKind.ASSISTANT,
                message=Message(role="assistant", content="hi"),
                meta=TraceMeta(timestamp=101.0),
            ),
        ]
        trace = Trace.from_records(
            id="restored",
            name="old-session",
            created_at=99.0,
            entries=entries,
        )
        assert trace.id == "restored"
        assert trace.name == "old-session"
        assert trace.created_at == 99.0
        assert len(trace) == 2
        # Next ID should continue from max
        e3 = trace.append(TraceEntry.user(Message(role="user", content="new")))
        assert e3.id == 3


class TestTraceManager:
    def test_create_and_get(self) -> None:
        mgr = TraceManager()
        trace = mgr.create("session-1")
        assert mgr.get(trace.id) is trace
        assert trace.name == "session-1"

    def test_list_traces(self) -> None:
        mgr = TraceManager()
        t1 = mgr.create("a")
        t2 = mgr.create("b")
        ids = mgr.list_traces()
        assert t1.id in ids
        assert t2.id in ids

    def test_register(self) -> None:
        mgr = TraceManager()
        trace = Trace(name="external")
        mgr.register(trace)
        assert mgr.get(trace.id) is trace

    def test_with_store(self) -> None:
        store = InMemoryTraceStore()
        mgr = TraceManager(store=store)

        trace = mgr.create("test")
        assert trace.id in store.list_traces()

        entry = TraceEntry.user(Message(role="user", content="hi"))
        trace.append(entry)

        # Verify it was persisted
        header, entries = store.load(trace.id)
        assert header["name"] == "test"
        assert len(entries) == 1
        assert entries[0].kind == TraceKind.USER

    def test_load_from_store(self) -> None:
        store = InMemoryTraceStore()
        # Set up data directly in store
        store.create("t1", "loaded-session", 1000.0)
        store.append(
            "t1",
            TraceEntry(
                id=1,
                kind=TraceKind.USER,
                message=Message(role="user", content="saved"),
                meta=TraceMeta(timestamp=1000.1),
            ),
        )

        mgr = TraceManager(store=store)
        trace = mgr.load("t1")
        assert trace.id == "t1"
        assert trace.name == "loaded-session"
        assert len(trace) == 1
        assert trace.messages()[0].extract_text() == "saved"

    def test_load_without_store_raises(self) -> None:
        mgr = TraceManager()
        with pytest.raises(RuntimeError, match="no TraceStore"):
            mgr.load("x")


class TestInMemoryTraceStore:
    def test_create_and_load(self) -> None:
        store = InMemoryTraceStore()
        store.create("t1", "test", 1.0)
        header, entries = store.load("t1")
        assert header["id"] == "t1"
        assert entries == []

    def test_append_and_load(self) -> None:
        store = InMemoryTraceStore()
        store.create("t1", "test", 1.0)
        entry = TraceEntry(
            id=1,
            kind=TraceKind.USER,
            message=Message(role="user", content="hi"),
            meta=TraceMeta(timestamp=1.0),
        )
        store.append("t1", entry)
        _, entries = store.load("t1")
        assert len(entries) == 1

    def test_append_unknown_raises(self) -> None:
        store = InMemoryTraceStore()
        with pytest.raises(KeyError):
            store.append("nope", TraceEntry.user(Message(role="user", content="x")))

    def test_load_unknown_raises(self) -> None:
        store = InMemoryTraceStore()
        with pytest.raises(KeyError):
            store.load("nope")

    def test_list_traces(self) -> None:
        store = InMemoryTraceStore()
        store.create("b", "b", 1.0)
        store.create("a", "a", 2.0)
        assert store.list_traces() == ["a", "b"]


class TestJsonlTraceStore:
    def test_create_writes_header(self, tmp_path: Path) -> None:
        store = JsonlTraceStore(tmp_path)
        store.create("t1", "my-session", 1000.0)

        path = tmp_path / "t1.jsonl"
        assert path.exists()
        header = json.loads(path.read_text().strip())
        assert header["type"] == "header"
        assert header["id"] == "t1"
        assert header["name"] == "my-session"

    def test_append_and_load(self, tmp_path: Path) -> None:
        store = JsonlTraceStore(tmp_path)
        store.create("t1", "test", 1.0)

        msg = Message(role="user", content="hello world")
        entry = TraceEntry(
            id=1,
            kind=TraceKind.USER,
            message=msg,
            meta=TraceMeta(timestamp=1.1, run_id="r1", turn_index=0),
        )
        store.append("t1", entry)

        header, entries = store.load("t1")
        assert header["id"] == "t1"
        assert len(entries) == 1
        assert entries[0].id == 1
        assert entries[0].kind == TraceKind.USER
        assert entries[0].message is not None
        assert entries[0].message.extract_text() == "hello world"
        assert entries[0].meta.run_id == "r1"
        assert entries[0].meta.turn_index == 0

    def test_roundtrip_with_usage(self, tmp_path: Path) -> None:
        store = JsonlTraceStore(tmp_path)
        store.create("t1", "test", 1.0)

        usage = TokenUsage(input_tokens=100, output_tokens=50, cache_read_tokens=10)
        msg = Message(role="assistant", content="reply", usage=usage)
        entry = TraceEntry(
            id=1,
            kind=TraceKind.ASSISTANT,
            message=msg,
            meta=TraceMeta(timestamp=1.1, usage=usage),
        )
        store.append("t1", entry)

        _, entries = store.load("t1")
        assert entries[0].meta.usage is not None
        assert entries[0].meta.usage.input_tokens == 100
        assert entries[0].meta.usage.cache_read_tokens == 10

    def test_roundtrip_with_data(self, tmp_path: Path) -> None:
        store = JsonlTraceStore(tmp_path)
        store.create("t1", "test", 1.0)
        entry = TraceEntry(
            id=1,
            kind=TraceKind.USER,
            data={"key": "value"},
            meta=TraceMeta(timestamp=1.1),
        )
        store.append("t1", entry)

        _, entries = store.load("t1")
        assert entries[0].kind == TraceKind.USER
        assert entries[0].data["key"] == "value"
        assert entries[0].message is None

    def test_append_unknown_raises(self, tmp_path: Path) -> None:
        store = JsonlTraceStore(tmp_path)
        with pytest.raises(KeyError):
            store.append("nope", TraceEntry.user(Message(role="user", content="x")))

    def test_load_unknown_raises(self, tmp_path: Path) -> None:
        store = JsonlTraceStore(tmp_path)
        with pytest.raises(KeyError):
            store.load("nope")

    def test_list_traces(self, tmp_path: Path) -> None:
        store = JsonlTraceStore(tmp_path)
        store.create("beta", "b", 1.0)
        store.create("alpha", "a", 2.0)
        assert store.list_traces() == ["alpha", "beta"]

    def test_multiple_entries(self, tmp_path: Path) -> None:
        store = JsonlTraceStore(tmp_path)
        store.create("t1", "multi", 1.0)
        for i in range(5):
            store.append(
                "t1",
                TraceEntry(
                    id=i + 1,
                    kind=TraceKind.USER,
                    message=Message(role="user", content=f"msg-{i}"),
                    meta=TraceMeta(timestamp=float(i)),
                ),
            )
        _, entries = store.load("t1")
        assert len(entries) == 5
        assert entries[4].message is not None
        assert entries[4].message.extract_text() == "msg-4"

    def test_session_resume_via_manager(self, tmp_path: Path) -> None:
        """End-to-end: create → append → (new manager) → load → continue."""
        store = JsonlTraceStore(tmp_path)

        # Session 1: create and record
        mgr1 = TraceManager(store=store)
        trace1 = mgr1.create("resumable")
        trace1.append(TraceEntry.user(Message(role="user", content="start")))
        trace1.append(
            TraceEntry.assistant(Message(role="assistant", content="ack")),
        )
        trace_id = trace1.id

        # Session 2: new manager, load, continue
        mgr2 = TraceManager(store=store)
        trace2 = mgr2.load(trace_id)
        assert len(trace2) == 2
        assert trace2.messages()[0].extract_text() == "start"
        assert trace2.messages()[1].extract_text() == "ack"

        # Continue appending
        trace2.append(TraceEntry.user(Message(role="user", content="resume")))
        assert len(trace2) == 3

        # Verify persisted
        _, entries = store.load(trace_id)
        assert len(entries) == 3
