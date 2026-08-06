from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fakes import FakeClient, FakeItem, FakeResponse
from kagent import (
    Agent,
    CompactionConfig,
    CompactionFinished,
    Env,
    ItemEntry,
    ModelStarted,
    RunFinished,
    Session,
    State,
    Tool,
    ToolExecutor,
    ToolFinished,
    ToolResult,
    ToolRuntime,
)
from openai.types.responses import ResponseFunctionToolCall, ResponseUsage
from pydantic import BaseModel


class EchoParams(BaseModel):
    text: str


async def echo(params: EchoParams, _: Env) -> str:
    return params.text.upper()


async def collect(
    agent: Agent,
    prompt: str,
    execute_tool: ToolExecutor | None = None,
) -> list[object]:
    return [event async for event in agent.run(prompt, execute_tool=execute_tool)]


async def test_simple_run_streams_and_settles() -> None:
    fake = FakeClient([FakeResponse("r1", "hello")])
    agent = Agent(client=fake.as_openai(), model="test", instructions="test")

    events = await collect(agent, "hi")

    assert isinstance(events[-1], RunFinished)
    assert events[-1].output == "hello"
    assert not agent.is_running
    assert agent.state.context()[0]["content"] == "hi"


async def test_tool_result_is_returned_to_next_model_turn() -> None:
    tool_call = ResponseFunctionToolCall(
        type="function_call",
        call_id="call-1",
        name="echo",
        arguments='{"text":"hello"}',
    )
    usage = ResponseUsage(
        input_tokens=10,
        input_tokens_details={"cached_tokens": 5},
        output_tokens=2,
        output_tokens_details={"reasoning_tokens": 1},
        total_tokens=12,
    )
    fake = FakeClient(
        [
            FakeResponse("r1", "", [tool_call], usage),
            FakeResponse("r2", "done", usage=usage),
        ]
    )
    tools = ToolRuntime(
        Env(Path.cwd()),
        [Tool(name="echo", description="Echo", params=EchoParams, run=echo)],
    )
    agent = Agent(
        client=fake.as_openai(),
        model="test",
        instructions="test",
        tools=tools.schemas,
    )

    events = await collect(agent, "use the tool", tools.execute)

    finished = next(event for event in events if isinstance(event, ToolFinished))
    assert finished.output == "HELLO"
    second_input = fake.responses.requests[1]["input"]
    assert "HELLO" in str(second_input)
    responses = [
        entry.response
        for entry in agent.state.entries
        if isinstance(entry, ItemEntry) and entry.response is not None
    ]
    assert [response.id for response in responses] == ["r1", "r2"]
    assert all(response.usage == usage for response in responses)


async def test_each_tool_result_is_persisted_before_the_next_tool(tmp_path: Path) -> None:
    calls: list[FakeItem | ResponseFunctionToolCall] = [
        ResponseFunctionToolCall(
            type="function_call",
            call_id=f"call-{index}",
            name="echo",
            arguments='{"text":"hello"}',
        )
        for index in (1, 2)
    ]
    fake = FakeClient([FakeResponse("r1", "", calls)])
    session = Session.create(tmp_path)
    second_started = asyncio.Event()

    async def execute(call: ResponseFunctionToolCall) -> ToolResult:
        if call.call_id == "call-1":
            return ToolResult("FIRST")
        second_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    agent = Agent(
        client=fake.as_openai(),
        model="test",
        instructions="test",
        state=session.state,
        commit=session.commit,
    )
    task = asyncio.create_task(collect(agent, "run both", execute))
    await second_started.wait()

    assert "FIRST" in session.info.path.read_text()
    agent.abort()
    with pytest.raises(asyncio.CancelledError):
        await task

    restored = Session.open(session.info.path)
    assert restored.state.unresolved_tool_call_ids() == ()
    assert "FIRST" in str(restored.state.context())
    assert "side effects are unknown" in str(restored.state.context())


async def test_steer_runs_at_next_turn_boundary() -> None:
    fake = FakeClient([FakeResponse("r1", "first"), FakeResponse("r2", "second")])
    agent = Agent(client=fake.as_openai(), model="test", instructions="test")
    events: list[object] = []

    async for event in agent.run("start"):
        events.append(event)
        if isinstance(event, ModelStarted) and event.turn == 1:
            agent.steer("change direction")

    assert len(fake.responses.requests) == 2
    assert "change direction" in str(fake.responses.requests[1]["input"])
    assert isinstance(events[-1], RunFinished)
    assert events[-1].output == "second"


async def test_queue_runs_only_after_agent_would_settle() -> None:
    fake = FakeClient([FakeResponse("r1", "first"), FakeResponse("r2", "follow-up")])
    agent = Agent(client=fake.as_openai(), model="test", instructions="test")

    async for event in agent.run("start"):
        if isinstance(event, ModelStarted) and event.turn == 1:
            agent.queue("afterwards")

    assert "afterwards" not in str(fake.responses.requests[0]["input"])
    assert "afterwards" in str(fake.responses.requests[1]["input"])


async def test_abort_cancels_active_run() -> None:
    fake = FakeClient([FakeResponse("r1", "restarted")])
    agent = Agent(client=fake.as_openai(), model="test", instructions="test")
    started = asyncio.Event()

    async def consume() -> None:
        async for event in agent.run("start"):
            if isinstance(event, ModelStarted):
                started.set()
                await asyncio.Event().wait()

    task = asyncio.create_task(consume())
    await started.wait()
    agent.steer("stale steering")
    agent.queue("stale follow-up")
    agent.abort()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not agent.is_running

    await collect(agent, "restart")
    assert "stale" not in str(agent.state.context())


async def test_failed_commit_rolls_back_the_uncommitted_tail() -> None:
    async def fail(_: object) -> None:
        raise OSError("disk full")

    agent = Agent(
        client=FakeClient([]).as_openai(),
        model="test",
        instructions="test",
        commit=fail,
    )

    with pytest.raises(OSError, match="disk full"):
        await collect(agent, "not durable")

    assert len(agent.state) == 0
    assert not agent.is_running


async def test_context_is_projected_once_per_model_turn() -> None:
    class CountingState(State):
        calls = 0

        def context(self) -> list[dict[str, object]]:
            self.calls += 1
            return super().context()

    state = CountingState()
    agent = Agent(
        client=FakeClient([FakeResponse("r1", "done")]).as_openai(),
        model="test",
        instructions="test",
        state=state,
    )

    await collect(agent, "hello")

    assert state.calls == 1


async def test_manual_compaction_preserves_history_and_projects_summary() -> None:
    usage = ResponseUsage(
        input_tokens=100,
        input_tokens_details={"cached_tokens": 0},
        output_tokens=20,
        output_tokens_details={"reasoning_tokens": 5},
        total_tokens=120,
    )
    fake = FakeClient([], summary="cumulative summary", summary_usage=usage)
    agent = Agent(
        client=fake.as_openai(),
        model="test",
        instructions="test",
        compaction=CompactionConfig(
            context_window=100,
            reserve_tokens=10,
            keep_recent_tokens=1,
        ),
    )
    first = agent.state.append_items([{"role": "user", "content": "old" * 20}])
    agent.state.append_items([{"role": "user", "content": "recent"}])

    result = await agent.compact()

    assert isinstance(result.summary, str)
    assert first.items[0] not in agent.state.context()
    assert any("cumulative summary" in str(item) for item in agent.state.context())
    assert len(agent.state.entries) == 3
    assert result.response is not None
    assert result.response.usage == usage


async def test_auto_compaction_emits_events() -> None:
    fake = FakeClient([FakeResponse("r1", "done")], summary="summary")
    agent = Agent(
        client=fake.as_openai(),
        model="test",
        instructions="test",
        compaction=CompactionConfig(
            context_window=30,
            reserve_tokens=10,
            keep_recent_tokens=1,
        ),
    )
    agent.state.append_items([{"role": "user", "content": "old" * 30}])

    events = await collect(agent, "new")

    assert any(isinstance(event, CompactionFinished) for event in events)
