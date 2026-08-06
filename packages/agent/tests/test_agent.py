from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fakes import FakeClient, FakeResponse
from kagent import (
    Agent,
    CompactionConfig,
    CompactionFinished,
    Env,
    ModelStarted,
    RunFinished,
    Tool,
    ToolExecutor,
    ToolFinished,
    ToolRuntime,
)
from openai.types.responses import ResponseFunctionToolCall
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
    fake = FakeClient([FakeResponse("r1", "", [tool_call]), FakeResponse("r2", "done")])
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


async def test_manual_compaction_preserves_history_and_projects_summary() -> None:
    fake = FakeClient([], summary="cumulative summary")
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
