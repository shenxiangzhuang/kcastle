"""Tests for Session/AgentRuntime integration."""

from __future__ import annotations

from pathlib import Path

import pytest
from kagent import Agent

from kcastle.session import Session


class MockProvider:
    """Simple mock provider that returns fixed responses."""

    def __init__(self) -> None:
        self.provider = "mock"
        self.model = "test"

    async def complete(self, messages: list, **kwargs):
        return {"content": "Hello! I'm working.", "usage": {}}

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_session_starts_runtime_on_first_run(tmp_path: Path) -> None:
    """Test that Session properly starts AgentRuntime on first run."""

    # Create a simple agent factory
    def agent_factory() -> Agent:
        return Agent(llm=MockProvider(), system="Test agent")

    # Create session
    session = Session.create(
        session_dir=tmp_path / "test_session",
        session_id="test-runtime-start",
        name="Test Session",
        agent_factory=agent_factory,
    )

    # Runtime should not be started yet
    assert not session._runtime_started
    assert session._runtime._loop_task is None

    # Run should start the runtime
    events = []
    async for event in session.run("Hello"):
        events.append(event)

    # Runtime should now be started
    assert session._runtime_started
    assert session._runtime._loop_task is not None
    assert session._runtime.is_running

    # Should have received events
    assert len(events) > 0

    # Suspend should stop the runtime
    session.suspend()
    assert not session._runtime_started
