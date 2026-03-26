"""Tests for Session/AgentRuntime integration."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from kagent import Agent
from kai import Context, Message, ProviderBase, StreamEvent

from kcastle.session import Session


class MockProvider(ProviderBase):
    """Simple mock provider that returns fixed responses."""

    @property
    def provider(self) -> str:
        return "mock"

    @property
    def model(self) -> str:
        return "test"

    async def complete(self, messages: list[Message], **kwargs: Any) -> dict[str, Any]:
        return {"content": "Hello! I'm working.", "usage": {}}

    async def stream(self, context: Context, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        """Mock stream implementation."""
        from kai.types.stream import Done, TextDelta

        for char in "Hello! I'm working.":
            yield TextDelta(delta=char)

        yield Done(message=Message(role="assistant", content="Hello! I'm working."))

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
    events: list[object] = []
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
