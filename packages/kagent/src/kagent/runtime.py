"""AgentRuntime — actor process for agents.

``AgentRuntime`` is the "process" side of the agent framework. While
``Agent`` is a pure handler (config + ``handle()``), the runtime owns
mutable state, a mailbox, sub-agent lifecycle, and concurrency control.

Architecture::

    Signal → Mailbox → _loop() → _dispatch() → Agent.handle()
                                                    ↓
                                              AgentEvent stream
                                                    ↓
                                              Response channel → send() caller

Abort and steer bypass the mailbox as direct method calls.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from kai import Message

from kagent.agent import Agent
from kagent.event import (
    AgentAbort,
    AgentEnd,
    AgentError,
    AgentEvent,
    TurnEnd,
)
from kagent.signal import ChildCompleted, ChildError, Signal, UserInput
from kagent.state import AgentState
from kagent.tools.subagent import CheckSubAgentTool, SpawnSubAgentTool
from kagent.trace.entry import TraceEntry

logger = logging.getLogger("kagent.runtime")

# Sentinel to signal end of response stream
_DONE = object()


@dataclass
class _ChildHandle:
    """Internal tracking for a spawned sub-agent."""

    id: str
    description: str
    status: Literal["running", "completed", "failed"] = "running"
    asyncio_task: asyncio.Task | None = None
    state: AgentState | None = None
    result: str | None = None
    error: str | None = None


class AgentRuntime:
    """Actor runtime for an Agent.

    Owns mutable state, processes signals from a mailbox sequentially,
    manages sub-agent lifecycle, and provides abort/steer controls.

    Example::

        agent = Agent(llm=provider, system="You are helpful.", tools=[my_tool])
        runtime = AgentRuntime(agent)
        await runtime.start()

        async for event in runtime.send(UserInput("Hello")):
            match event:
                case TurnEnd(message=msg):
                    print(msg.extract_text())

        await runtime.stop()
    """

    def __init__(
        self,
        agent: Agent,
        *,
        state: AgentState | None = None,
        can_spawn: bool = True,
    ) -> None:
        self._agent = agent
        self._state = state or AgentState(
            system=agent.system,
            tools=list(agent.tools),
        )
        self._mailbox: asyncio.Queue[tuple[Signal, asyncio.Queue]] = asyncio.Queue()
        self._children: dict[str, _ChildHandle] = {}
        self._abort_event: asyncio.Event | None = None
        self._steer_queue: list[Message] = []
        self._running = False
        self._loop_task: asyncio.Task | None = None

        # Inject runtime tools for sub-agent spawning
        if can_spawn:
            self._state.tools.append(SpawnSubAgentTool.for_runtime(self))
            self._state.tools.append(CheckSubAgentTool.for_runtime(self))

    # --- Properties ---

    @property
    def agent(self) -> Agent:
        """The agent configuration."""
        return self._agent

    @property
    def state(self) -> AgentState:
        """The mutable agent state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Whether the runtime loop is active."""
        return self._running

    @property
    def children(self) -> dict[str, _ChildHandle]:
        """Active and completed sub-agents."""
        return dict(self._children)

    # --- External API ---

    async def start(self) -> None:
        """Start the background processing loop.

        Must be called before ``send()``.

        Raises:
            RuntimeError: If already started.
        """
        if self._loop_task is not None:
            raise RuntimeError("Runtime already started")
        self._running = True
        self._loop_task = asyncio.create_task(self._loop())
        logger.info("AgentRuntime started")

    async def stop(self) -> None:
        """Stop the runtime gracefully.

        Cancels all running sub-agents and the processing loop.
        """
        self._running = False

        # Cancel children
        for child in self._children.values():
            if child.asyncio_task and not child.asyncio_task.done():
                child.asyncio_task.cancel()

        # Cancel loop
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None

        logger.info("AgentRuntime stopped")

    async def send(self, signal: Signal) -> AsyncIterator[AgentEvent]:
        """Send a signal and yield events produced by handling it.

        Creates a per-signal response channel. Events are yielded as the
        runtime processes the signal. Returns when processing is complete.

        Args:
            signal: The signal to process.

        Yields:
            Agent events produced while handling this signal.
        """
        # maxsize=1 provides backpressure: the processing loop blocks
        # after each event until the consumer reads it, ensuring abort/steer
        # signals from the consumer take effect promptly.
        channel: asyncio.Queue[AgentEvent | object] = asyncio.Queue(maxsize=1)
        await self._mailbox.put((signal, channel))

        while True:
            event = await channel.get()
            if event is _DONE:
                break
            yield event  # type: ignore[misc]

    def abort(self) -> None:
        """Abort the currently running handle.

        Takes effect at the next turn boundary. This bypasses the mailbox
        (you can't enqueue "cancel" behind the thing you want to cancel).
        """
        if self._abort_event is not None:
            self._abort_event.set()
            logger.info("Abort requested")

    def steer(self, message: Message) -> None:
        """Inject a steering message into the current run.

        Takes effect at the next turn boundary. The message is added to the
        conversation context and the loop continues.

        This bypasses the mailbox for the same reason as abort.
        """
        self._steer_queue.append(message)
        logger.info("Steer message queued")

    # --- Sub-agent API (called by tools) ---

    def spawn_child(self, *, task: str, system: str | None = None) -> str:
        """Spawn a sub-agent to work on a task.

        Called by ``SpawnSubAgentTool.execute()``.

        The child agent uses the parent's config but without spawn tools
        (flat constraint: children cannot spawn further children).

        Args:
            task: Task description for the sub-agent.
            system: Optional system prompt override.

        Returns:
            The child ID.
        """
        child_id = uuid4().hex[:8]

        # Build child agent WITHOUT spawn tools (flat)
        child_tools = [
            t
            for t in self._agent.tools
            if not isinstance(t, (SpawnSubAgentTool, CheckSubAgentTool))
        ]
        child_agent = Agent(
            llm=self._agent.llm,
            system=system or self._agent.system,
            tools=child_tools,
            context_builder=self._agent.context_builder,
            on_tool_result=self._agent.on_tool_result,
            hooks=self._agent.hooks,
            max_turns=self._agent.max_turns,
        )

        child_state = AgentState(
            system=child_agent.system,
            tools=list(child_tools),
        )
        child_state.trace.append(
            TraceEntry.user(Message(role="user", content=task))
        )

        child_handle = _ChildHandle(
            id=child_id,
            description=task,
            state=child_state,
        )
        self._children[child_id] = child_handle

        # Run in background
        child_handle.asyncio_task = asyncio.create_task(
            self._run_child(child_id, child_agent, child_state)
        )

        logger.info("Spawned child %s: %s", child_id, task[:80])
        return child_id

    def child_status(self, *, child_id: str | None = None) -> str:
        """Get sub-agent status.

        Called by ``CheckSubAgentTool.execute()``.

        Args:
            child_id: Specific child to check, or ``None`` for all.

        Returns:
            Human-readable status string.
        """
        if child_id is not None:
            child = self._children.get(child_id)
            if child is None:
                return f"No sub-agent with ID: {child_id}"
            return self._format_child_status(child)

        if not self._children:
            return "No sub-agents running."

        lines = [self._format_child_status(c) for c in self._children.values()]
        return "\n".join(lines)

    # --- Internal: processing loop ---

    async def _loop(self) -> None:
        """Background loop: process mailbox entries sequentially."""
        while self._running:
            try:
                signal, channel = await asyncio.wait_for(
                    self._mailbox.get(), timeout=1.0
                )
            except TimeoutError:
                continue

            try:
                async for event in self._dispatch(signal):
                    await channel.put(event)
            except Exception as e:
                logger.exception("Error dispatching signal %s", type(signal).__name__)
                await channel.put(AgentError(error=e))
            finally:
                await channel.put(_DONE)

    async def _dispatch(self, signal: Signal) -> AsyncIterator[AgentEvent]:
        """Route a signal to the appropriate handler."""
        match signal:
            case UserInput(text=text):
                async for event in self._handle_user_input(text):
                    yield event
            case ChildCompleted(child_id=cid, result=result):
                async for event in self._handle_child_completed(cid, result):
                    yield event
            case ChildError(child_id=cid, error=err):
                async for event in self._handle_child_error(cid, err):
                    yield event
            case _:
                logger.warning("Unknown signal type: %s", type(signal).__name__)

    async def _handle_user_input(self, text: str) -> AsyncIterator[AgentEvent]:
        """Handle a user input signal."""
        msg = Message(role="user", content=text)
        self._state.trace.append(TraceEntry.user(msg))

        self._abort_event = asyncio.Event()
        try:
            async for event in self._agent.handle(
                self._state,
                should_continue=self._make_should_continue(),
            ):
                if self._abort_event.is_set():
                    yield AgentAbort(messages=list(self._state.messages))
                    yield AgentEnd(messages=list(self._state.messages))
                    return
                yield event
        finally:
            self._abort_event = None

    async def _handle_child_completed(
        self, child_id: str, result: Message
    ) -> AsyncIterator[AgentEvent]:
        """Handle a child completion signal.

        Injects the result into the conversation and lets the agent respond.
        """
        child = self._children.get(child_id)
        desc = child.description if child else child_id

        result_text = result.extract_text() if hasattr(result, "extract_text") else str(result)
        notification = (
            f"[Sub-agent {child_id} completed task: {desc}]\n"
            f"Result: {result_text}"
        )
        msg = Message(role="user", content=notification)
        self._state.trace.append(TraceEntry.user(msg))

        self._abort_event = asyncio.Event()
        try:
            async for event in self._agent.handle(
                self._state,
                should_continue=self._make_should_continue(),
            ):
                yield event
        finally:
            self._abort_event = None

    async def _handle_child_error(
        self, child_id: str, error: Exception
    ) -> AsyncIterator[AgentEvent]:
        """Handle a child error signal.

        Injects the error into the conversation and lets the agent respond.
        """
        child = self._children.get(child_id)
        desc = child.description if child else child_id

        notification = (
            f"[Sub-agent {child_id} failed on task: {desc}]\n"
            f"Error: {error}"
        )
        msg = Message(role="user", content=notification)
        self._state.trace.append(TraceEntry.user(msg))

        self._abort_event = asyncio.Event()
        try:
            async for event in self._agent.handle(
                self._state,
                should_continue=self._make_should_continue(),
            ):
                yield event
        finally:
            self._abort_event = None

    # --- Internal: should_continue wrapper ---

    def _make_should_continue(self):
        """Create a should_continue callback with abort + steer support."""

        async def _should_continue(state: AgentState, assistant_msg: Message) -> bool:
            # Check abort
            if self._abort_event is not None and self._abort_event.is_set():
                return False

            # Check steering — inject queued messages
            if self._steer_queue:
                steer_msg = self._steer_queue.pop(0)
                state.trace.append(TraceEntry.user(steer_msg))
                return True

            # Default: continue if there are tool calls
            return bool(assistant_msg.tool_calls)

        return _should_continue

    # --- Internal: child execution ---

    async def _run_child(
        self, child_id: str, child_agent: Agent, child_state: AgentState
    ) -> None:
        """Run a child agent in the background and report results."""
        child = self._children.get(child_id)
        if child is None:
            return

        try:
            last_msg: Message | None = None
            async for event in child_agent.handle(child_state):
                if isinstance(event, TurnEnd):
                    last_msg = event.message

            if last_msg is not None:
                child.status = "completed"
                child.result = last_msg.extract_text()
                # Deliver to parent mailbox
                await self._mailbox.put((
                    ChildCompleted(child_id=child_id, result=last_msg),
                    asyncio.Queue(),  # fire-and-forget channel
                ))
            else:
                child.status = "failed"
                child.error = "No response produced"
                await self._mailbox.put((
                    ChildError(
                        child_id=child_id,
                        error=RuntimeError("Child produced no response"),
                    ),
                    asyncio.Queue(),
                ))

        except asyncio.CancelledError:
            child.status = "failed"
            child.error = "Cancelled"
        except Exception as e:
            child.status = "failed"
            child.error = str(e)
            logger.exception("Child %s failed", child_id)
            await self._mailbox.put((
                ChildError(child_id=child_id, error=e),
                asyncio.Queue(),
            ))

    # --- Internal: helpers ---

    @staticmethod
    def _format_child_status(child: _ChildHandle) -> str:
        """Format a single child's status for display."""
        line = f"[{child.id}] {child.status}: {child.description[:60]}"
        if child.status == "completed" and child.result:
            line += f"\n  Result: {child.result[:200]}"
        elif child.status == "failed" and child.error:
            line += f"\n  Error: {child.error[:200]}"
        return line
