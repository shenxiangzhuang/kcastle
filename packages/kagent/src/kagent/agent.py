"""Level 2: Stateful agent SDK.

The ``Agent`` class wraps ``agent_loop()`` with persistent state, interactive
control (steering, follow-up, abort), and dual consumption modes
(``run()`` for streaming, ``complete()`` for one-shot).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from kai import Message, Provider, Tool

from kagent.event import AgentEvent, TurnEnd
from kagent.loop import (
    BuildContextFn,
    OnToolResultFn,
    ShouldContinueFn,
    agent_loop,
)
from kagent.state import AgentState


class Agent:
    """A stateful agent with conversation persistence and interactive control.

    Example — streaming::

        agent = Agent(
            provider=OpenAICompletions(model="gpt-4o"),
            system="You are helpful.",
            tools=[my_tool],
        )
        async for event in agent.run("What's the weather?"):
            match event:
                case StreamChunk(event=e):
                    print(e, end="", flush=True)
                case TurnEnd(message=msg):
                    print(msg.extract_text())

    Example — one-shot::

        msg = await agent.complete("What's 2+2?")
        print(msg.extract_text())

    Example — multi-turn::

        await agent.complete("Remember: my name is Alice.")
        msg = await agent.complete("What's my name?")
        # "Alice" — conversation persists across calls.
    """

    def __init__(
        self,
        *,
        provider: Provider,
        system: str | None = None,
        tools: list[Tool] | None = None,
        build_context: BuildContextFn | None = None,
        on_tool_result: OnToolResultFn | None = None,
        should_continue: ShouldContinueFn | None = None,
        max_turns: int = 100,
    ) -> None:
        """Create an Agent.

        All callback parameters are optional. If not provided, sensible defaults
        are used. These are the same callbacks as ``agent_loop()`` — no wrapper types.
        """
        self._provider = provider
        self._build_context = build_context
        self._on_tool_result = on_tool_result
        self._should_continue = should_continue
        self._max_turns = max_turns
        self._state = AgentState(
            system=system,
            messages=[],
            tools=list(tools) if tools else [],
        )
        self._running = False
        self._abort_event: asyncio.Event | None = None
        self._steer_queue: list[Message] = []
        self._follow_up_queue: list[Message] = []

    # --- State ---

    @property
    def state(self) -> AgentState:
        """The mutable agent state. Read or modify directly."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Whether the agent is currently executing."""
        return self._running

    # --- Primary API ---

    async def run(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """Run the agent with streaming events (pull-based).

        Appends a user message, runs the loop, yields all events.
        """
        self._state.messages.append(Message(role="user", content=user_input))
        async for event in self._run_loop():
            yield event

    async def complete(self, user_input: str) -> Message:
        """Run the agent and return the final assistant message.

        Convenience wrapper — consumes the event stream silently.
        Mirrors kai's ``complete()`` pattern.

        Raises:
            RuntimeError: If the loop ends without producing an assistant message.
        """
        last_assistant: Message | None = None
        async for event in self.run(user_input):
            if isinstance(event, TurnEnd):
                last_assistant = event.message

        if last_assistant is None:
            raise RuntimeError("Agent loop ended without producing an assistant message")
        return last_assistant

    # --- Interactive control ---

    def steer(self, message: Message) -> None:
        """Interrupt the current run with a message.

        Injected after current tool finishes. Remaining tools are skipped.
        The loop continues with the steering message in context.

        Note: Steering takes effect on the next turn boundary.
        """
        self._steer_queue.append(message)

    def follow_up(self, message: Message) -> None:
        """Queue a message for after the current run.

        When the loop would stop, it continues with queued follow-ups instead.
        """
        self._follow_up_queue.append(message)

    def abort(self) -> None:
        """Cancel the current run."""
        if self._abort_event is not None:
            self._abort_event.set()

    # --- Internal ---

    async def _run_loop(self) -> AsyncIterator[AgentEvent]:
        """Run the agent loop with steering and follow-up support."""
        if self._running:
            raise RuntimeError("Agent is already running")

        self._running = True
        self._abort_event = asyncio.Event()

        try:
            while True:
                # Run the main loop
                async for event in agent_loop(
                    provider=self._provider,
                    state=self._state,
                    build_context=self._build_context,
                    on_tool_result=self._on_tool_result,
                    should_continue=self._wrap_should_continue(),
                    max_turns=self._max_turns,
                ):
                    if self._abort_event.is_set():
                        return
                    yield event

                # After the loop ends, check for follow-ups
                if self._follow_up_queue:
                    follow_up = self._follow_up_queue.pop(0)
                    self._state.messages.append(follow_up)
                else:
                    break
        finally:
            self._running = False
            self._abort_event = None

    def _wrap_should_continue(self) -> ShouldContinueFn | None:
        """Wrap the user's should_continue callback with steering logic."""
        user_cb = self._should_continue

        async def _combined(state: AgentState, assistant_msg: Message) -> bool:
            # Check abort
            if self._abort_event is not None and self._abort_event.is_set():
                return False

            # Check steering — inject queued steering messages
            if self._steer_queue:
                steer_msg = self._steer_queue.pop(0)
                state.messages.append(steer_msg)
                return True  # Continue looping with the steering message

            # Delegate to user callback if provided
            if user_cb is not None:
                return await user_cb(state, assistant_msg)

            # Default: continue if there are tool calls
            return bool(assistant_msg.tool_calls)

        return _combined
