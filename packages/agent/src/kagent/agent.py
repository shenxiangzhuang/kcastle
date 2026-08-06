"""The minimal stateful Agent core."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import cast

from openai import AsyncOpenAI
from openai.types.responses import (
    FunctionToolParam,
    Response,
    ResponseFunctionToolCall,
    ResponseInputParam,
)
from pydantic import BaseModel

from kagent.compaction import CompactionConfig, compact_state, context_tokens, needs_compaction
from kagent.errors import AgentBusyError, MaxTurnsExceeded
from kagent.events import (
    AgentEvent,
    CompactionFinished,
    CompactionStarted,
    ModelStarted,
    RunFinished,
    TextDelta,
    ToolExecutor,
    ToolFinished,
    ToolResult,
    ToolStarted,
)
from kagent.state import CompactionEntry, Item, State

type Commit = Callable[[], Awaitable[None]]


class Agent:
    """A continuing agent: model, capabilities, causal state, and control queues."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        instructions: str,
        tools: Sequence[FunctionToolParam] = (),
        state: State | None = None,
        commit: Commit | None = None,
        compaction: CompactionConfig | None = None,
        max_turns: int = 100,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.client = client
        self.model = model
        self.instructions = instructions
        self.tools = tuple(tools)
        self.state = state if state is not None else State()
        self.commit = commit
        self.compaction_config = compaction
        self.max_turns = max_turns
        self._steering: deque[str] = deque()
        self._followups: deque[str] = deque()
        self._active_task: asyncio.Task[object] | None = None

    @property
    def is_running(self) -> bool:
        return self._active_task is not None

    def steer(self, message: str) -> None:
        """Inject a message after the current assistant turn and its tools finish."""

        self._require_active(message)
        self._steering.append(message)

    def queue(self, message: str) -> None:
        """Run a message after the agent would otherwise settle."""

        self._require_active(message)
        self._followups.append(message)

    def abort(self) -> None:
        """Cancel the active model/tool/compaction operation."""

        if self._active_task is not None:
            self._active_task.cancel()

    async def run(
        self,
        user_input: str,
        *,
        execute_tool: ToolExecutor | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run until no tool calls, steering messages, or follow-ups remain."""

        if self.is_running:
            raise AgentBusyError("agent is already running; use steer() or queue()")
        if not user_input.strip():
            raise ValueError("user_input must not be empty")

        current = asyncio.current_task()
        if current is None:
            raise RuntimeError("Agent.run() requires an asyncio task")
        active_task = cast(asyncio.Task[object], current)
        self._active_task = active_task
        active_task.add_done_callback(self._clear_active_task)
        await self._append_user(user_input)

        try:
            turn = 0
            final_response: Response | None = None
            while True:
                while True:
                    if turn >= self.max_turns:
                        raise MaxTurnsExceeded(f"agent exceeded {self.max_turns} model turns")

                    async for event in self._compact_if_needed():
                        yield event

                    turn += 1
                    yield ModelStarted(turn)
                    response: Response | None = None
                    stream = await self.client.responses.create(
                        model=self.model,
                        instructions=self.instructions,
                        input=cast(ResponseInputParam, self.state.context()),
                        tools=self.tools,
                        store=False,
                        stream=True,
                    )
                    async for sdk_event in stream:
                        match sdk_event.type:
                            case "response.output_text.delta":
                                yield TextDelta(sdk_event.delta)
                            case "response.completed":
                                response = sdk_event.response
                            case _:
                                pass

                    if response is None:
                        raise RuntimeError("Responses stream ended without response.completed")

                    output_items = cast(list[BaseModel], response.output)
                    self.state.append_items(
                        [
                            cast(Item, item.model_dump(mode="json", exclude_none=True))
                            for item in output_items
                        ]
                    )
                    await self._commit()
                    calls = [
                        item for item in output_items if isinstance(item, ResponseFunctionToolCall)
                    ]
                    tool_outputs: list[Item] = []
                    for item in calls:
                        yield ToolStarted(item)
                        outcome = (
                            await execute_tool(item)
                            if execute_tool is not None
                            else ToolResult("No tool executor configured", is_error=True)
                        )
                        yield ToolFinished(item, outcome.output, outcome.is_error)
                        tool_outputs.append(
                            {
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": outcome.output,
                            }
                        )
                    if tool_outputs:
                        self.state.append_items(tool_outputs)
                        await self._commit()

                    final_response = response
                    if self._steering:
                        await self._append_user(self._steering.popleft())
                        continue
                    if calls:
                        continue
                    break

                if self._followups:
                    await self._append_user(self._followups.popleft())
                    continue
                break

            yield RunFinished(
                output=final_response.output_text,
                response_id=final_response.id,
                usage=final_response.usage,
            )
        finally:
            self._clear_active_task(active_task)

    async def compact(self, instructions: str | None = None) -> CompactionEntry:
        """Compact context while idle, preserving full append-only history."""

        if self.is_running:
            raise AgentBusyError("cannot manually compact a running agent")
        if self.compaction_config is None:
            raise ValueError("compaction is not configured")
        entry = await compact_state(
            client=self.client,
            model=self.model,
            state=self.state,
            config=self.compaction_config,
            instructions=self.instructions,
            tools=self.tools,
            custom_instructions=instructions,
        )
        await self._commit()
        return entry

    async def _compact_if_needed(self) -> AsyncIterator[AgentEvent]:
        config = self.compaction_config
        tools = self.tools
        if config is None or not needs_compaction(
            self.state,
            config,
            instructions=self.instructions,
            tools=tools,
        ):
            return

        tokens_before = context_tokens(
            self.state,
            instructions=self.instructions,
            tools=tools,
        )
        yield CompactionStarted(tokens_before)
        entry = await compact_state(
            client=self.client,
            model=self.model,
            state=self.state,
            config=config,
            instructions=self.instructions,
            tools=tools,
        )
        await self._commit()
        yield CompactionFinished(
            tokens_before=entry.tokens_before,
            first_kept_id=entry.first_kept_id,
            summary=entry.summary,
        )

    async def _append_user(self, message: str) -> None:
        self.state.append_user(message)
        await self._commit()

    async def _commit(self) -> None:
        if self.commit is not None:
            await self.commit()

    def _require_active(self, message: str) -> None:
        if not message.strip():
            raise ValueError("queued message must not be empty")
        if not self.is_running:
            raise RuntimeError("agent is idle; use run() to start a task")

    def _clear_active_task(self, task: asyncio.Task[object]) -> None:
        if self._active_task is task:
            self._steering.clear()
            self._followups.clear()
            self._active_task = None
