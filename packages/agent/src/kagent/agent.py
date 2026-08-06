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
from kagent.state import CompactionEntry, Item, ResponseMetadata, State, StateEntry

type Commit = Callable[[StateEntry], Awaitable[None]]


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

        try:
            await self._append_user(user_input)
            turn = 0
            final_response: Response | None = None
            while True:
                while True:
                    if turn >= self.max_turns:
                        raise MaxTurnsExceeded(f"agent exceeded {self.max_turns} model turns")

                    context = self.state.context()
                    config = self.compaction_config
                    if config is not None:
                        tokens_before = context_tokens(
                            context,
                            instructions=self.instructions,
                            tools=self.tools,
                        )
                    if config is not None and needs_compaction(tokens_before, config):
                        yield CompactionStarted(tokens_before)
                        entry = await compact_state(
                            client=self.client,
                            model=self.model,
                            state=self.state,
                            config=config,
                            tokens_before=tokens_before,
                        )
                        await self._commit(entry)
                        yield CompactionFinished(
                            tokens_before=entry.tokens_before,
                            first_kept_id=entry.first_kept_id,
                            summary=entry.summary,
                        )
                        context = self.state.context()

                    turn += 1
                    yield ModelStarted(turn)
                    response: Response | None = None
                    stream = await self.client.responses.create(
                        model=self.model,
                        instructions=self.instructions,
                        input=cast(ResponseInputParam, context),
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
                    response_entry = self.state.append_items(
                        [
                            cast(Item, item.model_dump(mode="json", exclude_none=True))
                            for item in output_items
                        ],
                        response=ResponseMetadata(
                            id=response.id,
                            model=response.model,
                            usage=response.usage,
                        ),
                    )
                    await self._commit(response_entry)
                    calls = [
                        item for item in output_items if isinstance(item, ResponseFunctionToolCall)
                    ]
                    for item in calls:
                        yield ToolStarted(item)
                        try:
                            outcome = (
                                await execute_tool(item)
                                if execute_tool is not None
                                else ToolResult(
                                    "No tool executor configured",
                                    is_error=True,
                                )
                            )
                        except Exception as error:
                            outcome = ToolResult(
                                f"{type(error).__name__}: {error}",
                                is_error=True,
                            )
                        yield ToolFinished(item, outcome.output, outcome.is_error)
                        tool_entry = self.state.append_items(
                            [
                                {
                                    "type": "function_call_output",
                                    "call_id": item.call_id,
                                    "output": outcome.output,
                                }
                            ]
                        )
                        await self._commit(tool_entry)

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
        except asyncio.CancelledError:
            await self._close_unresolved_tools()
            raise
        finally:
            self._clear_active_task(active_task)

    async def compact(self, instructions: str | None = None) -> CompactionEntry:
        """Compact context while idle, preserving full append-only history."""

        if self.is_running:
            raise AgentBusyError("cannot manually compact a running agent")
        if self.compaction_config is None:
            raise ValueError("compaction is not configured")
        context = self.state.context()
        tokens_before = context_tokens(
            context,
            instructions=self.instructions,
            tools=self.tools,
        )
        entry = await compact_state(
            client=self.client,
            model=self.model,
            state=self.state,
            config=self.compaction_config,
            tokens_before=tokens_before,
            custom_instructions=instructions,
        )
        await self._commit(entry)
        return entry

    async def _append_user(self, message: str) -> None:
        entry = self.state.append_user(message)
        await self._commit(entry)

    async def _commit(self, entry: StateEntry) -> None:
        if self.commit is None:
            return
        try:
            await self.commit(entry)
        except Exception:
            self.state.rollback(entry.id)
            raise

    async def _close_unresolved_tools(self) -> None:
        call_ids = self.state.unresolved_tool_call_ids()
        if not call_ids:
            return
        entry = self.state.append_items(
            [
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": (
                        "Tool execution was cancelled; its side effects are unknown. "
                        "Do not retry automatically."
                    ),
                }
                for call_id in call_ids
            ]
        )
        await self._commit(entry)

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
