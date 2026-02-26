"""Lifecycle hooks for observing agent execution.

The ``Hooks`` base class provides no-op implementations for all lifecycle
methods.  Subclass it and override only the methods you care about.

Built-in implementations:

- ``LoggingHooks`` — logs lifecycle events via stdlib ``logging``.
- ``MultiHooks`` — fans out to multiple hooks instances.

OTel integration is available in ``kagent.otel.OTelHooks`` (requires
``opentelemetry-api``).

Example — custom hooks::

    class MyHooks(Hooks):
        def on_agent_start(self, *, run_id, model, provider):
            print(f"Agent started: {model}")

        def on_tool_end(
            self, *, run_id, turn_index, call_id, tool_name, result, duration_ms, is_error
        ):
            if is_error:
                send_alert(f"Tool {tool_name} failed!")


    agent = Agent(provider=p, hooks=MyHooks())
"""

from __future__ import annotations

import logging
from typing import Any

from kai import Message, ToolResult
from kai.usage import TokenUsage

_log = logging.getLogger("kagent.hooks")


class Hooks:
    """Lifecycle hooks for observing agent execution.

    All methods are no-ops by default.  Subclass and override only the
    methods you need.  Hook methods are **synchronous** — if you need
    async I/O, buffer events and flush asynchronously.

    Hook call order within a turn::

        on_agent_start
        ├── on_turn_start
        │   ├── on_llm_start
        │   │   └── (LLM streaming)
        │   ├── on_llm_end
        │   ├── on_tool_start   (per tool call)
        │   ├── on_tool_end     (per tool call)
        │   └── on_turn_end
        └── on_agent_end
    """

    def on_agent_start(self, *, run_id: str, model: str, provider: str) -> None:
        """Called when the agent loop starts."""

    def on_agent_end(
        self,
        *,
        run_id: str,
        turn_count: int,
        duration_ms: float,
        usage: TokenUsage | None,
    ) -> None:
        """Called when the agent loop ends normally."""

    def on_turn_start(self, *, run_id: str, turn_index: int) -> None:
        """Called at the start of each turn."""

    def on_turn_end(
        self,
        *,
        run_id: str,
        turn_index: int,
        message: Message,
        tool_results: list[Message],
        llm_duration_ms: float,
        duration_ms: float,
    ) -> None:
        """Called at the end of each turn."""

    def on_llm_start(self, *, run_id: str, turn_index: int) -> None:
        """Called when the LLM streaming call starts (within a turn)."""

    def on_llm_end(
        self,
        *,
        run_id: str,
        turn_index: int,
        message: Message,
        duration_ms: float,
    ) -> None:
        """Called when the LLM streaming call completes."""

    def on_tool_start(
        self,
        *,
        run_id: str,
        turn_index: int,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Called when a tool execution starts."""

    def on_tool_end(
        self,
        *,
        run_id: str,
        turn_index: int,
        call_id: str,
        tool_name: str,
        result: ToolResult,
        duration_ms: float,
        is_error: bool,
    ) -> None:
        """Called when a tool execution ends."""


class LoggingHooks(Hooks):
    """Hooks implementation that logs lifecycle events via stdlib ``logging``.

    Uses the ``kagent.hooks`` logger.  Set ``level`` to control verbosity:

    - ``DEBUG``: logs all events including tool arguments and results.
    - ``INFO`` (default): logs agent/turn/tool summaries.

    Example::

        import logging

        logging.basicConfig(level=logging.INFO)

        agent = Agent(provider=p, hooks=LoggingHooks())
    """

    def __init__(self, *, level: int = logging.INFO) -> None:
        self._level = level

    def on_agent_start(self, *, run_id: str, model: str, provider: str) -> None:
        _log.log(
            self._level,
            "[%s] Agent start: provider=%s model=%s",
            run_id,
            provider,
            model,
        )

    def on_agent_end(
        self,
        *,
        run_id: str,
        turn_count: int,
        duration_ms: float,
        usage: TokenUsage | None,
    ) -> None:
        tokens = ""
        if usage:
            tokens = f" tokens={usage.input_tokens}/{usage.output_tokens}"
        _log.log(
            self._level,
            "[%s] Agent end: turns=%d duration=%.0fms%s",
            run_id,
            turn_count,
            duration_ms,
            tokens,
        )

    def on_turn_start(self, *, run_id: str, turn_index: int) -> None:
        _log.debug("[%s] Turn %d start", run_id, turn_index)

    def on_turn_end(
        self,
        *,
        run_id: str,
        turn_index: int,
        message: Message,
        tool_results: list[Message],
        llm_duration_ms: float,
        duration_ms: float,
    ) -> None:
        usage = message.usage
        tokens = ""
        if usage:
            tokens = f" tokens={usage.input_tokens}/{usage.output_tokens}"
        tools_info = f" tools={len(tool_results)}" if tool_results else ""
        _log.log(
            self._level,
            "[%s] Turn %d end: llm=%.0fms total=%.0fms%s%s",
            run_id,
            turn_index,
            llm_duration_ms,
            duration_ms,
            tokens,
            tools_info,
        )

    def on_llm_start(self, *, run_id: str, turn_index: int) -> None:
        _log.debug("[%s] Turn %d LLM start", run_id, turn_index)

    def on_llm_end(
        self,
        *,
        run_id: str,
        turn_index: int,
        message: Message,
        duration_ms: float,
    ) -> None:
        stop = message.stop_reason or "unknown"
        _log.debug(
            "[%s] Turn %d LLM end: stop=%s duration=%.0fms",
            run_id,
            turn_index,
            stop,
            duration_ms,
        )

    def on_tool_start(
        self,
        *,
        run_id: str,
        turn_index: int,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        _log.debug(
            "[%s] Turn %d tool %s start (call_id=%s)",
            run_id,
            turn_index,
            tool_name,
            call_id,
        )

    def on_tool_end(
        self,
        *,
        run_id: str,
        turn_index: int,
        call_id: str,
        tool_name: str,
        result: ToolResult,
        duration_ms: float,
        is_error: bool,
    ) -> None:
        status = "ERROR" if is_error else "ok"
        _log.log(
            self._level,
            "[%s] Turn %d tool %s: %s duration=%.0fms",
            run_id,
            turn_index,
            tool_name,
            status,
            duration_ms,
        )


class MultiHooks(Hooks):
    """Fan-out to multiple ``Hooks`` instances.

    Example::

        hooks = MultiHooks(LoggingHooks(), OTelHooks())
        agent = Agent(provider=p, hooks=hooks)
    """

    def __init__(self, *hooks: Hooks) -> None:
        self._hooks = list(hooks)

    def on_agent_start(self, *, run_id: str, model: str, provider: str) -> None:
        for h in self._hooks:
            h.on_agent_start(run_id=run_id, model=model, provider=provider)

    def on_agent_end(
        self,
        *,
        run_id: str,
        turn_count: int,
        duration_ms: float,
        usage: TokenUsage | None,
    ) -> None:
        for h in self._hooks:
            h.on_agent_end(
                run_id=run_id, turn_count=turn_count, duration_ms=duration_ms, usage=usage
            )

    def on_turn_start(self, *, run_id: str, turn_index: int) -> None:
        for h in self._hooks:
            h.on_turn_start(run_id=run_id, turn_index=turn_index)

    def on_turn_end(
        self,
        *,
        run_id: str,
        turn_index: int,
        message: Message,
        tool_results: list[Message],
        llm_duration_ms: float,
        duration_ms: float,
    ) -> None:
        for h in self._hooks:
            h.on_turn_end(
                run_id=run_id,
                turn_index=turn_index,
                message=message,
                tool_results=tool_results,
                llm_duration_ms=llm_duration_ms,
                duration_ms=duration_ms,
            )

    def on_llm_start(self, *, run_id: str, turn_index: int) -> None:
        for h in self._hooks:
            h.on_llm_start(run_id=run_id, turn_index=turn_index)

    def on_llm_end(
        self,
        *,
        run_id: str,
        turn_index: int,
        message: Message,
        duration_ms: float,
    ) -> None:
        for h in self._hooks:
            h.on_llm_end(
                run_id=run_id, turn_index=turn_index, message=message, duration_ms=duration_ms
            )

    def on_tool_start(
        self,
        *,
        run_id: str,
        turn_index: int,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        for h in self._hooks:
            h.on_tool_start(
                run_id=run_id,
                turn_index=turn_index,
                call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
            )

    def on_tool_end(
        self,
        *,
        run_id: str,
        turn_index: int,
        call_id: str,
        tool_name: str,
        result: ToolResult,
        duration_ms: float,
        is_error: bool,
    ) -> None:
        for h in self._hooks:
            h.on_tool_end(
                run_id=run_id,
                turn_index=turn_index,
                call_id=call_id,
                tool_name=tool_name,
                result=result,
                duration_ms=duration_ms,
                is_error=is_error,
            )
