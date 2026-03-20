# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false
"""OpenTelemetry hooks for kagent.

``OTelHooks`` maps agent lifecycle events to OpenTelemetry spans following
the ``gen_ai.*`` semantic conventions.

Requires ``opentelemetry-api`` as an optional dependency::

    pip install opentelemetry-api

Usage::

    from kagent import Agent
    from kagent.otel import OTelHooks

    agent = Agent(provider=p, hooks=OTelHooks())

Span hierarchy::

    agent.run
    +-- agent.turn: read_file, run_bash
    |   +-- chat deepseek-reasoner
    |   +-- agent.tool.read_file
    |   +-- agent.tool.run_bash
    +-- agent.turn (stop)
    |   +-- chat deepseek-reasoner
    +-- ...

All spans are annotated with ``gen_ai.*`` attributes per the OpenTelemetry
GenAI semantic conventions where applicable.
"""

from __future__ import annotations

import json
from typing import Any

from kai import Message, ToolResult
from kai.types.usage import TokenUsage

from kagent.hooks import Hooks


def _get_otel_trace() -> Any:
    """Import and return the ``opentelemetry.trace`` module.

    Raises ``ImportError`` with a helpful message if not installed.
    """
    try:
        import opentelemetry.trace as _trace  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "opentelemetry-api is required for OTelHooks. "
            "Install it with: pip install opentelemetry-api"
        ) from None
    result: Any = _trace
    return result


class OTelHooks(Hooks):
    """OpenTelemetry instrumentation hooks for kagent.

    Creates spans following the ``gen_ai.*`` semantic conventions.

    Args:
        tracer_name: Name for the OTel tracer (default: ``"kagent"``).
        tracer_version: Version string for the tracer.
        record_inputs: Whether to record input messages/arguments as span
            attributes.  Set to ``False`` to avoid recording sensitive data.
        record_outputs: Whether to record output messages/results as span
            attributes.  Set to ``False`` to avoid recording sensitive data.

    Example::

        from kagent.otel import OTelHooks

        hooks = OTelHooks(record_inputs=False)  # No sensitive data in spans
        agent = Agent(provider=p, hooks=hooks)
    """

    def __init__(
        self,
        *,
        tracer_name: str = "kagent",
        tracer_version: str = "",
        record_inputs: bool = True,
        record_outputs: bool = True,
    ) -> None:
        otel_trace = _get_otel_trace()
        self._otel_trace: Any = otel_trace
        self._tracer: Any = otel_trace.get_tracer(tracer_name, tracer_version)
        self._record_inputs = record_inputs
        self._record_outputs = record_outputs

        # Active spans keyed by composite string keys.
        self._agent_spans: dict[str, Any] = {}
        self._turn_spans: dict[str, Any] = {}
        self._llm_spans: dict[str, Any] = {}
        self._tool_spans: dict[str, Any] = {}
        # Per-run context: GenAI identity and model name.
        self._run_genai: dict[str, dict[str, str]] = {}
        # Per-turn tool names collected during the turn for span naming.
        self._turn_tools: dict[str, list[str]] = {}

    @staticmethod
    def _turn_key(run_id: str, turn_index: int) -> str:
        return f"{run_id}:{turn_index}"

    @staticmethod
    def _tool_key(run_id: str, turn_index: int, call_id: str) -> str:
        return f"{run_id}:{turn_index}:{call_id}"

    def _start_child_span(self, name: str, parent: Any, attributes: dict[str, Any]) -> Any:
        ctx = self._otel_trace.set_span_in_context(parent) if parent else None
        return self._tracer.start_span(name, context=ctx, attributes=attributes)

    # -- agent lifecycle --------------------------------------------------

    def on_agent_start(self, *, run_id: str, model: str, provider: str) -> None:
        self._run_genai[run_id] = {
            "gen_ai.system": provider,
            "gen_ai.request.model": model,
        }
        span = self._tracer.start_span(
            "agent.run",
            attributes={"agent.run_id": run_id},
        )
        self._agent_spans[run_id] = span

    def on_agent_end(
        self,
        *,
        run_id: str,
        turn_count: int,
        duration_ms: float,
        usage: TokenUsage | None,
    ) -> None:
        self._run_genai.pop(run_id, None)
        span = self._agent_spans.pop(run_id, None)
        if span is None:
            return
        span.set_attribute("agent.turn_count", turn_count)
        span.set_attribute("agent.duration_ms", duration_ms)
        if usage:
            span.set_attribute("agent.usage.input_tokens", usage.input_tokens)
            span.set_attribute("agent.usage.output_tokens", usage.output_tokens)
            span.set_attribute("agent.usage.total_tokens", usage.total_tokens)
        span.end()

    # -- turn lifecycle ---------------------------------------------------

    def on_turn_start(self, *, run_id: str, turn_index: int) -> None:
        key = self._turn_key(run_id, turn_index)
        self._turn_tools[key] = []
        parent = self._agent_spans.get(run_id)
        span = self._start_child_span(
            "agent.turn",
            parent,
            {"agent.run_id": run_id, "agent.turn_index": turn_index},
        )
        self._turn_spans[key] = span

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
        key = self._turn_key(run_id, turn_index)
        span = self._turn_spans.pop(key, None)
        if span is None:
            return

        # Rename span: "agent.turn: tool1, tool2" or "agent.turn (stop)"
        tools = self._turn_tools.pop(key, [])
        if tools:
            span.update_name(f"agent.turn: {', '.join(tools)}")
        elif message.stop_reason:
            span.update_name(f"agent.turn ({message.stop_reason})")

        span.set_attribute("agent.turn.duration_ms", duration_ms)
        span.set_attribute("agent.turn.llm_duration_ms", llm_duration_ms)
        span.set_attribute("agent.turn.tool_count", len(tool_results))
        span.end()

    # -- LLM call ---------------------------------------------------------

    def on_llm_start(self, *, run_id: str, turn_index: int) -> None:
        genai = self._run_genai.get(run_id, {})
        model = genai.get("gen_ai.request.model", "")
        # OTel GenAI convention: span name = "{operation} {model}"
        span_name = f"chat {model}" if model else "chat"

        key = self._turn_key(run_id, turn_index)
        parent = self._turn_spans.get(key)
        span = self._start_child_span(
            span_name,
            parent,
            {
                **genai,
                "gen_ai.operation.name": "chat",
                "agent.run_id": run_id,
                "agent.turn_index": turn_index,
            },
        )
        self._llm_spans[key] = span

    def on_llm_end(
        self,
        *,
        run_id: str,
        turn_index: int,
        message: Message,
        duration_ms: float,
    ) -> None:
        key = self._turn_key(run_id, turn_index)
        span = self._llm_spans.pop(key, None)
        if span is None:
            return
        span.set_attribute("gen_ai.client.operation.duration_ms", duration_ms)
        if message.stop_reason:
            span.set_attribute("gen_ai.response.finish_reasons", [message.stop_reason])
        if message.usage:
            span.set_attribute("gen_ai.usage.input_tokens", message.usage.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", message.usage.output_tokens)
        span.end()

    # -- tool call --------------------------------------------------------

    def on_tool_start(
        self,
        *,
        run_id: str,
        turn_index: int,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        # Collect tool name for turn span naming.
        turn_key = self._turn_key(run_id, turn_index)
        tools = self._turn_tools.get(turn_key)
        if tools is not None:
            tools.append(tool_name)

        parent = self._turn_spans.get(turn_key)
        attrs: dict[str, Any] = {
            "agent.run_id": run_id,
            "agent.turn_index": turn_index,
            "tool.call_id": call_id,
            "tool.name": tool_name,
        }
        if self._record_inputs:
            args_str = json.dumps(arguments, default=str)
            if len(args_str) <= 4096:
                attrs["tool.arguments"] = args_str
        span = self._start_child_span(f"agent.tool.{tool_name}", parent, attrs)
        self._tool_spans[self._tool_key(run_id, turn_index, call_id)] = span

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
        tool_key = self._tool_key(run_id, turn_index, call_id)
        span = self._tool_spans.pop(tool_key, None)
        if span is None:
            return
        span.set_attribute("tool.duration_ms", duration_ms)
        span.set_attribute("tool.is_error", is_error)
        if self._record_outputs:
            output = result.output
            if len(output) <= 4096:
                span.set_attribute("tool.result.output", output)
        if is_error:
            span.set_status(self._otel_trace.StatusCode.ERROR, result.output[:256])
        span.end()
