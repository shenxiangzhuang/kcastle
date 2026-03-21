# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false
"""OpenTelemetry hooks for kagent.

``OTelHooks`` maps agent lifecycle events to OpenTelemetry spans following
the GenAI semantic conventions (v1.40.0+).

Requires ``opentelemetry-api`` as an optional dependency::

    pip install opentelemetry-api

Usage::

    from kagent import Agent
    from kagent.otel import OTelHooks

    agent = Agent(provider=p, hooks=OTelHooks(record_inputs=True))

Span hierarchy::

    invoke_agent {agent_name}
    +-- agent.turn: read_file, run_bash
    |   +-- chat deepseek-reasoner      (SpanKind.CLIENT)
    |   +-- execute_tool read_file      (SpanKind.INTERNAL)
    |   +-- execute_tool run_bash       (SpanKind.INTERNAL)
    +-- agent.turn (stop)
    |   +-- chat deepseek-reasoner
    +-- ...

All spans are annotated with ``gen_ai.*`` attributes per the OTel GenAI
semantic conventions.  Input / output recording is **off** by default to
avoid capturing sensitive data (``record_inputs`` / ``record_outputs``).
"""

from __future__ import annotations

import json
from typing import Any

from kai import Context, Message, Tool, ToolResult
from kai.types.usage import TokenUsage

from kagent.hooks import Hooks


def _get_otel_trace() -> Any:
    """Import and return the ``opentelemetry.trace`` module.

    Raises ``ImportError`` with a helpful message if not installed.
    """
    try:
        import opentelemetry.trace as _trace
    except ImportError:
        raise ImportError(
            "opentelemetry-api is required for OTelHooks. "
            "Install it with: pip install opentelemetry-api"
        ) from None
    result: Any = _trace
    return result


def _get_otel_logger(name: str) -> Any | None:
    """Try to get an OTel Logger for emitting log records.

    Returns ``None`` if the OTel Logs API is not installed.
    """
    try:
        from opentelemetry._logs import get_logger_provider

        return get_logger_provider().get_logger(name)
    except (ImportError, AttributeError):
        return None


class OTelHooks(Hooks):
    """OpenTelemetry instrumentation hooks for kagent.

    Creates spans following the GenAI semantic conventions.

    Args:
        tracer_name: Name for the OTel tracer (default: ``"kagent"``).
        tracer_version: Version string for the tracer.
        record_inputs: Record input messages / tool arguments.  Default
            ``False`` to avoid recording sensitive data (per spec SHOULD NOT).
        record_outputs: Record output messages / tool results.  Default
            ``False`` to avoid recording sensitive data (per spec SHOULD NOT).

    Example::

        from kagent.otel import OTelHooks

        hooks = OTelHooks(record_inputs=True, record_outputs=True)
        agent = Agent(provider=p, hooks=hooks)
    """

    def __init__(
        self,
        *,
        tracer_name: str = "kagent",
        tracer_version: str = "",
        record_inputs: bool = False,
        record_outputs: bool = False,
    ) -> None:
        otel_trace = _get_otel_trace()
        self._otel_trace: Any = otel_trace
        self._tracer: Any = otel_trace.get_tracer(tracer_name, tracer_version)
        self._logger: Any | None = _get_otel_logger(tracer_name)
        self._record_inputs = record_inputs
        self._record_outputs = record_outputs

        # Active spans keyed by composite string keys.
        self._agent_spans: dict[str, Any] = {}
        self._turn_spans: dict[str, Any] = {}
        self._llm_spans: dict[str, Any] = {}
        self._tool_spans: dict[str, Any] = {}
        # Per-run GenAI attributes (provider + model).
        self._run_genai: dict[str, dict[str, str]] = {}
        # Per-turn tool names collected during the turn for span naming.
        self._turn_tools: dict[str, list[str]] = {}
        # Last finish reason per run (for invoke_agent span).
        self._run_finish_reason: dict[str, str] = {}

    @staticmethod
    def _turn_key(run_id: str, turn_index: int) -> str:
        return f"{run_id}:{turn_index}"

    @staticmethod
    def _tool_key(run_id: str, turn_index: int, call_id: str) -> str:
        return f"{run_id}:{turn_index}:{call_id}"

    def _start_child_span(
        self,
        name: str,
        parent: Any,
        attributes: dict[str, Any],
        *,
        kind: Any = None,
    ) -> Any:
        ctx = self._otel_trace.set_span_in_context(parent) if parent else None
        kwargs: dict[str, Any] = {"context": ctx, "attributes": attributes}
        if kind is not None:
            kwargs["kind"] = kind
        return self._tracer.start_span(name, **kwargs)

    def _emit_log(self, body: str, span: Any) -> None:
        """Emit an OTel log record linked to the given span's trace context."""
        if self._logger is None:
            return
        try:
            from opentelemetry._logs import LogRecord, SeverityNumber
            from opentelemetry.trace.propagation import set_span_in_context

            ctx = set_span_in_context(span)
            record = LogRecord(
                body=body,
                severity_number=SeverityNumber.INFO,
                context=ctx,
            )
            self._logger.emit(record)
        except (ImportError, AttributeError):
            pass

    def _emit_input_log(self, context: Context, span: Any) -> None:
        """Emit input messages as a log record (JSON array) for conversation replay."""
        messages: list[dict[str, str]] = []
        if context.system:
            messages.append({"role": "system", "content": context.system})
        for msg in context.messages:
            text = msg.extract_text()
            if text:
                messages.append({"role": msg.role, "content": text})
        if messages:
            self._emit_log(json.dumps(messages, ensure_ascii=False), span)

    def _emit_output_log(self, message: Message, span: Any) -> None:
        """Emit completion output as a log record for conversation replay."""
        text = message.extract_text()
        if not text:
            return
        body = {
            "finish_reason": message.stop_reason or "stop",
            "message": {"role": "assistant", "content": text},
        }
        self._emit_log(json.dumps(body, ensure_ascii=False), span)

    # -- agent lifecycle --------------------------------------------------

    def on_agent_start(
        self,
        *,
        run_id: str,
        model: str,
        provider: str,
        agent_name: str | None = None,
        agent_id: str | None = None,
        agent_description: str | None = None,
        conversation_id: str | None = None,
        system: str | None = None,
        tools: list[Tool] | None = None,
    ) -> None:
        self._run_genai[run_id] = {
            # TODO: gen_ai.system is deprecated; keep for TMA1 compat only.
            "gen_ai.system": provider,
            "gen_ai.provider.name": provider,
            "gen_ai.request.model": model,
        }

        span_name = f"invoke_agent {agent_name}" if agent_name else "invoke_agent"
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.provider.name": provider,
            "gen_ai.request.model": model,
        }
        if agent_name:
            attrs["gen_ai.agent.name"] = agent_name
        if agent_id:
            attrs["gen_ai.agent.id"] = agent_id
        if agent_description:
            attrs["gen_ai.agent.description"] = agent_description
        if conversation_id:
            attrs["gen_ai.conversation.id"] = conversation_id
        if self._record_inputs and system:
            attrs["gen_ai.system_instructions"] = system

        span = self._tracer.start_span(
            span_name,
            kind=self._otel_trace.SpanKind.INTERNAL,
            attributes=attrs,
        )
        self._agent_spans[run_id] = span

    def on_agent_end(
        self,
        *,
        run_id: str,
        turn_count: int,
        duration_ms: float,
        usage: TokenUsage | None,
        is_error: bool = False,
        error_type: str | None = None,
    ) -> None:
        self._run_genai.pop(run_id, None)
        self._close_dangling_spans(run_id)

        span = self._agent_spans.pop(run_id, None)
        if span is None:
            return

        finish_reason = self._run_finish_reason.pop(run_id, None)
        if finish_reason:
            span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])
        if usage:
            span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
            if usage.cache_read_tokens:
                span.set_attribute("gen_ai.usage.cache_read.input_tokens", usage.cache_read_tokens)
            if usage.cache_write_tokens:
                span.set_attribute(
                    "gen_ai.usage.cache_creation.input_tokens", usage.cache_write_tokens
                )
        if is_error:
            if error_type:
                span.set_attribute("error.type", error_type)
            span.set_status(self._otel_trace.StatusCode.ERROR, error_type or "AgentError")
        span.end()

    def _close_dangling_spans(self, run_id: str) -> None:
        """Close any LLM or turn spans left open by an error path."""
        prefix = f"{run_id}:"
        for key in [k for k in self._llm_spans if k.startswith(prefix)]:
            span = self._llm_spans.pop(key)
            span.set_attribute("error.type", "AgentError")
            span.set_status(self._otel_trace.StatusCode.ERROR, "Agent terminated")
            span.end()
        for key in [k for k in self._tool_spans if k.startswith(prefix)]:
            span = self._tool_spans.pop(key)
            span.set_attribute("error.type", "AgentError")
            span.set_status(self._otel_trace.StatusCode.ERROR, "Agent terminated")
            span.end()
        self._turn_tools.pop(next((k for k in self._turn_tools if k.startswith(prefix)), ""), None)
        for key in [k for k in self._turn_spans if k.startswith(prefix)]:
            span = self._turn_spans.pop(key)
            span.set_attribute("error.type", "AgentError")
            span.set_status(self._otel_trace.StatusCode.ERROR, "Agent terminated")
            span.end()

    # -- turn lifecycle ---------------------------------------------------

    def on_turn_start(self, *, run_id: str, turn_index: int) -> None:
        key = self._turn_key(run_id, turn_index)
        self._turn_tools[key] = []
        parent = self._agent_spans.get(run_id)
        span = self._start_child_span("agent.turn", parent, {})
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

        span.end()

    # -- LLM call ---------------------------------------------------------

    def on_llm_start(self, *, run_id: str, turn_index: int, context: Context) -> None:
        genai = self._run_genai.get(run_id, {})
        model = genai.get("gen_ai.request.model", "")
        span_name = f"chat {model}" if model else "chat"

        key = self._turn_key(run_id, turn_index)
        parent = self._turn_spans.get(key)
        otel_ctx = self._otel_trace.set_span_in_context(parent) if parent else None
        span = self._tracer.start_span(
            span_name,
            context=otel_ctx,
            kind=self._otel_trace.SpanKind.CLIENT,
            attributes={
                **genai,
                "gen_ai.operation.name": "chat",
            },
        )

        # Record input messages as span events per GenAI conventions.
        if self._record_inputs:
            if context.system:
                span.add_event("gen_ai.system.message", {"content": context.system})
            for msg in context.messages:
                text = msg.extract_text()
                if not text:
                    continue
                if msg.role == "user":
                    span.add_event("gen_ai.user.message", {"content": text})
                elif msg.role == "assistant":
                    span.add_event("gen_ai.assistant.message", {"content": text})

            # Emit input messages as an OTel log record for TMA1 conversation replay.
            self._emit_input_log(context, span)

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

        if message.stop_reason:
            span.set_attribute("gen_ai.response.finish_reasons", [message.stop_reason])
            self._run_finish_reason[run_id] = message.stop_reason
        if message.usage:
            span.set_attribute("gen_ai.usage.input_tokens", message.usage.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", message.usage.output_tokens)
            if message.usage.cache_read_tokens:
                span.set_attribute(
                    "gen_ai.usage.cache_read.input_tokens", message.usage.cache_read_tokens
                )
            if message.usage.cache_write_tokens:
                span.set_attribute(
                    "gen_ai.usage.cache_creation.input_tokens",
                    message.usage.cache_write_tokens,
                )

        # Record output as gen_ai.choice event.
        if self._record_outputs:
            text = message.extract_text()
            if text:
                span.add_event(
                    "gen_ai.choice",
                    {"index": 0, "message.role": "assistant", "message.content": text},
                )
                # Emit completion as an OTel log record for TMA1 conversation replay.
                self._emit_output_log(message, span)

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
        tool_description: str | None = None,
    ) -> None:
        # Collect tool name for turn span naming.
        turn_key = self._turn_key(run_id, turn_index)
        tools = self._turn_tools.get(turn_key)
        if tools is not None:
            tools.append(tool_name)

        parent = self._turn_spans.get(turn_key)
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool_name,
            "gen_ai.tool.call.id": call_id,
            "gen_ai.tool.type": "function",
        }
        if tool_description:
            attrs["gen_ai.tool.description"] = tool_description
        if self._record_inputs:
            args_str = json.dumps(arguments, default=str)
            if len(args_str) <= 4096:
                attrs["gen_ai.tool.call.arguments"] = args_str
        span = self._start_child_span(
            f"execute_tool {tool_name}",
            parent,
            attrs,
            kind=self._otel_trace.SpanKind.INTERNAL,
        )
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
        if self._record_outputs:
            output = result.output
            if len(output) <= 4096:
                span.set_attribute("gen_ai.tool.call.result", output)
        if is_error:
            span.set_attribute("error.type", "ToolError")
            span.set_status(self._otel_trace.StatusCode.ERROR, result.output[:256])
        span.end()
