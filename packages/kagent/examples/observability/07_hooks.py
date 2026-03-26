"""Example 07 — Hooks (LoggingHooks, Custom Hooks, MultiHooks)

Demonstrates the second layer of observability: **Hooks** — structured
lifecycle callbacks that fire at every key point of an agent run.

Three approaches are shown:

  1. ``LoggingHooks`` — built-in, logs lifecycle events via stdlib logging.
  2. Custom ``Hooks`` subclass — collect your own metrics / telemetry.
  3. ``MultiHooks`` — compose multiple hooks together.

Run:
    export DEEPSEEK_API_KEY=...
    uv run python examples/observability/07_hooks.py                # all demos
    uv run python examples/observability/07_hooks.py logging_hooks   # demo 1
    uv run python examples/observability/07_hooks.py custom_hooks    # demo 2
"""

import asyncio
import logging
import os
import sys
from typing import Any

from kai import AnthropicMessages, Tool, ToolResult
from kai.types.usage import TokenUsage
from pydantic import BaseModel, Field

from kagent import Agent, Hooks, LoggingHooks, MultiHooks, complete

# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


def make_provider() -> AnthropicMessages:
    return AnthropicMessages(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/anthropic",
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class Calculator(Tool):
    name: str = "calculator"
    description: str = "Evaluate an arithmetic expression."

    class Params(BaseModel):
        expression: str = Field(description="e.g. '(3 + 5) * 2'")

    async def execute(self, params: "Calculator.Params") -> ToolResult:
        try:
            value = eval(params.expression, {"__builtins__": {}})  # noqa: S307
            return ToolResult(output=str(value))
        except Exception as e:
            return ToolResult.error(str(e))


class GetWeather(Tool):
    name: str = "get_weather"
    description: str = "Get the current weather for a city."

    class Params(BaseModel):
        city: str = Field(description="City name")

    async def execute(self, params: "GetWeather.Params") -> ToolResult:
        data = {"Tokyo": "Sunny 22°C", "Paris": "Cloudy 15°C", "NYC": "Rainy 10°C"}
        return ToolResult(output=data.get(params.city, f"Clear 20°C in {params.city}"))


TOOLS: list[Tool] = [Calculator(), GetWeather()]
QUESTION = "What's the weather in Tokyo? Also, compute 42 * 58."


# ---------------------------------------------------------------------------
# Demo 1 — LoggingHooks (built-in)
# ---------------------------------------------------------------------------


async def demo_logging_hooks() -> None:
    """Use ``LoggingHooks`` for structured lifecycle logging.

    ``LoggingHooks`` logs at INFO level by default:
      - Agent start/end with provider, model, duration, token usage
      - Turn summaries with LLM time, tool count
      - Tool execution status and duration

    Set ``level=logging.DEBUG`` to also see turn/LLM start events
    and tool arguments.
    """
    print("=" * 60)
    print("Demo 1: LoggingHooks (built-in)")
    print("=" * 60)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)-16s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    agent = Agent(
        llm=make_provider(),
        system="You are a helpful assistant. Use tools when needed. Be concise.",
        tools=TOOLS,
        hooks=LoggingHooks(level=logging.INFO),
    )

    reply = await complete(agent, QUESTION)
    print(f"\n>>> Answer: {reply.extract_text()}\n")

    logging.root.handlers.clear()
    logging.root.setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Demo 2 — Custom Hooks + MultiHooks
# ---------------------------------------------------------------------------


class MetricsHooks(Hooks):
    """A custom hooks implementation that collects metrics in-memory.

    This shows how to build your own observability by subclassing ``Hooks``
    and overriding only the methods you care about.
    """

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self._current_run: dict[str, Any] = {}

    def on_agent_start(self, *, run_id: str, model: str, provider: str, **kwargs: Any) -> None:
        self._current_run = {"run_id": run_id, "model": model, "provider": provider}

    def on_agent_end(
        self,
        *,
        run_id: str,
        turn_count: int,
        duration_ms: float,
        usage: TokenUsage | None,
        **kwargs: Any,
    ) -> None:
        self._current_run["turn_count"] = turn_count
        self._current_run["duration_ms"] = round(duration_ms, 1)
        if usage:
            self._current_run["input_tokens"] = usage.input_tokens
            self._current_run["output_tokens"] = usage.output_tokens
            self._current_run["total_tokens"] = usage.total_tokens
        self.runs.append(self._current_run)

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
        self.tool_calls.append(
            {
                "run_id": run_id,
                "turn": turn_index,
                "tool": tool_name,
                "duration_ms": round(duration_ms, 2),
                "error": is_error,
                "output_preview": result.output[:80],
            }
        )

    def print_report(self) -> None:
        print("\n--- Metrics Report ---")
        for run in self.runs:
            print(f"  Run {run['run_id']}:")
            print(f"    Model:     {run['model']}")
            print(f"    Turns:     {run.get('turn_count', '?')}")
            print(f"    Duration:  {run.get('duration_ms', '?')}ms")
            if "total_tokens" in run:
                print(f"    Tokens:    {run['input_tokens']} in / {run['output_tokens']} out")
        if self.tool_calls:
            print("  Tool calls:")
            for tc in self.tool_calls:
                status = "ERR" if tc["error"] else "ok"
                preview = tc["output_preview"]
                print(f"    [{status}] {tc['tool']} ({tc['duration_ms']}ms) → {preview}")
        print()


async def demo_custom_hooks() -> None:
    """Subclass ``Hooks`` for custom metrics collection.

    Combines ``LoggingHooks`` + ``MetricsHooks`` with ``MultiHooks``::

        hooks = MultiHooks(LoggingHooks(), MetricsHooks())
        agent = Agent(llm=p, hooks=hooks)
    """
    print("=" * 60)
    print("Demo 2: Custom Hooks + MultiHooks")
    print("=" * 60)

    metrics = MetricsHooks()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-16s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    hooks = MultiHooks(LoggingHooks(), metrics)

    agent = Agent(
        llm=make_provider(),
        system="You are a helpful assistant. Use tools when needed. Be concise.",
        tools=TOOLS,
        hooks=hooks,
    )

    reply = await complete(agent, QUESTION)
    print(f"\n>>> Answer: {reply.extract_text()}")

    metrics.print_report()

    logging.root.handlers.clear()
    logging.root.setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    demos = {
        "logging_hooks": demo_logging_hooks,
        "custom_hooks": demo_custom_hooks,
    }

    if len(sys.argv) > 1:
        name = sys.argv[1]
        if name not in demos:
            print(f"Unknown demo: {name}")
            print(f"Available: {', '.join(demos)}")
            sys.exit(1)
        await demos[name]()
    else:
        for demo_fn in demos.values():
            await demo_fn()


if __name__ == "__main__":
    asyncio.run(main())
