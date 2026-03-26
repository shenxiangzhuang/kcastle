"""Example 08 — OTel Trace (built-in OTelHooks)

Demonstrates the third layer of observability: **OpenTelemetry tracing**
via the built-in ``OTelHooks``.

``OTelHooks`` emits hierarchical spans following ``gen_ai.*`` semantic
conventions.  Spans are exported to Jaeger via OTLP for visualization.

Trace tree structure:
  - ``agent.run``      — top-level run with total duration and tokens
  - ``agent.turn``     — each LLM turn
  - ``gen_ai.chat``    — the LLM streaming call within a turn
  - ``agent.tool.*``   — each tool execution

Prerequisites:
    uv add kagent[otel]           # install OTel dependencies
    docker run -d --name jaeger -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:1.64.0

Run:
    export DEEPSEEK_API_KEY=...
    uv run python examples/observability/08_trace.py

Then open http://localhost:16686 to view traces in Jaeger.
"""

import asyncio
import os

from kai import AnthropicMessages, Tool, ToolResult
from pydantic import BaseModel, Field

from kagent import Agent, complete

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
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 60)
    print("OTel Trace (built-in OTelHooks)")
    print("=" * 60)

    import opentelemetry.trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    # --- Configure TracerProvider with OTLP exporter → Jaeger ---
    resource = Resource.create({"service.name": "kagent-demo"})
    tp = TracerProvider(resource=resource)
    tp.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True))
    )
    opentelemetry.trace.set_tracer_provider(tp)

    print("  → Exporting to Jaeger via OTLP (http://localhost:4317)")
    print("  → View traces at http://localhost:16686")

    # --- Use the built-in OTelHooks ---
    from kagent.otel import OTelHooks

    agent = Agent(
        llm=make_provider(),
        system="You are a helpful assistant. Use tools when needed. Be concise.",
        tools=TOOLS,
        hooks=OTelHooks(),
    )

    reply = await complete(agent, QUESTION)
    print(f"\n>>> Answer: {reply.extract_text()}")

    tp.force_flush()
    print("\n  Spans sent to Jaeger — open http://localhost:16686 to view\n")


if __name__ == "__main__":
    asyncio.run(main())
