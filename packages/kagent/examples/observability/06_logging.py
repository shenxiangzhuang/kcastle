"""Example 06 — stdlib Logging

Demonstrates the first layer of observability: **stdlib logging**.

Every internal module in kai and kagent emits structured log messages via
the standard ``logging`` module.  No extra dependencies are needed — just
configure a handler and set the level.

Log hierarchy:
  - ``kai.stream``    — LLM streaming: start, completion, errors, timing
  - ``kagent.step``   — tool execution: warnings, errors, duration
  - ``kagent.loop``   — agent loop: start, end, turn count, total duration
  - ``kagent.agent``  — Agent.run/complete calls

Run:
    export DEEPSEEK_API_KEY=...
    uv run python examples/observability/06_logging.py
"""

import asyncio
import logging
import os

from kai import AnthropicMessages, Tool, ToolResult
from pydantic import BaseModel, Field

from kagent import Agent

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
    # Only configure kai.* and kagent.* loggers — third-party libs stay silent.
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)-16s %(levelname)-5s %(message)s", datefmt="%H:%M:%S")
    )

    for name in ("kai", "kagent"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    agent = Agent(
        llm=make_provider(),
        system="You are a helpful assistant. Use tools when needed. Be concise.",
        tools=TOOLS,
    )

    reply = await agent.complete(QUESTION)
    print(f"\n>>> Answer: {reply.extract_text()}\n")


if __name__ == "__main__":
    asyncio.run(main())
