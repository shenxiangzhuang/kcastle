"""Example 01 — Quickstart (Level 2: Agent)

Create an Agent with tools, ask a question, get an answer.
The agent automatically calls tools when needed and loops until done.

Run:
    export MINIMAX_API_KEY=...
    uv run python examples/basics/01_quickstart.py
"""

import asyncio
import os

from kai import AnthropicMessages, Tool, ToolResult
from pydantic import BaseModel, Field

from kagent import Agent, complete


def make_provider() -> AnthropicMessages:
    # return AnthropicMessages(model="claude-sonnet-4-20250514")
    # return OpenAICompletions(model="gpt-4o")
    # return AnthropicMessages(
    #     model="deepseek-chat",
    #     api_key=os.environ.get("DEEPSEEK_API_KEY"),
    #     base_url="https://api.deepseek.com/anthropic",
    # )
    return AnthropicMessages(
        model="MiniMax-M2.5",
        api_key=os.environ.get("MINIMAX_API_KEY"),
        base_url="https://api.minimaxi.com/anthropic",
    )


# ---------------------------------------------------------------------------
# Tools — subclass Tool, define Params, implement execute()
# ---------------------------------------------------------------------------


class GetWeather(Tool):
    name: str = "get_weather"
    description: str = "Get the current weather for a given city."

    class Params(BaseModel):
        city: str = Field(description="City name, e.g. 'Tokyo'")

    async def execute(self, params: "GetWeather.Params") -> ToolResult:
        # Stub — replace with a real API call.
        return ToolResult(output=f"Sunny, 22°C in {params.city}")


class SearchWeb(Tool):
    name: str = "search_web"
    description: str = "Search the web for up-to-date information."

    class Params(BaseModel):
        query: str = Field(description="Search query")

    async def execute(self, params: "SearchWeb.Params") -> ToolResult:
        return ToolResult(output=f"Top result for '{params.query}': <stub>")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    agent = Agent(
        llm=make_provider(),
        system="You are a helpful assistant. Use tools when needed.",
        tools=[GetWeather(), SearchWeb()],
    )

    # complete() runs the full agent loop and returns the final message.
    reply = await complete(agent, "What's the weather in Tokyo and Paris?")
    print(reply.extract_text())


if __name__ == "__main__":
    asyncio.run(main())
