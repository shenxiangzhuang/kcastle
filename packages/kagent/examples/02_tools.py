"""Example 02 — Tools (Level 2: Agent)

Shows how to define typed tools and observe the full event stream:
  - ToolExecStart / ToolExecEnd  show when a tool is called and what it returned
  - StreamChunk(TextDeltaEvent)  streams text in real time for every turn
  - TurnEnd(message) with `not msg.tool_calls` marks the final answer turn;
    intermediate turns that issued tool calls are skipped to avoid duplication

Run:
    export DEEPSEEK_API_KEY=sk-...
    uv run python examples/02_tools.py
"""

import asyncio
import os

from kai import Anthropic, Tool, ToolResult
from kai.event import TextDeltaEvent
from pydantic import BaseModel, Field

from kagent import Agent, AgentError, StreamChunk, ToolExecEnd, ToolExecStart, TurnEnd


def make_provider() -> Anthropic:
    # return Anthropic(model="claude-sonnet-4-20250514")
    # return OpenAICompletions(model="gpt-4o")
    return Anthropic(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/anthropic",
    )


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


class GetWeather(Tool):
    """Return the current weather for a city.

    Subclass Tool, define inner Params(BaseModel) for typed + auto-schema,
    and override execute() with the real implementation.
    """

    name: str = "get_weather"
    description: str = "Get the current weather for a given city."

    class Params(BaseModel):
        city: str = Field(description="City name, e.g. 'Tokyo'")
        unit: str = Field(default="celsius", description="'celsius' or 'fahrenheit'")

    async def execute(self, params: "GetWeather.Params") -> ToolResult:
        # Stub — replace with a real weather API call.
        return ToolResult(output=f"Sunny, 22°{params.unit[0].upper()} in {params.city}")


class SearchWeb(Tool):
    """Quick web search (stub)."""

    name: str = "search_web"
    description: str = "Search the web for up-to-date information."

    class Params(BaseModel):
        query: str = Field(description="Search query")

    async def execute(self, params: "SearchWeb.Params") -> ToolResult:
        return ToolResult(output=f"Top result for '{params.query}': <stub content>")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    agent = Agent(
        provider=make_provider(),
        system="You are a helpful assistant. Use tools when needed.",
        tools=[GetWeather(), SearchWeb()],
    )

    print(">>> What's the weather in Tokyo and Paris?\n")

    # agent.run() yields AgentEvents — handle whichever you care about.
    async for event in agent.run("What's the weather in Tokyo and Paris?"):
        match event:
            case StreamChunk(event=e) if isinstance(e, TextDeltaEvent):
                # Stream text deltas in real time (fires for every turn).
                print(e.delta, end="", flush=True)

            case ToolExecStart(tool_name=name, arguments=args):
                # Print a newline first so tool lines don't run into streamed text.
                print(f"\n[tool call]  {name}({args})")

            case ToolExecEnd(tool_name=name, result=result, is_error=err):
                status = "ERROR" if err else "ok"
                print(f"[tool result/{status}] {name} → {result.output}")

            case TurnEnd(message=msg) if not msg.tool_calls:
                # Guard: only print the *final* turn.
                # Intermediate turns (the ones that issued tool calls) are skipped —
                # their text was already streamed above and they carry no final answer.
                print(f"\n[assistant] {msg.extract_text()}")

            case AgentError(error=err):
                print(f"\n[error] {err}")

            case _:
                pass
            

if __name__ == "__main__":
    asyncio.run(main())
