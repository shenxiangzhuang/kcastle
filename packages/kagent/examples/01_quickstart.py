"""Example 01 — Quickstart (Level 2: Agent)

The simplest way to run an agent: create an Agent, call complete().
No tools, no streaming — just a single question and answer.

Run:
    export DEEPSEEK_API_KEY=sk-...
    uv run python examples/01_quickstart.py
"""

import asyncio
import os

from kai import Anthropic

from kagent import Agent


def make_provider() -> Anthropic:
    # return Anthropic(model="claude-sonnet-4-20250514")
    # return OpenAICompletions(model="gpt-4o")
    return Anthropic(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/anthropic",
    )


async def main() -> None:
    agent = Agent(
        provider=make_provider(),
        system="You are a concise assistant. Answer in one sentence.",
    )

    # complete() runs the agent and returns the final assistant message.
    reply = await agent.complete("What is the capital of France?")
    print(reply.extract_text())


if __name__ == "__main__":
    asyncio.run(main())
