"""Example 05 — Context Builders

Demonstrates the built-in ContextBuilder implementations:

1. DefaultBuilder     — pass-through (the implicit default)
2. SlidingWindowBuilder — keep first message + last N messages
3. CompactingBuilder    — summarize older messages via LLM
4. AdaptiveBuilder      — let the agent switch strategies at runtime

Run:
    export MINIMAX_API_KEY=...
    uv run python examples/advanced/05_context_builders.py
"""

import asyncio
import os

from kai import AnthropicMessages

from kagent import (
    AdaptiveBuilder,
    Agent,
    AgentRuntime,
    AgentState,
    CompactingBuilder,
    DefaultBuilder,
    SlidingWindowBuilder,
    TurnEnd,
    UserInput,
    complete,
)


def make_provider() -> AnthropicMessages:
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
# 1. SlidingWindowBuilder — keep recent context only
# ---------------------------------------------------------------------------


async def demo_sliding_window() -> None:
    print("=== SlidingWindowBuilder ===\n")

    provider = make_provider()
    agent = Agent(
        llm=provider,
        system="You are concise. Reply in one sentence.",
        context_builder=SlidingWindowBuilder(window_size=6),
    )

    # Shared state so conversation history persists across turns
    state = AgentState(system=agent.system, tools=list(agent.tools))

    # Simulate a multi-turn conversation
    for question in [
        "My name is Alice.",
        "What is 2 + 2?",
        "What's the capital of France?",
        "What's my name?",  # May not remember — depends on window
    ]:
        print(f"[user] {question}")
        msg = await complete(agent, question, state=state)
        print(f"[assistant] {msg.extract_text()}\n")


# ---------------------------------------------------------------------------
# 2. AdaptiveBuilder — agent-controlled context switching
# ---------------------------------------------------------------------------


async def demo_adaptive() -> None:
    print("\n=== AdaptiveBuilder (agent-controlled) ===\n")

    provider = make_provider()

    adaptive = AdaptiveBuilder(
        builders={
            "full": DefaultBuilder(),
            "window": SlidingWindowBuilder(window_size=6),
            "compact": CompactingBuilder(provider, threshold=10, max_preserved=4),
        },
        default="full",
    )

    agent = Agent(
        llm=provider,
        system=(
            "You are a helpful assistant. You have a tool to switch context strategies. "
            "When the conversation grows long, consider switching to 'compact' or 'window' "
            "strategy to manage context. Current strategy: full."
        ),
        context_builder=adaptive,
        tools=[adaptive.create_tool()],
    )

    print(f"[context strategy] {adaptive.current}")

    runtime = AgentRuntime(agent, can_spawn=False)
    await runtime.start()

    try:
        for question in [
            "Remember: my favorite color is blue.",
            "Switch to the 'window' context strategy.",
            "What is my favorite color?",
        ]:
            print(f"[user] {question}")
            async for event in runtime.send(UserInput(question)):
                if isinstance(event, TurnEnd) and event.message.extract_text():
                    print(f"[assistant] {event.message.extract_text()}")
            print(f"[context strategy] {adaptive.current}\n")
    finally:
        await runtime.stop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    await demo_sliding_window()
    await demo_adaptive()


if __name__ == "__main__":
    asyncio.run(main())
