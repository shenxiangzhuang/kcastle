"""Example 03 — Multi-turn + Interactive Control (Level 2: Agent)

Demonstrates:
  1. Persistent conversation state across multiple complete() calls.
  2. agent.steer()  — inject a message mid-run to redirect the agent.
  3. agent.abort()  — cancel a running agent from another coroutine.

The Agent stores the full conversation history in agent.state.messages,
so each call to complete() or run() sees all previous turns.

Run:
    export DEEPSEEK_API_KEY=sk-...
    uv run python examples/03_multi_turn.py             # all demos
    uv run python examples/03_multi_turn.py multi_turn   # demo 1 only
    uv run python examples/03_multi_turn.py steer        # demo 2 only
    uv run python examples/03_multi_turn.py abort        # demo 3 only
"""

import asyncio
import os
import sys

from kai import Anthropic, Message

from kagent import Agent, AgentAbort, AgentEnd, AgentError, AgentEvent


def make_provider() -> Anthropic:
    # return Anthropic(model="claude-sonnet-4-20250514")
    # return OpenAICompletions(model="gpt-4o")
    return Anthropic(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/anthropic",
    )


# ---------------------------------------------------------------------------
# 1. Persistent multi-turn conversation
# ---------------------------------------------------------------------------


async def demo_multi_turn() -> None:
    print("=" * 60)
    print("Demo 1: multi-turn conversation")
    print("=" * 60)

    agent = Agent(
        provider=make_provider(),
        system="You are a helpful assistant with a good memory.",
    )

    # First turn — store a fact.
    await agent.complete("Remember: my favourite colour is indigo.")
    print(f"[turn 1] Messages in history: {len(agent.state.messages)}")

    # Second turn — recall it. The agent sees the full history.
    reply = await agent.complete("What is my favourite colour?")
    print(f"[turn 2] {reply.extract_text()}")

    # You can inspect or modify history directly.
    print(f"\nFull conversation ({len(agent.state.messages)} messages):")
    for msg in agent.state.messages:
        print(f"  [{msg.role}] {str(msg.content)}")

    print()


# ---------------------------------------------------------------------------
# 2. Steering — redirect the agent mid-run
# ---------------------------------------------------------------------------


async def demo_steer() -> None:
    print("=" * 60)
    print("Demo 2: steer()")
    print("=" * 60)

    agent = Agent(provider=make_provider(), system="You are a helpful assistant.")

    # Schedule a steering message before the run starts.
    # It will be injected after the current tool (if any) finishes,
    # redirecting the loop without starting a brand-new run.
    agent.steer(
        Message(role="user", content="Actually, answer in Chinese with short explain instead.")
    )

    reply = await agent.complete("What is 1 + 2 + ... + 9 ?")
    print(f"[steered reply] {reply.extract_text()}")

    print()


# ---------------------------------------------------------------------------
# 3. Abort — cancel a running agent from another coroutine
# ---------------------------------------------------------------------------


async def demo_abort() -> None:
    print("=" * 60)
    print("Demo 3: abort()")
    print("=" * 60)

    agent = Agent(provider=make_provider(), system="You are a helpful assistant.")

    async def cancel_after(delay: float) -> None:
        await asyncio.sleep(delay)
        print("[aborting agent…]")
        agent.abort()

    # Fire-and-forget the canceller.
    asyncio.create_task(cancel_after(3))

    events: list[AgentEvent] = []
    async for event in agent.run("Write me a very long essay about the history of computing."):
        events.append(event)
        if isinstance(event, AgentAbort):
            print(f"[aborted] collected {len(events)} events before abort")
        elif isinstance(event, AgentEnd):
            print(f"[agent ended] {len(event.messages)} messages in history")
        elif isinstance(event, AgentError):
            print(f"[error] {event.error}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "multi_turn"):
        await demo_multi_turn()
    if mode in ("all", "steer"):
        await demo_steer()
    if mode in ("all", "abort"):
        await demo_abort()


if __name__ == "__main__":
    asyncio.run(main())
