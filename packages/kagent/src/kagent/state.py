"""Mutable agent state.

``AgentState`` is the single source of truth for everything that changes
during an agent run. The loop mutates it in-place — appending assistant
messages and tool results to ``messages`` as the conversation progresses.
"""

from dataclasses import dataclass, field

from kai import Message, Tool


@dataclass
class AgentState:
    """The evolving conversation state of an agent.

    This is mutable by design — the loop appends to ``messages``
    in-place. Users can also modify ``system`` or swap ``tools``
    between runs.

    Example::

        state = AgentState(
            system="You are helpful.",
            messages=[Message(role="user", content="Hello!")],
            tools=[my_tool],
        )
    """

    system: str | None = None
    """System prompt."""

    messages: list[Message] = field(default_factory=lambda: list[Message]())
    """Conversation history. Mutated in-place by the loop."""

    tools: list[Tool] = field(default_factory=lambda: list[Tool]())
    """Available tools for the agent."""
