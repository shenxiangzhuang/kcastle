"""Mutable agent state.

``AgentState`` is the single source of truth for everything that changes
during an agent run.  The loop appends entries to ``trace`` as the
conversation progresses.

The ``messages`` property provides backward-compatible access by
deriving a ``list[Message]`` from the trace.
"""

from dataclasses import dataclass, field

from kai import Message, Tool

from kagent.trace.trace import Trace


@dataclass
class AgentState:
    """The evolving conversation state of an agent.

    The trace is the single source of truth — the loop appends
    ``TraceEntry`` objects to it.  ``messages`` is a derived view.

    Example::

        state = AgentState(
            system="You are helpful.",
            tools=[my_tool],
        )
        # Trace is auto-created; messages derived from it:
        print(state.messages)  # []
    """

    system: str | None = None
    """System prompt."""

    trace: Trace = field(default_factory=Trace)
    """Append-only execution trace.  The single source of truth."""

    tools: list[Tool] = field(default_factory=lambda: list[Tool]())
    """Available tools for the agent."""

    @property
    def messages(self) -> list[Message]:
        """Conversation history derived from the trace (backward-compatible)."""
        return self.trace.messages()
