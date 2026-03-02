"""Context builders for agent loops.

A ``ContextBuilder`` protocol defines how ``AgentState`` is transformed into
a ``kai.Context`` snapshot before each LLM call. Built-in implementations
cover common patterns; users can implement the protocol for custom strategies.

Built-in builders:

- ``DefaultBuilder`` — pass-through, sends all messages.
- ``SlidingWindowBuilder`` — keeps the last *N* messages.
- ``CompactingBuilder`` — summarizes older messages via an LLM call.
- ``AdaptiveBuilder`` — delegates to a named builder; switchable at runtime.

The ``ContextSwitchTool`` lets the agent itself choose its context strategy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kai import LLM, Context, Message, Tool, ToolResult, complete
from kai.message import TextPart, ThinkPart
from pydantic import BaseModel, Field, PrivateAttr

from kagent.state import AgentState


@runtime_checkable
class ContextBuilder(Protocol):
    """Protocol for building a ``kai.Context`` from ``AgentState``.

    Any object with an ``async def build(self, state: AgentState) -> Context``
    method satisfies this protocol (structural typing — no inheritance needed).

    Example — custom builder::

        class MyBuilder:
            async def build(self, state: AgentState) -> Context:
                return Context(
                    system=state.system,
                    messages=state.messages[-10:],
                    tools=state.tools,
                )
    """

    async def build(self, state: AgentState) -> Context: ...


class DefaultBuilder:
    """Pass-through builder: sends all messages without modification.

    This is the implicit default when no ``context_builder`` is provided.

    Example::

        builder = DefaultBuilder()
        ctx = await builder.build(state)
    """

    async def build(self, state: AgentState) -> Context:
        return Context(
            system=state.system,
            messages=state.messages,
            tools=state.tools,
        )


class SlidingWindowBuilder:
    """Keep the first user message and the last *window_size* messages.

    The first message (typically the user's initial task/goal) is always
    preserved regardless of window size, so the agent never loses sight of
    its objective.

    Tool-result messages whose corresponding tool-call message is outside
    the window are dropped to avoid orphaned references.

    Args:
        window_size: Maximum number of recent messages to keep (default 20).

    Example::

        builder = SlidingWindowBuilder(window_size=10)
        agent = Agent(provider=p, system="…", context_builder=builder)
    """

    def __init__(self, window_size: int = 20) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size

    async def build(self, state: AgentState) -> Context:
        messages = state.messages
        if len(messages) <= self.window_size:
            return Context(system=state.system, messages=messages, tools=state.tools)

        # Always keep the first message (the user's initial goal)
        first = messages[0]
        tail = messages[-self.window_size :]

        # If the first message is already in the tail, no need to prepend
        selected = tail if tail[0] is first else [first, *tail]

        # Drop orphaned tool-result messages whose tool_call is outside the window
        selected = _drop_orphaned_tool_results(selected)

        return Context(system=state.system, messages=selected, tools=state.tools)


class CompactingBuilder:
    """Summarize older messages when the conversation grows too long.

    When ``len(state.messages)`` exceeds ``threshold``, older messages are
    compacted into a single summary via an LLM call. The most recent
    ``max_preserved`` messages are always kept verbatim.

    The summary is cached internally: only the newly accumulated prefix
    is re-summarized, so repeated calls on a growing conversation are
    efficient.

    Args:
        llm: LLM used for summarization.
        max_preserved: Number of recent messages to keep verbatim (default 6).
        threshold: Trigger compaction when message count exceeds this (default 20).
        summary_system: System prompt for the summarization call.

    Example::

        builder = CompactingBuilder(my_llm, threshold=30)
        agent = Agent(llm=my_llm, system="…", context_builder=builder)
    """

    _DEFAULT_SUMMARY_SYSTEM = (
        "You are a conversation summarizer. Summarize the following conversation "
        "into a concise but comprehensive summary. Preserve all key facts, decisions, "
        "tool results, and action items. Be concise."
    )

    def __init__(
        self,
        llm: LLM,
        *,
        max_preserved: int = 6,
        threshold: int = 20,
        summary_system: str | None = None,
    ) -> None:
        if max_preserved < 1:
            raise ValueError("max_preserved must be >= 1")
        if threshold < max_preserved + 1:
            raise ValueError("threshold must be > max_preserved")
        self._llm = llm
        self._max_preserved = max_preserved
        self._threshold = threshold
        self._summary_system = summary_system or self._DEFAULT_SUMMARY_SYSTEM
        # Cache: (compacted_prefix_length, summary_text)
        self._cache: tuple[int, str] | None = None

    async def build(self, state: AgentState) -> Context:
        messages = state.messages
        if len(messages) <= self._threshold:
            return Context(system=state.system, messages=messages, tools=state.tools)

        # Split into compact-target and preserved
        split_idx = len(messages) - self._max_preserved
        to_compact = messages[:split_idx]
        preserved = messages[split_idx:]

        # Check cache — only re-summarize if the prefix grew
        if self._cache is not None and self._cache[0] == len(to_compact):
            summary_text = self._cache[1]
        else:
            summary_text = await self._summarize(to_compact)
            self._cache = (len(to_compact), summary_text)

        summary_msg = Message(role="user", content=f"[Conversation summary]\n{summary_text}")
        return Context(
            system=state.system,
            messages=[summary_msg, *preserved],
            tools=state.tools,
        )

    async def _summarize(self, messages: list[Message]) -> str:
        """Call the LLM to summarize messages."""
        # Format messages for the summarization prompt
        lines: list[str] = []
        for msg in messages:
            text = _extract_text_no_think(msg)
            if text:
                lines.append(f"[{msg.role}] {text}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(f"[tool_call] {tc.name}({tc.arguments})")

        conversation_text = "\n".join(lines)
        ctx = Context(
            system=self._summary_system,
            messages=[Message(role="user", content=conversation_text)],
        )
        result = await complete(self._llm, ctx)
        return result.extract_text()


class AdaptiveBuilder:
    """Delegates to a named ``ContextBuilder``, switchable at runtime.

    Register multiple strategies and switch between them — either
    programmatically or via the ``ContextSwitchTool`` (which lets the
    agent itself choose its context strategy).

    Args:
        builders: A ``{name: builder}`` mapping.
        default: Name of the initial active builder. Must be a key in *builders*.

    Example::

        adaptive = AdaptiveBuilder(
            builders={
                "full": DefaultBuilder(),
                "window": SlidingWindowBuilder(window_size=10),
                "compact": CompactingBuilder(provider=p),
            },
            default="full",
        )
        agent = Agent(
            provider=p, system="…", context_builder=adaptive, tools=[adaptive.create_tool()]
        )
    """

    def __init__(self, builders: dict[str, ContextBuilder], *, default: str) -> None:
        if not builders:
            raise ValueError("builders must not be empty")
        if default not in builders:
            raise ValueError(f"default '{default}' not found in builders: {list(builders)}")
        self._builders = dict(builders)
        self._current = default

    @property
    def current(self) -> str:
        """Name of the currently active builder."""
        return self._current

    @property
    def available(self) -> list[str]:
        """Names of all registered builders."""
        return list(self._builders)

    def register(self, name: str, builder: ContextBuilder) -> None:
        """Register a new builder (or replace an existing one)."""
        self._builders[name] = builder

    def switch(self, name: str) -> None:
        """Switch to a registered builder by name.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in self._builders:
            raise KeyError(f"Unknown builder '{name}'. Available: {list(self._builders)}")
        self._current = name

    async def build(self, state: AgentState) -> Context:
        """Delegate to the currently active builder."""
        return await self._builders[self._current].build(state)

    def create_tool(self) -> ContextSwitchTool:
        """Create a tool that lets the agent switch context strategies.

        The returned tool should be added to the agent's tool list.

        Example::

            adaptive = AdaptiveBuilder(...)
            agent = Agent(..., tools=[adaptive.create_tool(), ...])
        """
        return ContextSwitchTool.for_builder(self)


class ContextSwitchTool(Tool):
    """A tool that lets the agent switch context building strategies.

    Created via ``AdaptiveBuilder.create_tool()`` — not instantiated directly.
    """

    name: str = "switch_context_strategy"
    description: str = "Switch the context building strategy for subsequent turns."

    class Params(BaseModel):
        strategy: str = Field(description="Name of the context strategy to switch to.")

    # Reference to the adaptive builder (excluded from schema serialization)
    _adaptive: AdaptiveBuilder = PrivateAttr()

    @classmethod
    def for_builder(cls, adaptive: AdaptiveBuilder) -> ContextSwitchTool:
        """Create a ContextSwitchTool bound to an AdaptiveBuilder."""
        available = ", ".join(adaptive.available)
        tool = cls(
            description=(
                f"Switch the context building strategy. "
                f"Available strategies: {available}. "
                f"Use this to manage conversation context — e.g., switch to a "
                f"compact strategy when the conversation grows long."
            ),
        )
        tool._adaptive = adaptive
        return tool

    async def execute(self, params: ContextSwitchTool.Params) -> ToolResult:
        try:
            self._adaptive.switch(params.strategy)
            return ToolResult(output=f"Context strategy switched to '{params.strategy}'.")
        except KeyError as e:
            return ToolResult.error(str(e))


def _drop_orphaned_tool_results(messages: list[Message]) -> list[Message]:
    """Remove tool-result messages whose tool_call_id has no matching tool_call."""
    # Collect all tool_call IDs present in assistant messages within the window
    valid_call_ids: set[str] = set()
    for msg in messages:
        if msg.tool_calls:
            for tc in msg.tool_calls:
                valid_call_ids.add(tc.id)

    return [
        msg
        for msg in messages
        if msg.role != "tool"
        or (msg.tool_call_id is not None and msg.tool_call_id in valid_call_ids)
    ]


def _extract_text_no_think(msg: Message) -> str:
    """Extract text parts, skipping ThinkPart content."""
    return "".join(
        part.text
        for part in msg.content
        if isinstance(part, TextPart) and not isinstance(part, ThinkPart)
    )
