"""kcastle.channels — Channel protocol and adapters.

A ``Channel`` is a communication adapter that connects the agent to a
platform (CLI, Telegram, etc.).  Each channel receives input, resolves
a session, calls ``session.run()``, and renders ``AgentEvent``s for its
platform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from kcastle.castle import Castle


class Channel(Protocol):
    """Protocol for kcastle communication channels."""

    @property
    def name(self) -> str:
        """Human-readable channel name (e.g. ``'cli'``, ``'telegram'``)."""
        ...

    async def start(self, castle: Castle) -> None:
        """Start the channel.  Called by ``Castle.run()``.

        This method should block until the channel is done (e.g. user exits
        the CLI, or the Telegram bot is shut down).
        """
        ...

    async def stop(self) -> None:
        """Stop the channel gracefully.  Called by ``Castle.shutdown()``."""
        ...


__all__ = ["Channel"]
