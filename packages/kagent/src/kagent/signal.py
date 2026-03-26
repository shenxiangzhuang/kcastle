"""Typed signals for the agent runtime mailbox.

Signals are the messages that flow into ``AgentRuntime``'s mailbox.
Each signal has a ``type`` literal field for pattern matching, following
the same convention as ``AgentEvent``.

Control signals (abort, steer) bypass the mailbox as direct method calls
— you can't enqueue "cancel" behind the thing you want to cancel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kai import Message


@dataclass(frozen=True, slots=True)
class UserInput:
    """User sent a message to the agent."""

    text: str
    type: Literal["user_input"] = "user_input"


@dataclass(frozen=True, slots=True)
class ChildCompleted:
    """A sub-agent finished its task successfully."""

    child_id: str
    result: Message
    type: Literal["child_completed"] = "child_completed"


@dataclass(frozen=True, slots=True)
class ChildError:
    """A sub-agent failed with an error."""

    child_id: str
    error: Exception
    type: Literal["child_error"] = "child_error"


type Signal = UserInput | ChildCompleted | ChildError
"""Union of all signal types."""
