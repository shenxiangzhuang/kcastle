"""Message types for conversations with LLMs.

Defines the unified message model used across all providers:
- ContentPart hierarchy (TextPart, ThinkPart, ImagePart)
- ToolCall for function call requests
- Message with role-based conversation structure
- Context as the input to stream()/complete()
"""

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, field_serializer, field_validator

from kai.tool import Tool
from kai.types.usage import TokenUsage


class TextPart(BaseModel, frozen=True):
    """A text content block."""

    type: Literal["text"] = "text"
    text: str


class ThinkPart(BaseModel, frozen=True):
    """A thinking/reasoning content block."""

    type: Literal["think"] = "think"
    text: str
    signature: str | None = None
    """Encrypted thinking signature for cross-turn replay."""


class ImagePart(BaseModel, frozen=True):
    """An image content block (base64 encoded)."""

    type: Literal["image"] = "image"
    data: str
    """Base64-encoded image data."""
    mime_type: str
    """MIME type, e.g. 'image/png'."""


type ContentPart = TextPart | ThinkPart | ImagePart
"""Union of all content part types."""


class ToolCall(BaseModel, frozen=True):
    """A tool/function call requested by the assistant."""

    id: str
    """Unique identifier for this tool call."""
    name: str
    """Name of the tool to invoke."""
    arguments: str
    """Arguments as a JSON string."""


class Message(BaseModel):
    """A message in a conversation.

    Supports three roles:
    - ``user``: Human input
    - ``assistant``: Model output (may contain tool_calls and usage)
    - ``tool``: Tool execution result (must have tool_call_id)

    Content can be provided as a plain string (auto-wrapped to ``[TextPart]``),
    a single ContentPart, or a list of ContentPart.
    """

    role: Literal["user", "assistant", "tool"]
    """The role of the message sender."""

    content: list[ContentPart]
    """The content blocks of the message."""

    tool_calls: list[ToolCall] | None = None
    """Tool calls requested by the assistant (only for role='assistant')."""

    tool_call_id: str | None = None
    """The ID of the tool call this message responds to (only for role='tool')."""

    usage: TokenUsage | None = None
    """Token usage for this response (only for role='assistant')."""

    stop_reason: Literal["stop", "length", "tool_use", "error"] | None = None
    """Why the model stopped generating (only for role='assistant')."""

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_content(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [TextPart(text=value)]
        return value

    @field_serializer("content")
    def _serialize_content(self, content: list[ContentPart]) -> str | list[dict[str, Any]]:
        if len(content) == 1 and isinstance(content[0], TextPart):
            return content[0].text
        return [part.model_dump() for part in content]

    def __init__(
        self,
        *,
        role: Literal["user", "assistant", "tool"],
        content: list[ContentPart] | ContentPart | str | None = None,
        tool_calls: list[ToolCall] | None = None,
        tool_call_id: str | None = None,
        usage: TokenUsage | None = None,
        stop_reason: Literal["stop", "length", "tool_use", "error"] | None = None,
    ) -> None:
        parsed_content: list[ContentPart]
        if content is None:
            parsed_content = []
        elif isinstance(content, str):
            parsed_content = [TextPart(text=content)]
        elif isinstance(content, (TextPart, ThinkPart, ImagePart)):
            parsed_content = [content]
        else:
            parsed_content = list(content)
        super().__init__(
            role=role,
            content=parsed_content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            usage=usage,
            stop_reason=stop_reason,
        )

    def extract_text(self, sep: str = "") -> str:
        """Extract and join all text parts in the message content."""
        return sep.join(part.text for part in self.content if isinstance(part, TextPart))

    @staticmethod
    def tool_result(
        tool_call_id: str,
        content: str | list[ContentPart],
        *,
        is_error: bool = False,
    ) -> "Message":
        """Create a tool result message.

        Args:
            tool_call_id: The ID of the tool call being responded to.
            content: The result content (string or content parts).
            is_error: Whether this result represents an error.
        """
        if isinstance(content, str) and is_error:
            content = f"Error: {content}"
        return Message(role="tool", content=content, tool_call_id=tool_call_id)


class Context(BaseModel):
    """Input context for stream() and complete().

    Example::

        context = Context(
            system="You are a helpful assistant.",
            messages=[Message(role="user", content="Hello!")],
        )
    """

    system: str | None = None
    """System prompt."""

    messages: Sequence[Message]
    """Conversation history."""

    tools: Sequence[Tool] = ()
    """Available tools for the model."""
