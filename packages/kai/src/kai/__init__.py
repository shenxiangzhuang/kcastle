"""kai — Unified multi-provider LLM API.

Provides a simple, provider-agnostic interface for streaming LLM completions
with tool calling support. Two entry points:

- ``stream(provider, context)`` — async iterator of rich stream events
- ``complete(provider, context)`` — get a complete response message

Example::

    from kai import OpenAICompletions, Context, Message, complete

    provider = OpenAICompletions(model="gpt-4o")
    context = Context(
        system="You are a helpful assistant.",
        messages=[Message(role="user", content="Hello!")],
    )
    message = await complete(provider, context)
    print(message.extract_text())
"""

# Core functions
# Errors
from kai.errors import (
    ConnectionError,
    EmptyResponseError,
    KaiError,
    ProviderError,
    StatusError,
    TimeoutError,
)

# Stream events
from kai.event import (
    DoneEvent,
    ErrorEvent,
    StartEvent,
    StreamEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkDeltaEvent,
    ThinkEndEvent,
    ThinkStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)

# Message types
from kai.message import ContentPart, Context, ImagePart, Message, TextPart, ThinkPart, ToolCall

# Provider protocol
from kai.providers import Provider
from kai.providers.anthropic import Anthropic

# Concrete providers
from kai.providers.openai import OpenAICompletions, OpenAIResponses
from kai.stream import complete, stream

# Tool definition
from kai.tool import Tool, ToolResult

# Token usage
from kai.usage import TokenUsage

__all__ = [
    # Functions
    "stream",
    "complete",
    # Message types
    "Message",
    "ContentPart",
    "TextPart",
    "ThinkPart",
    "ImagePart",
    "ToolCall",
    "Context",
    # Tool
    "Tool",
    "ToolResult",
    # Events
    "StreamEvent",
    "StartEvent",
    "TextStartEvent",
    "TextDeltaEvent",
    "TextEndEvent",
    "ThinkStartEvent",
    "ThinkDeltaEvent",
    "ThinkEndEvent",
    "ToolCallStartEvent",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "DoneEvent",
    "ErrorEvent",
    # Provider
    "Provider",
    # Usage
    "TokenUsage",
    # Errors
    "KaiError",
    "ProviderError",
    "ConnectionError",
    "TimeoutError",
    "StatusError",
    "EmptyResponseError",
    # Providers
    "OpenAICompletions",
    "OpenAIResponses",
    "Anthropic",
]
