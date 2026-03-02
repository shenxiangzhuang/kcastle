"""kai — Unified LLM API layer.

Provides a simple, API-agnostic interface for streaming LLM completions
with tool calling support. Two entry points:

- ``stream(llm, context)`` — async iterator of rich stream events
- ``complete(llm, context)`` — get a complete response message

Example::

    from kai import OpenAIChatCompletions, Context, Message, complete

    llm = OpenAIChatCompletions(model="gpt-4o")
    context = Context(
        system="You are a helpful assistant.",
        messages=[Message(role="user", content="Hello!")],
    )
    message = await complete(llm, context)
    print(message.extract_text())
"""

import logging as _logging

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

# Protocol contract
from kai.providers import (
    LLM,
)
from kai.providers.anthropic import AnthropicMessages

# Concrete implementations
from kai.providers.openai import OpenAIChatCompletions, OpenAIResponses
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
    # Protocol
    "LLM",
    # Usage
    "TokenUsage",
    # Errors
    "KaiError",
    "ProviderError",
    "ConnectionError",
    "TimeoutError",
    "StatusError",
    "EmptyResponseError",
    # Implementations
    "OpenAIChatCompletions",
    "OpenAIResponses",
    "AnthropicMessages",
]

_logging.getLogger("kai").addHandler(_logging.NullHandler())
