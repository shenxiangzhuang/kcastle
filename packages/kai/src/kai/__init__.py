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

from kai.errors import ErrorKind, KaiError

from kai.providers import (
    AnthropicMessages,
    DeepseekAnthropic,
    DeepseekOpenAI,
    MinimaxAnthropic,
    MinimaxOpenAI,
    OpenAIChatCompletions,
    OpenAIResponses,
    ProviderBase,
)
from kai.stream import complete, stream

from kai.tool import Tool, ToolResult

from kai.types.message import (
    ContentPart,
    Context,
    ImagePart,
    Message,
    TextPart,
    ThinkPart,
    ToolCall,
)
from kai.types.stream import (
    Done,
    Error,
    StreamEvent,
    TextDelta,
    ThinkDelta,
    ThinkSignature,
    ToolCallBegin,
    ToolCallDelta,
    ToolCallEnd,
    Usage,
)
from kai.types.usage import TokenUsage

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
    # Stream events
    "StreamEvent",
    "TextDelta",
    "ThinkDelta",
    "ThinkSignature",
    "ToolCallBegin",
    "ToolCallDelta",
    "ToolCallEnd",
    "Usage",
    "Done",
    "Error",
    # Protocol
    "ProviderBase",
    # Usage
    "TokenUsage",
    # Errors
    "KaiError",
    "ErrorKind",
    # Implementations
    "OpenAIChatCompletions",
    "OpenAIResponses",
    "AnthropicMessages",
    "DeepseekOpenAI",
    "MinimaxOpenAI",
    "DeepseekAnthropic",
    "MinimaxAnthropic",
]
