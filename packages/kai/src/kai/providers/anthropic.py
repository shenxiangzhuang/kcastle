"""Anthropic Messages API protocol implementation."""

from __future__ import annotations

import json
import os
from abc import ABC
from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

from anthropic import (
    AnthropicError,
    AsyncAnthropic,
    AsyncStream,
)
from anthropic import (
    APIConnectionError as AnthropicConnectionError,
)
from anthropic import (
    APIStatusError as AnthropicStatusError,
)
from anthropic import (
    APITimeoutError as AnthropicTimeoutError,
)
from anthropic.types import (
    Base64ImageSourceParam,
    ContentBlockParam,
    ImageBlockParam,
    MessageDeltaEvent,
    MessageParam,
    MessageStartEvent,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageStreamEvent,
    TextBlockParam,
    ThinkingConfigParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)

from kai.errors import ConnectionError, ProviderError, StatusError, TimeoutError
from kai.providers.base import LLMBase
from kai.tool import Tool
from kai.types.message import Context, ImagePart, Message, TextPart, ThinkPart
from kai.types.stream import (
    Chunk,
    TextChunk,
    ThinkChunk,
    ThinkSignatureChunk,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageChunk,
)
from kai.types.usage import TokenUsage


class AnthropicBase(LLMBase, ABC):
    """Shared Anthropic-compatible implementation base."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 16384,
        thinking: ThinkingConfigParam | None = None,
        **client_kwargs: Any,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._thinking = thinking

        if "auth_token" not in client_kwargs and base_url:
            client_kwargs["auth_token"] = api_key

        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url, **client_kwargs)

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    async def stream_raw(self, context: Context, **kwargs: Any) -> AsyncIterator[Chunk]:
        """Stream raw chunks from the Anthropic Messages API."""
        system = context.system or ""
        messages = _build_messages(context)
        tools = _build_tools(context.tools) if context.tools else []

        api_kwargs: dict[str, Any] = {
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
        }
        if self._thinking:
            api_kwargs["thinking"] = self._thinking
        if "temperature" in kwargs:
            api_kwargs["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            api_kwargs["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            api_kwargs["top_k"] = kwargs["top_k"]

        try:
            response = await self._client.messages.create(
                model=self._model,
                messages=messages,
                system=system if system else [],
                tools=tools,
                stream=True,
                **api_kwargs,
            )
            async for chunk in _convert_stream(response):
                yield chunk
        except AnthropicError as e:
            raise _convert_error(e) from e


class AnthropicMessages(AnthropicBase):
    """Anthropic Messages API implementation.

    Example::

        llm = AnthropicMessages(model="claude-sonnet-4-20250514")
        llm = AnthropicMessages(model="claude-sonnet-4-20250514", max_tokens=8192)
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 16384,
        thinking: ThinkingConfigParam | None = None,
        **client_kwargs: Any,
    ) -> None:
        super().__init__(
            provider="anthropic",
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            thinking=thinking,
            **client_kwargs,
        )


def _build_messages(context: Context) -> list[MessageParam]:
    """Convert Context messages to Anthropic format.

    Consecutive tool-result messages are merged into a single ``user`` message
    with multiple ``tool_result`` blocks. The Anthropic API requires that all
    ``tool_result`` blocks for a given assistant turn appear together in one
    message; sending them as separate messages causes a 400 error.
    """
    result: list[MessageParam] = []
    msgs = context.messages
    i = 0
    while i < len(msgs):
        msg = msgs[i]
        if msg.role == "tool":
            # Collect all consecutive tool-result messages into one user message.
            tool_blocks: list[ContentBlockParam] = []
            while i < len(msgs) and msgs[i].role == "tool":
                tm = msgs[i]
                if tm.tool_call_id is None:
                    raise ProviderError("Tool result message missing tool_call_id.")
                tool_blocks.append(
                    ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=tm.tool_call_id,
                        content=tm.extract_text(),
                    )
                )
                i += 1
            result.append(MessageParam(role="user", content=tool_blocks))
        else:
            result.append(_convert_message(msg))
            i += 1
    return result


def _convert_message(message: Message) -> MessageParam:
    """Convert a single kai Message to the Anthropic wire format."""
    if message.role == "tool":
        if message.tool_call_id is None:
            raise ProviderError("Tool result message missing tool_call_id.")
        block = ToolResultBlockParam(
            type="tool_result",
            tool_use_id=message.tool_call_id,
            content=message.extract_text(),
        )
        return MessageParam(role="user", content=[block])

    if message.role == "user":
        blocks: list[ContentBlockParam] = []
        for part in message.content:
            if isinstance(part, TextPart):
                blocks.append(TextBlockParam(type="text", text=part.text))
            elif isinstance(part, ImagePart):
                blocks.append(_image_to_anthropic(part))
        return MessageParam(role="user", content=blocks if blocks else "")

    # assistant
    blocks = []
    for part in message.content:
        if isinstance(part, TextPart):
            blocks.append(TextBlockParam(type="text", text=part.text))
        elif isinstance(part, ThinkPart):
            if part.signature is None:
                continue  # Strip unsigned thinking blocks
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": part.text,
                    "signature": part.signature,
                }
            )

    for tc in message.tool_calls or []:
        try:
            raw: object = json.loads(tc.arguments) if tc.arguments else {}
        except json.JSONDecodeError as e:
            raise ProviderError("Tool call arguments must be valid JSON.") from e
        if not isinstance(raw, dict):
            raise ProviderError("Tool call arguments must be a JSON object.")
        tool_input = cast(dict[str, object], raw)
        blocks.append(
            ToolUseBlockParam(
                type="tool_use",
                id=tc.id,
                name=tc.name,
                input=tool_input,
            )
        )

    return MessageParam(role="assistant", content=blocks)


_ANTHROPIC_MEDIA_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def _image_to_anthropic(part: ImagePart) -> ImageBlockParam:
    """Convert an ImagePart to Anthropic's image block format."""
    if part.mime_type not in _ANTHROPIC_MEDIA_TYPES:
        raise ProviderError(f"Unsupported image type: {part.mime_type}")
    media_type = cast(Any, part.mime_type)
    return ImageBlockParam(
        type="image",
        source=Base64ImageSourceParam(
            type="base64",
            data=part.data,
            media_type=media_type,
        ),
    )


def _build_tools(tools: Sequence[Tool]) -> list[ToolParam]:
    """Convert kai Tools to Anthropic tool format."""
    return [
        ToolParam(
            name=tool.name,
            description=tool.description,
            input_schema=tool.parameters,
        )
        for tool in tools
    ]


async def _convert_stream(
    response: AsyncStream[RawMessageStreamEvent],
) -> AsyncIterator[Chunk]:
    """Convert Anthropic stream events to kai Chunks."""
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_write = 0
    active_tool = False

    async with response as stream:
        async for event in stream:
            if isinstance(event, MessageStartEvent):
                usage = event.message.usage
                input_tokens = usage.input_tokens or 0
                output_tokens = usage.output_tokens or 0
                cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
                cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

            elif isinstance(event, RawContentBlockStartEvent):
                block = event.content_block
                match block.type:
                    case "text":
                        if block.text:
                            yield TextChunk(text=block.text)
                    case "thinking":
                        if hasattr(block, "thinking") and block.thinking:
                            yield ThinkChunk(text=block.thinking)
                    case "tool_use":
                        if active_tool:
                            yield ToolCallEnd()
                        active_tool = True
                        yield ToolCallStart(id=block.id, name=block.name)
                    case _:
                        pass

            elif isinstance(event, RawContentBlockDeltaEvent):
                delta = event.delta
                match delta.type:
                    case "text_delta":
                        yield TextChunk(text=delta.text)
                    case "thinking_delta":
                        yield ThinkChunk(text=delta.thinking)
                    case "input_json_delta":
                        yield ToolCallDelta(arguments=delta.partial_json)
                    case "signature_delta":
                        yield ThinkSignatureChunk(signature=delta.signature)
                    case _:
                        pass

            elif isinstance(event, MessageDeltaEvent):
                if event.usage:
                    delta_usage = event.usage
                    if hasattr(delta_usage, "output_tokens") and delta_usage.output_tokens:
                        output_tokens = delta_usage.output_tokens

    # Close any active tool call
    if active_tool:
        yield ToolCallEnd()

    # Emit final usage
    yield UsageChunk(
        usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )
    )


def _convert_error(error: AnthropicError) -> ProviderError:
    """Convert Anthropic errors to kai errors."""
    if isinstance(error, AnthropicStatusError):
        return StatusError(error.status_code, str(error))
    if isinstance(error, AnthropicConnectionError):
        return ConnectionError(str(error))
    if isinstance(error, AnthropicTimeoutError):
        return TimeoutError(str(error))
    return ProviderError(f"Anthropic error: {error}")


__all__ = [
    "AnthropicBase",
    "AnthropicMessages",
]
