"""OpenAI Chat Completions API protocol implementation."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)

from kai.chunk import (
    Chunk,
    TextChunk,
    ThinkChunk,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageChunk,
)
from kai.message import (
    ContentPart,
    Context,
    ImagePart,
    Message,
    TextPart,
    ThinkPart,
)
from kai.providers.openai._common import build_tools, convert_error
from kai.usage import TokenUsage


class OpenAIChatCompletions:
    """OpenAI Chat Completions API implementation.

    Supports OpenAI and any OpenAI-compatible API (via base_url).

    Example::

        llm = OpenAIChatCompletions(model="gpt-4o")
        llm = OpenAIChatCompletions(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
        )
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        self._model = model
        self._extra_body = extra_body
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    async def stream_raw(self, context: Context, **kwargs: Any) -> AsyncIterator[Chunk]:
        """Stream raw chunks from the OpenAI Chat Completions API."""
        messages = _build_messages(context)
        tools = build_tools(context.tools) if context.tools else None

        api_kwargs: dict[str, Any] = {}
        if tools:
            api_kwargs["tools"] = tools
        if "temperature" in kwargs:
            api_kwargs["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            api_kwargs["max_tokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            api_kwargs["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            api_kwargs["stop"] = kwargs["stop"]

        if self._extra_body:
            api_kwargs["extra_body"] = self._extra_body

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                **api_kwargs,
            )
            async for chunk in _convert_stream(response):
                yield chunk
        except (OpenAIError, httpx.HTTPError) as e:
            raise convert_error(e) from e


def _build_messages(context: Context) -> list[ChatCompletionMessageParam]:
    """Convert a Context into OpenAI message format."""
    messages: list[ChatCompletionMessageParam] = []

    if context.system:
        messages.append({"role": "system", "content": context.system})

    for msg in context.messages:
        messages.append(_convert_message(msg))

    return messages


def _convert_message(message: Message) -> ChatCompletionMessageParam:
    """Convert a single kai Message to OpenAI format."""
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "content": message.extract_text(),
        }

    if message.role == "user":
        content = _convert_content_for_openai(message.content)
        return cast(ChatCompletionMessageParam, {"role": "user", "content": content})

    # assistant
    result: dict[str, Any] = {"role": "assistant"}

    # Separate think parts from text parts
    think_parts = [p for p in message.content if isinstance(p, ThinkPart)]
    text_parts = [p for p in message.content if isinstance(p, TextPart)]
    if text_parts:
        result["content"] = "".join(p.text for p in text_parts)
    else:
        result["content"] = None

    # DeepSeek reasoning models require reasoning_content on assistant messages
    # during tool-call sub-turns (see https://api-docs.deepseek.com/guides/thinking_mode)
    if think_parts:
        result["reasoning_content"] = "".join(p.text for p in think_parts)

    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in message.tool_calls
        ]

    return result  # type: ignore[return-value]


def _convert_content_for_openai(
    content: list[ContentPart],
) -> str | list[dict[str, Any]]:
    """Convert content parts to OpenAI content format."""
    if len(content) == 1 and isinstance(content[0], TextPart):
        return content[0].text

    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            data_url = f"data:{part.mime_type};base64,{part.data}"
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )
    return parts if parts else ""


# Fields that various OpenAI-compatible providers use for thinking/reasoning
# content on ``delta``.  We probe them in order and use the first non-empty
# one to avoid duplication (some providers populate multiple fields with
# identical content).
_REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")


def _extract_reasoning_text(delta: object) -> str | None:
    """Extract reasoning/thinking text from a chat completion delta.

    Supports three mechanisms used by different providers:

    1. **Scalar fields** — ``reasoning_content`` (DeepSeek, llama.cpp),
       ``reasoning`` or ``reasoning_text`` (other OpenAI-compatible APIs).
    2. **``reasoning_details``** — An array of detail objects used by MiniMax.
       Items with ``type == "reasoning.text"`` carry a ``text`` field
       containing the thinking fragment.

    Returns the extracted text, or *None* if no reasoning content was found.
    """
    # 1. Scalar reasoning fields
    for field in _REASONING_FIELDS:
        value = getattr(delta, field, None)
        if value and isinstance(value, str):
            return value

    # 2. reasoning_details array (MiniMax)
    details = getattr(delta, "reasoning_details", None)
    if details and isinstance(details, list):
        parts: list[str] = []
        for item in details:  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(item, dict):
                continue
            item_dict = cast(dict[str, Any], item)
            if item_dict.get("type") == "reasoning.text":
                text = item_dict.get("text")
                if text and isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)

    return None


async def _convert_stream(
    response: AsyncIterator[ChatCompletionChunk],
) -> AsyncIterator[Chunk]:
    """Convert OpenAI stream chunks to kai Chunks.

    Tracks the current in-progress tool call via its ID.  When a chunk
    arrives for a *different* tool call (detected by a new ``tool_call.id``),
    the previous one is closed with ``ToolCallEnd`` before starting the next.
    After the stream ends the last active tool call is closed as well.

    Reasoning/thinking content is extracted via :func:`_extract_reasoning_text`
    which probes dedicated reasoning fields (``reasoning_content``,
    ``reasoning_details``, etc.).  For providers that embed ``<think>`` tags
    in ``delta.content`` (e.g. MiniMax OpenAI), use ``extra_body={"reasoning_split": True}``
    to instruct the API to separate thinking into dedicated fields instead.
    """
    current_tool_id: str | None = None

    async for chunk in response:
        # Extract usage if present
        if chunk.usage:
            cached = 0
            input_tokens = chunk.usage.prompt_tokens
            if (
                chunk.usage.prompt_tokens_details
                and chunk.usage.prompt_tokens_details.cached_tokens
            ):
                cached = chunk.usage.prompt_tokens_details.cached_tokens
                input_tokens -= cached
            yield UsageChunk(
                usage=TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                    cache_read_tokens=cached,
                )
            )

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        # Reasoning/thinking via dedicated fields (DeepSeek, MiniMax-M1, …)
        reasoning = _extract_reasoning_text(delta)
        if reasoning:
            yield ThinkChunk(text=reasoning)

        # Text content
        if delta.content:
            yield TextChunk(text=delta.content)

        # Tool calls — detect new vs. continuation by comparing IDs
        for tool_call in delta.tool_calls or []:
            if not tool_call.function:
                continue

            # A new tool call is detected when we have no active tool, or
            # the chunk carries a tool_call.id different from the current one.
            if current_tool_id is None or (tool_call.id and tool_call.id != current_tool_id):
                # Close the previous tool call if any
                if current_tool_id is not None:
                    yield ToolCallEnd()

                current_tool_id = tool_call.id or str(uuid.uuid4())
                yield ToolCallStart(
                    id=current_tool_id,
                    name=tool_call.function.name or "",
                )
                if tool_call.function.arguments:
                    yield ToolCallDelta(arguments=tool_call.function.arguments)
            elif tool_call.function.arguments:
                yield ToolCallDelta(arguments=tool_call.function.arguments)

    # Close the last active tool call
    if current_tool_id is not None:
        yield ToolCallEnd()
