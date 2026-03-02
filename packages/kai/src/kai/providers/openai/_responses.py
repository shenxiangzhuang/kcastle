"""OpenAI Responses API provider."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx
from openai import AsyncOpenAI, OpenAIError
from openai.types.responses import (
    ResponseInputItemParam,
    ResponseInputParam,
    ResponseStreamEvent,
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
)
from kai.providers.openai._common import build_tools, convert_error
from kai.usage import TokenUsage


class OpenAIResponses:
    """OpenAI Responses API provider.

    Uses the newer ``/v1/responses`` endpoint. Supports streaming with
    rich event types including reasoning summaries and function calls.

    Example::

        provider = OpenAIResponses(model="gpt-4.1")
        provider = OpenAIResponses(
            model="o3",
            reasoning={"effort": "medium", "summary": "auto"},
        )
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._reasoning = reasoning

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    async def stream_raw(self, context: Context, **kwargs: Any) -> AsyncIterator[Chunk]:
        """Stream raw chunks from the OpenAI Responses API."""
        input_items = _build_input(context)
        tools = build_tools(context.tools) if context.tools else None

        api_kwargs: dict[str, Any] = {"store": False}
        if tools:
            api_kwargs["tools"] = tools
        if "temperature" in kwargs:
            api_kwargs["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            api_kwargs["max_output_tokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            api_kwargs["top_p"] = kwargs["top_p"]

        # Reasoning configuration
        reasoning = kwargs.get("reasoning", self._reasoning)
        if reasoning:
            api_kwargs["reasoning"] = reasoning
            api_kwargs["include"] = ["reasoning.encrypted_content"]

        try:
            response = await self._client.responses.create(
                model=self._model,
                input=input_items,
                stream=True,
                **api_kwargs,
            )
            async for chunk in _convert_stream(response):
                yield chunk
        except (OpenAIError, httpx.HTTPError) as e:
            raise convert_error(e) from e


def _build_input(context: Context) -> ResponseInputParam:
    """Convert a Context into Responses API input format."""
    items: list[ResponseInputItemParam] = []

    if context.system:
        items.append({"role": "developer", "content": context.system})

    for msg in context.messages:
        items.extend(_convert_message(msg))

    return items


def _convert_message(message: Message) -> list[ResponseInputItemParam]:
    """Convert a single kai Message to Responses API input items.

    A single kai Message may expand to multiple Responses API input items
    (e.g., an assistant message with tool calls becomes a message item +
    function_call items).
    """
    if message.role == "tool":
        return [
            {
                "type": "function_call_output",
                "call_id": message.tool_call_id or "",
                "output": message.extract_text(),
            }
        ]

    if message.role == "user":
        content = _convert_content_for_responses(message.content)
        return [{"role": "user", "content": content}]  # type: ignore[list-item]

    # assistant — split into message item + function_call items
    result: list[ResponseInputItemParam] = []

    # Text content → output_text parts in a message item
    text_parts = [p for p in message.content if isinstance(p, TextPart)]
    if text_parts:
        output_text_parts: list[dict[str, Any]] = [
            {"type": "output_text", "text": p.text, "annotations": []} for p in text_parts
        ]
        result.append(
            cast(
                ResponseInputItemParam,
                {
                    "type": "message",
                    "role": "assistant",
                    "content": output_text_parts,
                    "status": "completed",
                },
            )
        )

    # Tool calls → separate function_call items
    for tc in message.tool_calls or []:
        result.append(
            {
                "type": "function_call",
                "call_id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
            }
        )

    return result


def _convert_content_for_responses(
    content: list[ContentPart],
) -> str | list[dict[str, Any]]:
    """Convert content parts to Responses API input content format."""
    if len(content) == 1 and isinstance(content[0], TextPart):
        return content[0].text

    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            parts.append({"type": "input_text", "text": part.text})
        elif isinstance(part, ImagePart):
            data_url = f"data:{part.mime_type};base64,{part.data}"
            parts.append(
                {
                    "type": "input_image",
                    "image_url": data_url,
                    "detail": "auto",
                }
            )
    return parts if parts else ""


async def _convert_stream(
    response: AsyncIterator[ResponseStreamEvent],
) -> AsyncIterator[Chunk]:
    """Convert Responses API stream events to kai Chunks.

    Responses API stream events are named events (not delta-on-choices like
    Completions). We map them to the same Chunk types used by all providers.
    """
    async for event in response:
        match event.type:
            # --- Text output ---
            case "response.output_text.delta":
                yield TextChunk(text=event.delta)

            # --- Reasoning / thinking ---
            case "response.reasoning_summary_text.delta":
                yield ThinkChunk(text=event.delta)

            # --- Tool calls ---
            case "response.output_item.added":
                item = event.item
                if item.type == "function_call":
                    yield ToolCallStart(
                        id=item.call_id or str(uuid.uuid4()),
                        name=item.name,
                    )

            case "response.function_call_arguments.delta":
                yield ToolCallDelta(arguments=event.delta)

            case "response.output_item.done":
                item = event.item
                if item.type == "function_call":
                    yield ToolCallEnd()

            # --- Usage & completion ---
            case "response.completed":
                resp = event.response
                if resp.usage:
                    cached = 0
                    input_tokens = resp.usage.input_tokens
                    details = resp.usage.input_tokens_details
                    if details and details.cached_tokens:
                        cached = details.cached_tokens
                        input_tokens -= cached
                    yield UsageChunk(
                        usage=TokenUsage(
                            input_tokens=input_tokens,
                            output_tokens=resp.usage.output_tokens,
                            cache_read_tokens=cached,
                        )
                    )

            # --- Errors ---
            case "error":
                from kai.errors import ProviderError

                msg = getattr(event, "message", "Unknown error")
                code = getattr(event, "code", None)
                raise ProviderError(f"Responses API error ({code}): {msg}")

            case _:
                pass  # Ignore unhandled events
