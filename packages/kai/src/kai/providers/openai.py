"""OpenAI-compatible providers in a single module."""

from __future__ import annotations

import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx
import openai
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.responses import (
    ResponseInputItemParam,
    ResponseInputParam,
    ResponseStreamEvent,
)

from kai.errors import ConnectionError, ProviderError, StatusError, TimeoutError
from kai.providers.base import ProviderBase
from kai.types.message import (
    ContentPart,
    Context,
    ImagePart,
    Message,
    TextPart,
    ThinkPart,
)
from kai.types.stream import (
    Chunk,
    TextChunk,
    ThinkChunk,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageChunk,
)
from kai.types.usage import TokenUsage


class OpenAIBase(ProviderBase, ABC):
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        self._provider = provider
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @abstractmethod
    async def stream(self, context: Context, **kwargs: Any) -> AsyncIterator[Chunk]:
        raise NotImplementedError  # pragma: no cover
        yield  # pragma: no cover  # noqa: RET503

    @staticmethod
    def _convert_error(error: OpenAIError | httpx.HTTPError) -> ProviderError:
        if isinstance(error, openai.APIStatusError):
            return StatusError(error.status_code, error.message)
        if isinstance(error, openai.APIConnectionError):
            return ConnectionError(error.message)
        if isinstance(error, openai.APITimeoutError):
            return TimeoutError(error.message)
        if isinstance(error, httpx.TimeoutException):
            return TimeoutError(str(error))
        if isinstance(error, httpx.NetworkError):
            return ConnectionError(str(error))
        if isinstance(error, httpx.HTTPStatusError):
            return StatusError(error.response.status_code, str(error))
        return ProviderError(f"OpenAI error: {error}")


class OpenAIChatBase(OpenAIBase, ABC):
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> None:
        super().__init__(provider=provider, model=model, api_key=api_key, base_url=base_url)
        self._extra_body = extra_body

    async def stream(self, context: Context, **kwargs: Any) -> AsyncIterator[Chunk]:
        messages = _build_messages(context)
        tools = _build_tools(context.tools) if context.tools else None

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
            async for chunk in _convert_chat_stream(response):
                yield chunk
        except (OpenAIError, httpx.HTTPError) as e:
            raise self._convert_error(e) from e


class OpenAIResponsesBase(OpenAIBase, ABC):
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(provider=provider, model=model, api_key=api_key, base_url=base_url)
        self._reasoning = reasoning

    async def stream(self, context: Context, **kwargs: Any) -> AsyncIterator[Chunk]:
        input_items = _build_input(context)
        tools = _build_tools(context.tools) if context.tools else None

        api_kwargs: dict[str, Any] = {"store": False}
        if tools:
            api_kwargs["tools"] = tools
        if "temperature" in kwargs:
            api_kwargs["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            api_kwargs["max_output_tokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            api_kwargs["top_p"] = kwargs["top_p"]

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
            async for chunk in _convert_responses_stream(response):
                yield chunk
        except (OpenAIError, httpx.HTTPError) as e:
            raise self._convert_error(e) from e


class OpenAIChatCompletions(OpenAIChatBase):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            provider="openai",
            model=model,
            api_key=api_key,
            base_url=base_url,
            extra_body=extra_body,
        )


class OpenAIResponses(OpenAIResponsesBase):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            provider="openai-responses",
            model=model,
            api_key=api_key,
            base_url=base_url,
            reasoning=reasoning,
        )


def _build_tools(tools: Any) -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def _build_messages(context: Context) -> list[ChatCompletionMessageParam]:
    messages: list[ChatCompletionMessageParam] = []

    if context.system:
        messages.append({"role": "system", "content": context.system})

    for msg in context.messages:
        messages.append(_convert_message(msg))

    return messages


def _convert_message(message: Message) -> ChatCompletionMessageParam:
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "content": message.extract_text(),
        }

    if message.role == "user":
        content = _convert_content_for_openai(message.content)
        return cast(ChatCompletionMessageParam, {"role": "user", "content": content})

    result: dict[str, Any] = {"role": "assistant"}

    think_parts = [p for p in message.content if isinstance(p, ThinkPart)]
    text_parts = [p for p in message.content if isinstance(p, TextPart)]
    if text_parts:
        result["content"] = "".join(p.text for p in text_parts)
    else:
        result["content"] = None

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


_REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")


def _extract_reasoning_text(delta: object) -> str | None:
    for field in _REASONING_FIELDS:
        value = getattr(delta, field, None)
        if value and isinstance(value, str):
            return value

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


async def _convert_chat_stream(
    response: AsyncIterator[ChatCompletionChunk],
) -> AsyncIterator[Chunk]:
    current_tool_id: str | None = None

    async for chunk in response:
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

        reasoning = _extract_reasoning_text(delta)
        if reasoning:
            yield ThinkChunk(text=reasoning)

        if delta.content:
            yield TextChunk(text=delta.content)

        for tool_call in delta.tool_calls or []:
            if not tool_call.function:
                continue

            if current_tool_id is None or (tool_call.id and tool_call.id != current_tool_id):
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

    if current_tool_id is not None:
        yield ToolCallEnd()


def _build_input(context: Context) -> ResponseInputParam:
    items: list[ResponseInputItemParam] = []

    if context.system:
        items.append({"role": "developer", "content": context.system})

    for msg in context.messages:
        items.extend(_convert_message_for_responses(msg))

    return items


def _convert_message_for_responses(message: Message) -> list[ResponseInputItemParam]:
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

    result: list[ResponseInputItemParam] = []

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


async def _convert_responses_stream(
    response: AsyncIterator[ResponseStreamEvent],
) -> AsyncIterator[Chunk]:
    async for event in response:
        match event.type:
            case "response.output_text.delta":
                yield TextChunk(text=event.delta)

            case "response.reasoning_summary_text.delta":
                yield ThinkChunk(text=event.delta)

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

            case "error":
                msg = getattr(event, "message", "Unknown error")
                code = getattr(event, "code", None)
                raise ProviderError(f"Responses API error ({code}): {msg}")

            case _:
                pass
