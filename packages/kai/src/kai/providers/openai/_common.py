"""Shared utilities for OpenAI providers (Completions and Responses)."""

from __future__ import annotations

from typing import Any

import httpx
import openai
from openai import OpenAIError
from openai.types.chat import ChatCompletionToolParam

from kai.errors import ConnectionError, ProviderError, StatusError, TimeoutError


def build_tools(tools: Any) -> list[ChatCompletionToolParam]:
    """Convert kai Tools to OpenAI function tool format.

    Works for both Chat Completions and Responses APIs.
    """
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


def convert_error(error: OpenAIError | httpx.HTTPError) -> ProviderError:
    """Convert OpenAI/httpx errors to kai errors."""
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
