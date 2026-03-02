"""DeepSeek provider implementations."""

from __future__ import annotations

from typing import Any

from anthropic.types import ThinkingConfigParam

from kai.providers.anthropic import AnthropicBase
from kai.providers.openai import OpenAIChatBase


class DeepseekOpenAI(OpenAIChatBase):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = "https://api.deepseek.com",
        extra_body: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            provider="deepseek-openai",
            model=model,
            api_key=api_key,
            base_url=base_url,
            extra_body=extra_body,
        )


class DeepseekAnthropic(AnthropicBase):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = "https://api.deepseek.com/anthropic",
        max_tokens: int = 16384,
        thinking: ThinkingConfigParam | None = None,
        **client_kwargs: Any,
    ) -> None:
        super().__init__(
            provider="deepseek-anthropic",
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            thinking=thinking,
            **client_kwargs,
        )


__all__ = ["DeepseekOpenAI", "DeepseekAnthropic"]
