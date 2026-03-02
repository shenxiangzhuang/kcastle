"""LLM factory and registry for kcastle application layer."""

from __future__ import annotations

from collections.abc import Callable

from kai import ProviderBase
from kai.providers.anthropic import AnthropicMessages
from kai.providers.deepseek import DeepseekAnthropic, DeepseekOpenAI
from kai.providers.minimax import MinimaxAnthropic, MinimaxOpenAI
from kai.providers.openai import (
    OpenAIChatCompletions,
    OpenAIResponses,
)

from kcastle.provider_config import ProviderConfig

type ProviderFactory = Callable[[ProviderConfig], ProviderBase]


class ProviderRegistry:
    """Registry mapping provider IDs to factory callables."""

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, provider: str, factory: ProviderFactory) -> None:
        """Register a factory for a provider ID."""
        self._factories[provider.lower()] = factory

    def create(self, config: ProviderConfig) -> ProviderBase:
        """Create an ProviderBase from config using registered provider factory."""
        provider = config.provider.lower()
        factory = self._factories.get(provider)
        if factory is None:
            available = sorted(self._factories.keys())
            raise ValueError(f"Unknown provider: {config.provider!r}. Available: {available}")
        return factory(config)


def _openai_completions_factory(config: ProviderConfig) -> ProviderBase:
    kwargs: dict[str, object] = {
        "model": config.model,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.extra_body:
        kwargs["extra_body"] = config.extra_body
    if config.options:
        kwargs.update(config.options)
    return OpenAIChatCompletions(**kwargs)  # type: ignore[arg-type]


def _deepseek_openai_factory(config: ProviderConfig) -> ProviderBase:
    kwargs: dict[str, object] = {
        "model": config.model,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.extra_body:
        kwargs["extra_body"] = config.extra_body
    if config.options:
        kwargs.update(config.options)
    return DeepseekOpenAI(**kwargs)  # type: ignore[arg-type]


def _minimax_openai_factory(config: ProviderConfig) -> ProviderBase:
    kwargs: dict[str, object] = {
        "model": config.model,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.extra_body:
        kwargs["extra_body"] = config.extra_body
    if config.options:
        kwargs.update(config.options)
    return MinimaxOpenAI(**kwargs)  # type: ignore[arg-type]


def _openai_responses_factory(config: ProviderConfig) -> ProviderBase:
    kwargs: dict[str, object] = {
        "model": config.model,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.options:
        kwargs.update(config.options)
    return OpenAIResponses(**kwargs)  # type: ignore[arg-type]


def _anthropic_factory(config: ProviderConfig) -> ProviderBase:
    kwargs: dict[str, object] = {
        "model": config.model,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.options:
        kwargs.update(config.options)
    return AnthropicMessages(**kwargs)  # type: ignore[arg-type]


def _deepseek_anthropic_factory(config: ProviderConfig) -> ProviderBase:
    kwargs: dict[str, object] = {
        "model": config.model,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.options:
        kwargs.update(config.options)
    return DeepseekAnthropic(**kwargs)  # type: ignore[arg-type]


def _minimax_anthropic_factory(config: ProviderConfig) -> ProviderBase:
    kwargs: dict[str, object] = {
        "model": config.model,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.options:
        kwargs.update(config.options)
    return MinimaxAnthropic(**kwargs)  # type: ignore[arg-type]


_DEFAULT_REGISTRY = ProviderRegistry()
_DEFAULT_REGISTRY.register("openai", _openai_completions_factory)
_DEFAULT_REGISTRY.register("openai-responses", _openai_responses_factory)
_DEFAULT_REGISTRY.register("anthropic", _anthropic_factory)
_DEFAULT_REGISTRY.register("deepseek-openai", _deepseek_openai_factory)
_DEFAULT_REGISTRY.register("deepseek-anthropic", _deepseek_anthropic_factory)
_DEFAULT_REGISTRY.register("minimax-openai", _minimax_openai_factory)
_DEFAULT_REGISTRY.register("minimax-anthropic", _minimax_anthropic_factory)


def create_provider(
    config: ProviderConfig,
    *,
    registry: ProviderRegistry | None = None,
) -> ProviderBase:
    """Create a ProviderBase from config.

    Uses the default registry unless a custom *registry* is supplied.
    """
    active_registry = registry or _DEFAULT_REGISTRY
    return active_registry.create(config)
