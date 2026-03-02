"""LLM factory and registry for kcastle application layer."""

from __future__ import annotations

from collections.abc import Callable

from kai import LLM
from kai.providers.anthropic import AnthropicMessages
from kai.providers.openai import OpenAIChatCompletions, OpenAIResponses

from kcastle.provider_config import ProviderConfig

type ProviderFactory = Callable[[ProviderConfig], LLM]


class ProviderRegistry:
    """Registry mapping protocol names to factory callables."""

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, protocol: str, factory: ProviderFactory) -> None:
        """Register a factory for a protocol."""
        self._factories[protocol.lower()] = factory

    def create(self, config: ProviderConfig) -> LLM:
        """Create an LLM from config using registered protocol factory."""
        protocol = config.protocol.lower()
        factory = self._factories.get(protocol)
        if factory is None:
            available = sorted(self._factories.keys())
            raise ValueError(f"Unknown protocol: {config.protocol!r}. Available: {available}")
        return factory(config)


def _openai_completions_factory(config: ProviderConfig) -> LLM:
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


def _openai_responses_factory(config: ProviderConfig) -> LLM:
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


def _anthropic_factory(config: ProviderConfig) -> LLM:
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


_DEFAULT_REGISTRY = ProviderRegistry()
_DEFAULT_REGISTRY.register("openai-completions", _openai_completions_factory)
_DEFAULT_REGISTRY.register("openai-responses", _openai_responses_factory)
_DEFAULT_REGISTRY.register("anthropic", _anthropic_factory)


def create_provider(config: ProviderConfig, *, registry: ProviderRegistry | None = None) -> LLM:
    """Create an LLM from config.

    Uses the default registry unless a custom *registry* is supplied.
    """
    active_registry = registry or _DEFAULT_REGISTRY
    return active_registry.create(config)
