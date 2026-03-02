"""Provider factory and registry.

Defines a typed provider config and a registry-based construction mechanism
for creating concrete providers from protocol names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from kai.providers.anthropic import Anthropic
from kai.providers.openai import OpenAICompletions, OpenAIResponses

if TYPE_CHECKING:
    from kai.providers import Provider


type ProviderFactory = Callable[["ProviderConfig"], "Provider"]


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Provider construction config.

    A config captures vendor identity, protocol, model, and endpoint/auth
    options needed to construct a concrete provider instance.
    """

    vendor: str
    protocol: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    extra_body: dict[str, object] | None = None
    options: dict[str, object] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]

    @property
    def name(self) -> str:
        """Canonical profile name, e.g. ``deepseek-openai-completions``."""
        return f"{self.vendor}-{self.protocol.lower()}"


class ProviderRegistry:
    """Registry mapping protocol names to provider factory callables."""

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, protocol: str, factory: ProviderFactory) -> None:
        """Register a factory for a protocol."""
        self._factories[protocol.lower()] = factory

    def create(self, config: ProviderConfig) -> "Provider":
        """Create a provider from config using registered protocol factory."""
        protocol = config.protocol.lower()
        factory = self._factories.get(protocol)
        if factory is None:
            available = sorted(self._factories.keys())
            raise ValueError(f"Unknown protocol: {config.protocol!r}. Available: {available}")
        return factory(config)


_DEFAULT_REGISTRY = ProviderRegistry()


def _openai_completions_factory(config: ProviderConfig) -> "Provider":
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
    return OpenAICompletions(**kwargs)  # type: ignore[arg-type]


def _openai_responses_factory(config: ProviderConfig) -> "Provider":
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


def _anthropic_factory(config: ProviderConfig) -> "Provider":
    kwargs: dict[str, object] = {
        "model": config.model,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.options:
        kwargs.update(config.options)
    return Anthropic(**kwargs)  # type: ignore[arg-type]


_DEFAULT_REGISTRY.register("openai-completions", _openai_completions_factory)
_DEFAULT_REGISTRY.register("openai-responses", _openai_responses_factory)
_DEFAULT_REGISTRY.register("anthropic", _anthropic_factory)


def create_provider(config: ProviderConfig, *, registry: ProviderRegistry | None = None) -> "Provider":
    """Create a provider from config.

    Uses the default registry unless a custom *registry* is supplied.
    """
    active_registry = registry or _DEFAULT_REGISTRY
    return active_registry.create(config)
