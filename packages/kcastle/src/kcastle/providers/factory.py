"""LLM factory and registry for kcastle application layer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kai import ProviderBase
from kai.providers.anthropic import AnthropicMessages
from kai.providers.deepseek import DeepseekAnthropic, DeepseekOpenAI
from kai.providers.minimax import MinimaxAnthropic, MinimaxOpenAI
from kai.providers.openai import (
    OpenAIChatCompletions,
    OpenAIResponses,
)

from kcastle.providers.config import ModelConfig, ProviderConfig, ProviderEntry

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


def _to_str_dict(d: object) -> dict[str, Any]:
    """Coerce an untyped dict from YAML into ``dict[str, Any]``."""
    if not isinstance(d, dict):
        return {}
    return {
        str(k): v  # pyright: ignore[reportUnknownArgumentType]
        for k, v in d.items()  # pyright: ignore[reportUnknownVariableType]
    }


def parse_models(raw: object) -> list[ModelConfig]:
    """Parse the ``models`` mapping inside a provider."""
    if not isinstance(raw, dict):
        return []
    models: list[ModelConfig] = []
    for model_id, model_cfg in raw.items():  # pyright: ignore[reportUnknownVariableType]
        mid = str(model_id)  # pyright: ignore[reportUnknownArgumentType]
        if isinstance(model_cfg, dict):
            mc = _to_str_dict(model_cfg)  # pyright: ignore[reportUnknownArgumentType]
            active = bool(mc.pop("active", True))
            options: dict[str, object] = {str(k): v for k, v in mc.items()}
            models.append(ModelConfig(id=mid, active=active, options=options))
        else:
            # Bare entry or ``model_id: true/false``
            active = bool(model_cfg) if model_cfg is not None else True  # pyright: ignore[reportUnknownArgumentType]
            models.append(ModelConfig(id=mid, active=active))
    return models


def build_provider_entry(
    *,
    provider_name: str,
    cfg_dict: dict[str, Any],
) -> ProviderEntry:
    """Build a typed provider profile from an untyped mapping."""
    base_url_val = cfg_dict.get("base_url")
    base_url: str | None = str(base_url_val) if base_url_val else None
    extra_body_val = cfg_dict.get("extra_body")
    extra_body: dict[str, object] | None = (
        _to_str_dict(extra_body_val)  # pyright: ignore[reportUnknownArgumentType]
        if isinstance(extra_body_val, dict)
        else None
    )
    return ProviderEntry(
        config=ProviderConfig(
            provider=provider_name,
            model="",
            api_key=str(cfg_dict.get("api_key", "")) or None,
            base_url=base_url,
            extra_body=extra_body,
            options={},
        ),
        models=parse_models(cfg_dict.get("models")),
    )


def parse_providers(data: dict[str, Any]) -> dict[str, ProviderEntry]:
    """Parse the ``providers`` section of a config dict."""
    raw: object = data.get("providers")
    if not isinstance(raw, dict):
        return {}
    providers: dict[str, ProviderEntry] = {}
    for name, cfg in raw.items():  # pyright: ignore[reportUnknownVariableType]
        provider_name = str(name).lower()  # pyright: ignore[reportUnknownArgumentType]
        cfg_dict = _to_str_dict(cfg)  # pyright: ignore[reportUnknownArgumentType]
        if not cfg_dict:
            continue
        providers[provider_name] = build_provider_entry(
            provider_name=provider_name,
            cfg_dict=cfg_dict,
        )
    return providers
