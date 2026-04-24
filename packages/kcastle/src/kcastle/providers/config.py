"""Provider configuration types for kcastle.

Contains the typed dataclasses that describe provider identity, model
catalogues, and construction parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Provider construction config.

    A config captures provider identity, model, and endpoint/auth
    options needed to construct a concrete provider instance.
    """

    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    extra_body: dict[str, object] | None = None
    options: dict[str, object] = field(default_factory=dict[str, object])

    @property
    def name(self) -> str:
        """Canonical profile name, e.g. ``deepseek-openai``."""
        return self.provider.lower()


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for a single model within a provider."""

    id: str
    """Model identifier (e.g. ``deepseek-v4-flash``, ``gpt-4o``)."""

    active: bool = True
    """Whether this model is available for use."""

    options: dict[str, object] = field(default_factory=dict[str, object])
    """Provider-specific model options (``max_tokens``, ``reasoning``, etc.)."""


@dataclass(frozen=True, slots=True)
class ProviderEntry:
    """Configuration entry for one provider profile.

    Keeps runtime provider construction fields in ``config`` (a
    :class:`ProviderConfig`) and catalog-only fields (model list) here.
    """

    config: ProviderConfig
    """Runtime provider config."""

    models: list[ModelConfig] = field(default_factory=list[ModelConfig])
    """Available models for this provider."""

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def api_key(self) -> str:
        return self.config.api_key or ""

    @property
    def base_url(self) -> str | None:
        return self.config.base_url

    @property
    def extra_body(self) -> dict[str, object] | None:
        return self.config.extra_body

    def active_models(self) -> list[ModelConfig]:
        """Return only active models."""
        return [m for m in self.models if m.active]

    def get_model(self, model_id: str) -> ModelConfig | None:
        """Find a model by ID."""
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def to_provider_config(self, model_id: str) -> ProviderConfig:
        """Build a provider config for the given model.

        Raises:
            ValueError: If the requested model is not available in this provider.
        """
        model_cfg = self.get_model(model_id)
        if model_cfg is None:
            raise ValueError(f"Unknown model: {model_id!r} in provider {self.name!r}")

        return replace(
            self.config,
            model=model_id,
            options=dict(model_cfg.options),
        )
