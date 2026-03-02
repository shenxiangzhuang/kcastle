"""Configuration loading for kcastle.

Reads ``config.yaml`` from ``~/.kcastle/`` (or ``KCASTLE_HOME``).  String
values may reference environment variables via ``${VAR}`` syntax.

Provides ``CastleConfig`` — a typed dataclass consumed by ``Castle`` at
startup.

Provider configuration
~~~~~~~~~~~~~~~~~~~~~~

Providers are defined by *vendor* entries with nested ``protocols``
mapping. Each protocol profile specifies API protocol
(``openai-completions``, ``openai-responses``, or ``anthropic``), endpoint,
credentials, and model catalogue::

providers:
    deepseek:
        protocols:
            openai-completions:
                base_url: https://api.deepseek.com
                api_key: ${DEEPSEEK_API_KEY}
                models:
                    deepseek-chat:
                        active: true
            anthropic:
                base_url: https://api.deepseek.com/anthropic
                api_key: ${DEEPSEEK_API_KEY}
                models:
                    deepseek-chat:
                        active: true

The ``default`` section selects which provider/model to use::

    default:
      provider: deepseek
      model: deepseek-chat
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import yaml

from kcastle.provider_config import ProviderConfig

_DEFAULT_HOME = Path.home() / ".kcastle"
_CONFIG_FILENAME = "config.yaml"
_DEFAULT_SYSTEM_PROMPT = ""
_ENV_VAR_RE = re.compile(r"\$\{(\w+)}")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for a single model within a provider."""

    id: str
    """Model identifier (e.g. ``deepseek-chat``, ``gpt-4o``)."""

    active: bool = True
    """Whether this model is available for use."""

    options: dict[str, object] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]
    """Provider-specific model options (``max_tokens``, ``reasoning``, etc.)."""


@dataclass(frozen=True, slots=True)
class ProviderEntry:
    """Configuration entry for one provider profile.

    Keeps runtime provider construction fields in ``provider`` (kcastle-owned)
    and catalog-only fields (model list) in this entry.
    """

    provider: ProviderConfig
    """Runtime provider config owned by kcastle."""

    models: list[ModelConfig] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    """Available models for this provider."""

    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def vendor(self) -> str:
        return self.provider.vendor

    @property
    def protocol(self) -> str:
        return self.provider.protocol

    @property
    def api_key(self) -> str:
        return self.provider.api_key or ""

    @property
    def base_url(self) -> str | None:
        return self.provider.base_url

    @property
    def extra_body(self) -> dict[str, object] | None:
        return self.provider.extra_body

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
            self.provider,
            model=model_id,
            options=dict(model_cfg.options),
        )


@dataclass(frozen=True, slots=True)
class ChannelConfig:
    """Configuration for a single channel."""

    enabled: bool = True
    options: dict[str, object] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]


@dataclass(frozen=True, slots=True)
class CastleConfig:
    """Typed configuration for the kcastle application."""

    home: Path
    """Root directory for kcastle data (default: ``~/.kcastle``)."""

    sessions_dir: Path
    """Directory for session data."""

    skills_dir: Path
    """User-level skills directory."""

    providers: dict[str, ProviderEntry] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]
    """Configured providers keyed by name."""

    default_provider: str = ""
    """Name of the default provider."""

    default_model: str = ""
    """Default model identifier."""

    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    """Base identity / persona prompt."""

    max_turns: int = 100
    """Maximum turns per agent run."""

    cli: ChannelConfig = field(default_factory=ChannelConfig)
    telegram: ChannelConfig = field(
        default_factory=lambda: ChannelConfig(enabled=False),
    )

    telegram_token: str = ""
    """Telegram bot token."""

    def active_provider(self) -> ProviderEntry:
        """Return the currently selected provider config.

        Raises ``ValueError`` if ``default_provider`` is not configured.
        """
        if self.default_provider not in self.providers:
            available = list(self.providers.keys())
            raise ValueError(
                f"Default provider {self.default_provider!r} not found. Available: {available}"
            )
        return self.providers[self.default_provider]

    def provider_config(self, provider_name: str, model_id: str) -> ProviderConfig:
        """Build a provider config from explicit provider/model selection."""
        if provider_name not in self.providers:
            raise ValueError(f"Unknown provider: {provider_name!r}")
        return self.providers[provider_name].to_provider_config(model_id)

    def active_provider_config(self) -> ProviderConfig:
        """Build a provider config for the active provider/model selection."""
        return self.active_provider().to_provider_config(self.default_model)


def _expand_env(value: str) -> str:
    """Expand ``${VAR}`` references in a string value."""

    def _replacer(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), "")

    return _ENV_VAR_RE.sub(_replacer, value)


def _expand_env_recursive(obj: object) -> object:
    """Recursively expand ``${VAR}`` in all string values."""
    if isinstance(obj, str):
        return _expand_env(obj)
    if isinstance(obj, dict):
        return {  # pyright: ignore[reportUnknownVariableType]
            k: _expand_env_recursive(v)  # pyright: ignore[reportUnknownArgumentType]
            for k, v in obj.items()  # pyright: ignore[reportUnknownVariableType]
        }
    if isinstance(obj, list):
        return [
            _expand_env_recursive(v)  # pyright: ignore[reportUnknownArgumentType]
            for v in obj  # pyright: ignore[reportUnknownVariableType]
        ]
    return obj


def _resolve_home(override: Path | None = None) -> Path:
    """Determine the kcastle home directory."""
    if override:
        return override
    env = os.environ.get("KCASTLE_HOME")
    return Path(env) if env else _DEFAULT_HOME


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file.  Returns empty dict if not found."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    result: object = yaml.safe_load(text)
    if not isinstance(result, dict):
        return {}
    return dict(result)  # pyright: ignore[reportUnknownArgumentType]


def _to_str_dict(d: object) -> dict[str, Any]:
    """Coerce an untyped dict from YAML into ``dict[str, Any]``."""
    if not isinstance(d, dict):
        return {}
    return {
        str(k): v  # pyright: ignore[reportUnknownArgumentType]
        for k, v in d.items()  # pyright: ignore[reportUnknownVariableType]
    }


def _parse_models(raw: object) -> list[ModelConfig]:
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


def _provider_profile_name(vendor: str, protocol: str) -> str:
    """Build canonical provider profile name from vendor + protocol."""
    return f"{vendor}-{protocol.lower()}"


def _build_provider_config(
    *,
    name: str,
    vendor: str,
    protocol: str,
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
        provider=ProviderConfig(
            vendor=vendor,
            protocol=protocol,
            model="",
            api_key=str(cfg_dict.get("api_key", "")) or None,
            base_url=base_url,
            extra_body=extra_body,
            options={},
        ),
        models=_parse_models(cfg_dict.get("models")),
    )


def _parse_providers(data: dict[str, Any]) -> dict[str, ProviderEntry]:
    """Parse the ``providers`` section."""
    raw: object = data.get("providers")
    if not isinstance(raw, dict):
        return {}
    providers: dict[str, ProviderEntry] = {}
    for name, cfg in raw.items():  # pyright: ignore[reportUnknownVariableType]
        vendor = str(name)  # pyright: ignore[reportUnknownArgumentType]
        cfg_dict = _to_str_dict(cfg)  # pyright: ignore[reportUnknownArgumentType]
        if not cfg_dict:
            continue

        protocols_raw = cfg_dict.get("protocols")
        if not isinstance(protocols_raw, dict):
            raise ValueError(f"Provider vendor {vendor!r} must define a 'protocols' mapping")

        common: dict[str, Any] = {k: v for k, v in cfg_dict.items() if k != "protocols"}
        for protocol_name, protocol_cfg in protocols_raw.items():  # pyright: ignore[reportUnknownVariableType]
            protocol = str(protocol_name).lower()  # pyright: ignore[reportUnknownArgumentType]
            protocol_dict = _to_str_dict(protocol_cfg)  # pyright: ignore[reportUnknownArgumentType]
            merged_cfg = {**common, **protocol_dict, "protocol": protocol}
            profile_name = _provider_profile_name(vendor, protocol)
            providers[profile_name] = _build_provider_config(
                name=profile_name,
                vendor=vendor,
                protocol=protocol,
                cfg_dict=merged_cfg,
            )
    return providers


def _resolve_default_provider_name(provider_value: str, protocol_value: str) -> str:
    """Resolve default provider profile from vendor + protocol."""
    provider_name = provider_value.strip()
    if not provider_name:
        return ""
    protocol_name = protocol_value.strip().lower() or "openai-completions"
    return _provider_profile_name(provider_name, protocol_name)


def _parse_channel(
    data: dict[str, Any],
    channel: str,
    *,
    default_enabled: bool = False,
) -> ChannelConfig:
    """Parse a single channel section."""
    channels_raw: object = data.get("channels")
    if not isinstance(channels_raw, dict):
        return ChannelConfig(enabled=default_enabled)
    section = _to_str_dict(
        channels_raw.get(channel),  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    )
    if not section:
        return ChannelConfig(enabled=default_enabled)
    enabled = bool(section.pop("enabled", default_enabled))
    options: dict[str, object] = dict(section)
    return ChannelConfig(enabled=enabled, options=options)


def _builtin_provider_dicts() -> dict[str, dict[str, Any]]:
    """Built-in provider definitions.

    Each supported vendor is registered twice — once for the OpenAI protocol
    and once for the Anthropic protocol.  User config can override any field;
    missing fields fall back to these defaults.
    """
    ds_models: dict[str, object] = {
        "deepseek-chat": {"active": True},
        "deepseek-reasoner": {"active": True},
    }
    mm_models: dict[str, object] = {
        "MiniMax-M2.5": {"active": True},
        "MiniMax-M2.5-highspeed": {"active": True},
        "MiniMax-M2": {"active": True},
    }
    return {
        "deepseek": {
            "api_key": "${DEEPSEEK_API_KEY}",
            "models": dict(ds_models),
            "protocols": {
                "openai-completions": {
                    "base_url": "https://api.deepseek.com",
                },
                "anthropic": {
                    "base_url": "https://api.deepseek.com/anthropic",
                },
            },
        },
        "minimax": {
            "api_key": "${MINIMAX_API_KEY}",
            "models": dict(mm_models),
            "protocols": {
                "openai-completions": {
                    "base_url": "https://api.minimaxi.com/v1",
                    # MiniMax OpenAI endpoint embeds thinking as <think> tags in
                    # content by default.  Setting reasoning_split=True instructs
                    # the API to separate thinking into the reasoning_details field.
                    "extra_body": {"reasoning_split": True},
                },
                "anthropic": {
                    "base_url": "https://api.minimaxi.com/anthropic",
                },
            },
        },
    }


def _merge_builtin_providers(data: dict[str, Any]) -> None:
    """Merge built-in provider definitions into raw config data (in-place).

    Builtins form the base; user-provided providers override individual
    fields.  New user-defined providers are added as-is.
    """
    merged: dict[str, dict[str, Any]] = _builtin_provider_dicts()
    user_providers: object = data.get("providers")
    if isinstance(user_providers, dict):
        for vendor, cfg in user_providers.items():  # pyright: ignore[reportUnknownVariableType]
            vendor_name = str(vendor)  # pyright: ignore[reportUnknownArgumentType]
            if not isinstance(cfg, dict):
                continue

            user_cfg = _to_str_dict(cfg)  # pyright: ignore[reportUnknownArgumentType]
            if vendor_name not in merged:
                merged[vendor_name] = user_cfg
                continue

            base_cfg = merged[vendor_name]
            merged_vendor: dict[str, Any] = {
                **base_cfg,
                **{k: v for k, v in user_cfg.items() if k != "protocols"},
            }

            base_protocols = _to_str_dict(base_cfg.get("protocols"))
            user_protocols_raw = user_cfg.get("protocols")
            if isinstance(user_protocols_raw, dict):
                user_protocols = {
                    str(k): v for k, v in cast(dict[object, object], user_protocols_raw).items()
                }
                protocol_names = set(base_protocols.keys()) | set(user_protocols.keys())
                merged_protocols: dict[str, Any] = {}
                for protocol_name in protocol_names:
                    base_protocol_cfg = _to_str_dict(base_protocols.get(protocol_name))
                    user_protocol_cfg = _to_str_dict(user_protocols.get(protocol_name))
                    merged_protocols[protocol_name] = {
                        **base_protocol_cfg,
                        **user_protocol_cfg,
                    }
                merged_vendor["protocols"] = merged_protocols
            else:
                merged_vendor["protocols"] = base_protocols

            merged[vendor_name] = merged_vendor
    data["providers"] = merged


def config_file_path(home: Path | None = None) -> Path:
    """Return the path to the configuration file."""
    return _resolve_home(home) / _CONFIG_FILENAME


def load_config(home: Path | None = None) -> CastleConfig:
    """Load kcastle configuration from YAML and environment.

    Resolution order (highest priority first):

    1. Environment variables (``${VAR}`` in YAML strings, ``KCASTLE_*``
       overrides)
    2. ``config.yaml`` values
    3. Built-in defaults
    """
    home = _resolve_home(home)
    raw_data = _read_yaml(home / _CONFIG_FILENAME)

    # Merge built-in providers before env var expansion so that
    # ${VAR} references in builtins get expanded too.
    _merge_builtin_providers(raw_data)

    data: dict[str, Any] = _to_str_dict(_expand_env_recursive(raw_data))

    sessions_dir = home / "sessions"
    skills_dir = home / "skills"

    providers = _parse_providers(data)

    default_section = _to_str_dict(data.get("default"))
    default_provider = str(default_section.get("provider", ""))
    default_protocol = str(default_section.get("protocol", ""))
    default_model = str(default_section.get("model", ""))

    default_provider = os.environ.get("KCASTLE_PROVIDER", default_provider) or default_provider
    default_protocol = os.environ.get("KCASTLE_PROTOCOL", default_protocol) or default_protocol
    default_model = os.environ.get("KCASTLE_MODEL", default_model) or default_model
    default_provider = _resolve_default_provider_name(default_provider, default_protocol)

    agent = _to_str_dict(data.get("agent"))
    system_prompt = str(agent.get("system_prompt", _DEFAULT_SYSTEM_PROMPT))
    max_turns = int(agent.get("max_turns", 100))

    cli_cfg = _parse_channel(data, "cli", default_enabled=True)
    tg_cfg = _parse_channel(data, "telegram", default_enabled=False)

    # Telegram token: channel option → env var
    tg_token_opt: object = tg_cfg.options.get("token")
    tg_token = str(tg_token_opt) if tg_token_opt else ""
    tg_token = os.environ.get("KCASTLE_TG_TOKEN", tg_token) or tg_token

    return CastleConfig(
        home=home,
        sessions_dir=sessions_dir,
        skills_dir=skills_dir,
        providers=providers,
        default_provider=default_provider,
        default_model=default_model,
        system_prompt=system_prompt,
        max_turns=max_turns,
        cli=cli_cfg,
        telegram=tg_cfg,
        telegram_token=tg_token,
    )
