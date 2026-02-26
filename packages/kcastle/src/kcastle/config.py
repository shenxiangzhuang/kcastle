"""Configuration loading for kcastle.

Reads ``config.yaml`` from ``~/.kcastle/`` (or ``KCASTLE_HOME``).  String
values may reference environment variables via ``${VAR}`` syntax.

Provides ``CastleConfig`` — a typed dataclass consumed by ``Castle`` at
startup.

Provider configuration
~~~~~~~~~~~~~~~~~~~~~~

Providers are defined as named entries under the top-level ``providers``
key.  Each provider specifies an API *protocol* (``openai``,
``openai-responses``, or ``anthropic``), endpoint, credentials, and a
model catalogue::

    providers:
      deepseek:
        protocol: openai
        base_url: https://api.deepseek.com
        api_key: ${DEEPSEEK_API_KEY}
        models:
          deepseek-chat:
            active: true
          deepseek-reasoner:
            active: true

The ``default`` section selects which provider/model to use::

    default:
      provider: deepseek
      model: deepseek-chat
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_HOME = Path.home() / ".kcastle"
_CONFIG_FILENAME = "config.yaml"
_DEFAULT_SYSTEM_PROMPT = ""
_ENV_VAR_RE = re.compile(r"\$\{(\w+)}")


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


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
class ProviderConfig:
    """Configuration for a single LLM provider.

    A *provider* maps to a concrete API endpoint (e.g. DeepSeek, Kimi,
    OpenAI, Anthropic).  The ``protocol`` field selects the kai driver
    class (``openai`` → ``OpenAICompletions``).
    """

    name: str
    """Provider name (e.g. ``deepseek``, ``kimi``, ``openai``)."""

    protocol: str = "openai"
    """API protocol: ``openai`` | ``openai-responses`` | ``anthropic``."""

    api_key: str = ""
    """API key for this provider."""

    base_url: str | None = None
    """Base URL for the API endpoint."""

    models: list[ModelConfig] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    """Available models for this provider."""

    def active_models(self) -> list[ModelConfig]:
        """Return only active models."""
        return [m for m in self.models if m.active]

    def get_model(self, model_id: str) -> ModelConfig | None:
        """Find a model by ID."""
        for m in self.models:
            if m.id == model_id:
                return m
        return None


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

    # Providers ----------------------------------------------------------
    providers: dict[str, ProviderConfig] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]
    """Configured providers keyed by name."""

    default_provider: str = ""
    """Name of the default provider."""

    default_model: str = ""
    """Default model identifier."""

    # Agent settings -----------------------------------------------------
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    """Base identity / persona prompt."""

    max_turns: int = 100
    """Maximum turns per agent run."""

    # Channels -----------------------------------------------------------
    cli: ChannelConfig = field(default_factory=ChannelConfig)
    telegram: ChannelConfig = field(
        default_factory=lambda: ChannelConfig(enabled=False),
    )

    telegram_token: str = ""
    """Telegram bot token."""

    # Helpers ------------------------------------------------------------

    def active_provider(self) -> ProviderConfig:
        """Return the currently selected provider config.

        Raises ``ValueError`` if ``default_provider`` is not configured.
        """
        if self.default_provider not in self.providers:
            available = list(self.providers.keys())
            raise ValueError(
                f"Default provider {self.default_provider!r} not found. Available: {available}"
            )
        return self.providers[self.default_provider]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------


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
            options: dict[str, object] = {
                str(k): v
                for k, v in mc.items()  # pyright: ignore[reportUnknownVariableType]
            }
            models.append(ModelConfig(id=mid, active=active, options=options))
        else:
            # Bare entry or ``model_id: true/false``
            active = bool(model_cfg) if model_cfg is not None else True  # pyright: ignore[reportUnknownArgumentType]
            models.append(ModelConfig(id=mid, active=active))
    return models


def _parse_providers(data: dict[str, Any]) -> dict[str, ProviderConfig]:
    """Parse the ``providers`` section."""
    raw: object = data.get("providers")
    if not isinstance(raw, dict):
        return {}
    providers: dict[str, ProviderConfig] = {}
    for name, cfg in raw.items():  # pyright: ignore[reportUnknownVariableType]
        name_str = str(name)  # pyright: ignore[reportUnknownArgumentType]
        cfg_dict = _to_str_dict(cfg)  # pyright: ignore[reportUnknownArgumentType]
        if not cfg_dict:
            continue
        base_url_val = cfg_dict.get("base_url")
        base_url: str | None = str(base_url_val) if base_url_val else None
        providers[name_str] = ProviderConfig(
            name=name_str,
            protocol=str(cfg_dict.get("protocol", "openai")),
            api_key=str(cfg_dict.get("api_key", "")),
            base_url=base_url,
            models=_parse_models(cfg_dict.get("models")),
        )
    return providers


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


# ---------------------------------------------------------------------------
# Built-in provider registry
# ---------------------------------------------------------------------------


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
        # --- DeepSeek ---------------------------------------------------
        "deepseek-openai": {
            "protocol": "openai",
            "base_url": "https://api.deepseek.com",
            "api_key": "${DEEPSEEK_API_KEY}",
            "models": dict(ds_models),
        },
        "deepseek-anthropic": {
            "protocol": "anthropic",
            "base_url": "https://api.deepseek.com/anthropic",
            "api_key": "${DEEPSEEK_API_KEY}",
            "models": dict(ds_models),
        },
        # --- MiniMax ----------------------------------------------------
        "minimax-openai": {
            "protocol": "openai",
            "base_url": "https://api.minimaxi.com/v1",
            "api_key": "${MINIMAX_API_KEY}",
            "models": dict(mm_models),
        },
        "minimax-anthropic": {
            "protocol": "anthropic",
            "base_url": "https://api.minimaxi.com/anthropic",
            "api_key": "${MINIMAX_API_KEY}",
            "models": dict(mm_models),
        },
    }


def _merge_builtin_providers(data: dict[str, Any]) -> None:
    """Merge built-in provider definitions into raw config data (in-place).

    Builtins form the base; user-provided providers override individual
    fields.  New user-defined providers are added as-is.
    """
    merged = _builtin_provider_dicts()
    user_providers: object = data.get("providers")
    if isinstance(user_providers, dict):
        for name, cfg in user_providers.items():  # pyright: ignore[reportUnknownVariableType]
            ns = str(name)  # pyright: ignore[reportUnknownArgumentType]
            if ns in merged and isinstance(cfg, dict):
                merged[ns] = {
                    **merged[ns],
                    **_to_str_dict(cfg),  # pyright: ignore[reportUnknownArgumentType]
                }
            elif isinstance(cfg, dict):
                merged[ns] = _to_str_dict(cfg)  # pyright: ignore[reportUnknownArgumentType]
    data["providers"] = merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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

    # --- Providers ------------------------------------------------------
    providers = _parse_providers(data)

    # --- Default provider / model ---------------------------------------
    default_section = _to_str_dict(data.get("default"))
    default_provider = str(default_section.get("provider", ""))
    default_model = str(default_section.get("model", ""))

    # Env overrides
    default_provider = os.environ.get("KCASTLE_PROVIDER", default_provider) or default_provider
    default_model = os.environ.get("KCASTLE_MODEL", default_model) or default_model

    # --- Agent ----------------------------------------------------------
    agent = _to_str_dict(data.get("agent"))
    system_prompt = str(agent.get("system_prompt", _DEFAULT_SYSTEM_PROMPT))
    max_turns = int(agent.get("max_turns", 100))

    # --- Channels -------------------------------------------------------
    cli_cfg = _parse_channel(data, "cli", default_enabled=True)
    tg_cfg = _parse_channel(data, "telegram", default_enabled=False)

    # Telegram token: channel option → env var
    tg_token_opt: object = tg_cfg.options.get("token")
    tg_token = str(tg_token_opt) if tg_token_opt else ""  # pyright: ignore[reportUnknownArgumentType]
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
