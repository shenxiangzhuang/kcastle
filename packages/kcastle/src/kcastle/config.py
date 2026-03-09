"""Configuration loading for kcastle.

Reads ``config.yaml`` from ``~/.kcastle/`` (or ``KCASTLE_HOME``).  String
values may reference environment variables via ``${VAR}`` syntax.

Provides ``CastleConfig`` — a typed dataclass consumed by ``Castle`` at
startup.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kcastle.providers import (
    ModelConfig,
    ProviderConfig,
    ProviderEntry,
    merge_builtin_providers,
    parse_providers,
)

_DEFAULT_HOME = Path.home() / ".kcastle"
_CONFIG_FILENAME = "config.yaml"
_DEFAULT_SYSTEM_PROMPT = ""
_ENV_VAR_RE = re.compile(r"\$\{(\w+)}")


# Re-export provider types so existing ``from kcastle.config import ...``
# statements keep working during the transition.
__all__ = [
    "CastleConfig",
    "ChannelConfig",
    "ModelConfig",
    "ProviderConfig",
    "ProviderEntry",
    "config_file_path",
    "load_config",
]


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


def _resolve_default_provider_name(provider_value: str) -> str:
    """Resolve default provider profile from provider ID."""
    provider_name = provider_value.strip()
    return provider_name.lower() if provider_name else ""


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
    merge_builtin_providers(raw_data)

    data: dict[str, Any] = _to_str_dict(_expand_env_recursive(raw_data))

    sessions_dir = home / "sessions"
    skills_dir = Path.home() / ".agent" / "skills"

    providers = parse_providers(data)

    default_section = _to_str_dict(data.get("default"))
    default_provider = str(default_section.get("provider", ""))
    default_model = str(default_section.get("model", ""))

    default_provider = os.environ.get("KCASTLE_PROVIDER", default_provider) or default_provider
    default_model = os.environ.get("KCASTLE_MODEL", default_model) or default_model
    default_provider = _resolve_default_provider_name(default_provider)

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
