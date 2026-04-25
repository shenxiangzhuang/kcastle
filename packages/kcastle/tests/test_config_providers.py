from __future__ import annotations

from pathlib import Path

import pytest

from kcastle.config import load_config


def _write_config(home: Path, text: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text, encoding="utf-8")


def test_load_config_accepts_explicit_provider_profiles(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
providers:
  deepseek-openai:
    api_key: sk-test
    base_url: https://api.deepseek.com
    models:
      deepseek-v4-flash:
        active: true

default:
  provider: deepseek-openai
  model: deepseek-v4-flash
""",
    )

    cfg = load_config(home=tmp_path)
    assert "deepseek-openai" in cfg.providers


def test_load_config_provider_profiles(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
providers:
  deepseek-openai:
    api_key: sk-test
    base_url: https://api.deepseek.com
    models:
      deepseek-v4-flash:
        active: true
  deepseek-anthropic:
    api_key: sk-test
    base_url: https://api.deepseek.com/anthropic
    models:
      deepseek-v4-flash:
        active: true

default:
  provider: deepseek-anthropic
  model: deepseek-v4-flash
""",
    )

    cfg = load_config(home=tmp_path)

    assert "deepseek-openai" in cfg.providers
    assert "deepseek-anthropic" in cfg.providers

    openai_profile = cfg.providers["deepseek-openai"]
    anthropic_profile = cfg.providers["deepseek-anthropic"]

    assert openai_profile.provider == "deepseek-openai"
    assert anthropic_profile.provider == "deepseek-anthropic"
    assert openai_profile.base_url == "https://api.deepseek.com"
    assert anthropic_profile.base_url == "https://api.deepseek.com/anthropic"
    assert cfg.default_provider == "deepseek-anthropic"


def test_provider_profile_overrides_builtin(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
providers:
  deepseek-openai:
    api_key: sk-custom
    base_url: https://custom.deepseek.local

default:
  provider: deepseek-openai
  model: deepseek-v4-flash
""",
    )

    cfg = load_config(home=tmp_path)
    provider = cfg.providers["deepseek-openai"]

    assert provider.provider == "deepseek-openai"
    assert provider.api_key == "sk-custom"
    assert provider.base_url == "https://custom.deepseek.local"


def test_default_provider_is_used_directly(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
providers:
  deepseek-openai:
    api_key: sk-test
    base_url: https://api.deepseek.com
    models:
      deepseek-v4-flash:
        active: true

default:
  provider: deepseek-openai
  model: deepseek-v4-flash
""",
    )

    cfg = load_config(home=tmp_path)
    assert cfg.default_provider == "deepseek-openai"


def test_provider_entry_to_provider_config(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
providers:
  deepseek-openai:
    api_key: sk-test
    base_url: https://api.deepseek.com
    models:
      deepseek-v4-flash:
        active: true
      deepseek-v4-pro:
        active: true

default:
  provider: deepseek-openai
  model: deepseek-v4-flash
""",
    )

    cfg = load_config(home=tmp_path)
    provider = cfg.providers["deepseek-openai"]
    provider_config = provider.to_provider_config("deepseek-v4-pro")

    assert provider_config.provider == "deepseek-openai"
    assert provider_config.model == "deepseek-v4-pro"
    assert provider_config.base_url == "https://api.deepseek.com"


def test_castle_config_provider_config_helpers(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
providers:
  deepseek-openai:
    api_key: sk-test
    base_url: https://api.deepseek.com
    models:
      deepseek-v4-flash:
        active: true
  deepseek-anthropic:
    api_key: sk-test
    base_url: https://api.deepseek.com/anthropic
    models:
      deepseek-v4-flash:
        active: true

default:
  provider: deepseek-anthropic
  model: deepseek-v4-flash
""",
    )

    cfg = load_config(home=tmp_path)

    active_config = cfg.active_provider_config()
    assert active_config.provider == "deepseek-anthropic"

    explicit_config = cfg.provider_config("deepseek-openai", "deepseek-v4-flash")
    assert explicit_config.provider == "deepseek-openai"


def test_active_provider_resolves_entry(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
providers:
  deepseek-openai:
    api_key: sk-test
    base_url: https://api.deepseek.com
    models:
      deepseek-v4-flash:
        active: true

default:
  provider: deepseek-openai
  model: deepseek-v4-flash
""",
    )

    cfg = load_config(home=tmp_path)
    active = cfg.active_provider()

    assert active.provider == "deepseek-openai"


def test_load_config_otel_disabled_by_default(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
providers:
  deepseek-openai:
    api_key: sk-test

default:
  provider: deepseek-openai
  model: deepseek-v4-flash
""",
    )

    cfg = load_config(home=tmp_path)

    assert cfg.otel_endpoint == ""


def test_load_config_reads_otel_endpoint_from_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        """
providers:
  deepseek-openai:
    api_key: sk-test

default:
  provider: deepseek-openai
  model: deepseek-v4-flash
""",
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel.example:4317")

    cfg = load_config(home=tmp_path)

    assert cfg.otel_endpoint == "http://otel.example:4317"


def test_load_config_reads_telegram_token_env_without_enabling_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        """
default:
  provider: deepseek-openai
  model: deepseek-v4-flash
""",
    )
    monkeypatch.setenv("KCASTLE_TG_TOKEN", "test-token")

    cfg = load_config(home=tmp_path)

    assert not cfg.telegram.enabled
    assert cfg.telegram_token == "test-token"


def test_load_config_reads_telegram_token_env_when_explicitly_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        """
default:
  provider: deepseek-openai
  model: deepseek-v4-flash
channels:
  telegram:
    enabled: false
""",
    )
    monkeypatch.setenv("KCASTLE_TG_TOKEN", "test-token")

    cfg = load_config(home=tmp_path)

    assert not cfg.telegram.enabled
    assert cfg.telegram_token == "test-token"
