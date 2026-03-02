from __future__ import annotations

from pathlib import Path

import pytest

from kcastle.config import load_config


def _write_config(home: Path, text: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text, encoding="utf-8")


def test_load_config_rejects_legacy_flat_provider(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
providers:
  deepseek-openai:
    protocol: openai-completions
    api_key: sk-test
    base_url: https://api.deepseek.com
    models:
      deepseek-chat:
        active: true

default:
  provider: deepseek-openai-completions
  model: deepseek-chat
""",
    )

    with pytest.raises(ValueError, match="must define a 'protocols' mapping"):
        load_config(home=tmp_path)


def test_load_config_vendor_protocol_profiles(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
providers:
  deepseek:
    api_key: sk-test
    models:
      deepseek-chat:
        active: true
    protocols:
      openai-completions:
        base_url: https://api.deepseek.com
      anthropic:
        base_url: https://api.deepseek.com/anthropic

default:
  provider: deepseek
  protocol: anthropic
  model: deepseek-chat
""",
    )

    cfg = load_config(home=tmp_path)

    assert "deepseek-openai-completions" in cfg.providers
    assert "deepseek-anthropic" in cfg.providers

    openai_profile = cfg.providers["deepseek-openai-completions"]
    anthropic_profile = cfg.providers["deepseek-anthropic"]

    assert openai_profile.vendor == "deepseek"
    assert anthropic_profile.vendor == "deepseek"
    assert openai_profile.protocol == "openai-completions"
    assert anthropic_profile.protocol == "anthropic"
    assert openai_profile.base_url == "https://api.deepseek.com"
    assert anthropic_profile.base_url == "https://api.deepseek.com/anthropic"
    assert cfg.default_provider == "deepseek-anthropic"


def test_vendor_protocol_profile_overrides_builtin(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
providers:
  deepseek:
    protocols:
      openai-completions:
        api_key: sk-custom
        base_url: https://custom.deepseek.local

default:
  provider: deepseek
  protocol: openai-completions
  model: deepseek-chat
""",
    )

    cfg = load_config(home=tmp_path)
    provider = cfg.providers["deepseek-openai-completions"]

    assert provider.vendor == "deepseek"
    assert provider.protocol == "openai-completions"
    assert provider.api_key == "sk-custom"
    assert provider.base_url == "https://custom.deepseek.local"


def test_default_protocol_falls_back_to_openai(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
providers:
  deepseek:
    protocols:
      openai-completions:
        api_key: sk-test
        base_url: https://api.deepseek.com
        models:
          deepseek-chat:
            active: true

default:
  provider: deepseek
  model: deepseek-chat
""",
    )

    cfg = load_config(home=tmp_path)
    assert cfg.default_provider == "deepseek-openai-completions"


def test_provider_entry_to_provider_config(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
providers:
  deepseek:
    protocols:
      openai-completions:
        api_key: sk-test
        base_url: https://api.deepseek.com
        models:
          deepseek-chat:
            active: true
          deepseek-reasoner:
            active: true

default:
  provider: deepseek
  protocol: openai-completions
  model: deepseek-chat
""",
    )

    cfg = load_config(home=tmp_path)
    provider = cfg.providers["deepseek-openai-completions"]
    provider_config = provider.to_provider_config("deepseek-reasoner")

    assert provider_config.vendor == "deepseek"
    assert provider_config.protocol == "openai-completions"
    assert provider_config.model == "deepseek-reasoner"
    assert provider_config.base_url == "https://api.deepseek.com"


def test_castle_config_provider_config_helpers(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
providers:
  deepseek:
    protocols:
      openai-completions:
        api_key: sk-test
        base_url: https://api.deepseek.com
        models:
          deepseek-chat:
            active: true
      anthropic:
        api_key: sk-test
        base_url: https://api.deepseek.com/anthropic
        models:
          deepseek-chat:
            active: true

default:
  provider: deepseek
  protocol: anthropic
  model: deepseek-chat
""",
    )

    cfg = load_config(home=tmp_path)

    active_config = cfg.active_provider_config()
    assert active_config.vendor == "deepseek"
    assert active_config.protocol == "anthropic"

    explicit_config = cfg.provider_config("deepseek-openai-completions", "deepseek-chat")
    assert explicit_config.protocol == "openai-completions"


def test_active_provider_resolves_entry(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
providers:
  deepseek:
    protocols:
      openai-completions:
        api_key: sk-test
        base_url: https://api.deepseek.com
        models:
          deepseek-chat:
            active: true

default:
  provider: deepseek
  protocol: openai-completions
  model: deepseek-chat
""",
    )

    cfg = load_config(home=tmp_path)
    active = cfg.active_provider()

    assert active.vendor == "deepseek"
    assert active.protocol == "openai-completions"
