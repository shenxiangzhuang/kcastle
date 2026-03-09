from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from kcastle.providers import ProviderConfig


def test_provider_config_name_uses_lower_provider() -> None:
    cfg = ProviderConfig(
        provider="DeepSeek-OpenAI",
        model="deepseek-chat",
    )
    assert cfg.name == "deepseek-openai"


def test_provider_config_options_default_independent() -> None:
    a = ProviderConfig(provider="a", model="m1")
    b = ProviderConfig(provider="b", model="m2")

    a.options["x"] = 1
    assert b.options == {}


def test_provider_config_is_frozen() -> None:
    cfg = ProviderConfig(provider="x", model="m")
    with pytest.raises(FrozenInstanceError):
        cfg.provider = "y"  # type: ignore[misc]
