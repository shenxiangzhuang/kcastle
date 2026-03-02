from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from kcastle.provider_config import ProviderConfig


def test_provider_config_name_uses_lower_protocol() -> None:
    cfg = ProviderConfig(
        vendor="deepseek",
        protocol="OPENAI-COMPLETIONS",
        model="deepseek-chat",
    )
    assert cfg.name == "deepseek-openai-completions"


def test_provider_config_options_default_independent() -> None:
    a = ProviderConfig(vendor="a", protocol="openai-completions", model="m1")
    b = ProviderConfig(vendor="b", protocol="openai-completions", model="m2")

    a.options["x"] = 1
    assert b.options == {}


def test_provider_config_is_frozen() -> None:
    cfg = ProviderConfig(vendor="x", protocol="openai-completions", model="m")
    with pytest.raises(FrozenInstanceError):
        cfg.vendor = "y"  # type: ignore[misc]
