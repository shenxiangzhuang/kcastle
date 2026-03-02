from __future__ import annotations

import pytest
from kai import ProviderProfile, ProviderRegistry, create_provider
from kai.providers.openai import OpenAICompletions, OpenAIResponses


def test_create_provider_openai_completions() -> None:
    provider = create_provider(
        ProviderProfile(
            vendor="deepseek",
            protocol="openai-completions",
            model="deepseek-chat",
            api_key="test-key",
            base_url="https://api.deepseek.com",
        )
    )
    assert isinstance(provider, OpenAICompletions)
    assert provider.model == "deepseek-chat"


def test_create_provider_openai_responses() -> None:
    provider = create_provider(
        ProviderProfile(
            vendor="openai",
            protocol="openai-responses",
            model="gpt-4.1",
            api_key="test-key",
            options={"reasoning": {"effort": "medium"}},
        )
    )
    assert isinstance(provider, OpenAIResponses)
    assert provider.model == "gpt-4.1"


def test_create_provider_unknown_protocol_raises() -> None:
    with pytest.raises(ValueError, match="Unknown protocol"):
        create_provider(
            ProviderProfile(
                vendor="x",
                protocol="unknown-protocol",
                model="m",
            )
        )


def test_provider_registry_register() -> None:
    class DummyProvider:
        def __init__(self) -> None:
            self._model = "dummy-model"

        @property
        def name(self) -> str:
            return "dummy"

        @property
        def model(self) -> str:
            return self._model

        async def stream_raw(self, context, **kwargs):  # type: ignore[no-untyped-def]
            if False:
                yield None

    def _factory(profile: ProviderProfile) -> DummyProvider:
        assert profile.protocol == "dummy"
        return DummyProvider()

    registry = ProviderRegistry()
    registry.register("dummy", _factory)
    provider = create_provider(
        ProviderProfile(vendor="x", protocol="dummy", model="m"),
        registry=registry,
    )
    assert provider.name == "dummy"
