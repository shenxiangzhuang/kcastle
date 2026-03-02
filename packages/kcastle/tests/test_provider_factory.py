from __future__ import annotations

import pytest
from kai.providers import ProviderBase
from kai.providers.deepseek import DeepseekOpenAI
from kai.providers.openai import OpenAIResponses

from kcastle.provider_config import ProviderConfig
from kcastle.provider_factory import ProviderRegistry, create_provider


def test_create_provider_openai_completions() -> None:
    provider = create_provider(
        ProviderConfig(
            provider="deepseek-openai",
            model="deepseek-chat",
            api_key="test-key",
            base_url="https://api.deepseek.com",
        )
    )
    assert isinstance(provider, DeepseekOpenAI)
    assert provider.model == "deepseek-chat"


def test_create_provider_openai_responses() -> None:
    provider = create_provider(
        ProviderConfig(
            provider="openai-responses",
            model="gpt-4.1",
            api_key="test-key",
            options={"reasoning": {"effort": "medium"}},
        )
    )
    assert isinstance(provider, OpenAIResponses)
    assert provider.model == "gpt-4.1"


def test_create_provider_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider(
            ProviderConfig(
                provider="unknown-provider",
                model="m",
            )
        )


def test_provider_registry_register() -> None:
    class DummyProvider(ProviderBase):
        def __init__(self) -> None:
            self._model = "dummy-model"

        @property
        def provider(self) -> str:
            return "dummy"

        @property
        def model(self) -> str:
            return self._model

        async def stream_raw(self, context, **kwargs):  # type: ignore[no-untyped-def]
            if False:
                yield None

    def _factory(config: ProviderConfig) -> DummyProvider:
        assert config.provider == "dummy"
        return DummyProvider()

    registry = ProviderRegistry()
    registry.register("dummy", _factory)
    provider = create_provider(
        ProviderConfig(provider="dummy", model="m"),
        registry=registry,
    )
    assert provider.provider == "dummy"
