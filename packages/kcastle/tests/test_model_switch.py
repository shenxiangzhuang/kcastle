from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from kagent import Agent
from kagent.otel import OTelHooks
from kai import Context
from kai.providers import ProviderBase
from kai.types.stream import StreamEvent

from kcastle.castle import Castle
from kcastle.config import CastleConfig, ChannelConfig
from kcastle.providers import ModelConfig, ProviderConfig, ProviderEntry
from kcastle.session.manager import SessionManager


class DummyProvider(ProviderBase):
    def __init__(self, *, provider: str, model: str) -> None:
        self._provider = provider
        self._model = model

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    async def stream(self, context: Context, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        if False:
            yield cast(StreamEvent, None)


def _build_config(tmp_path: Path) -> CastleConfig:
    provider = ProviderEntry(
        config=ProviderConfig(
            provider="mock",
            model="",
            api_key="test-key",
        ),
        models=[
            ModelConfig(id="model-a", active=True),
            ModelConfig(id="model-b", active=True),
        ],
    )
    return CastleConfig(
        home=tmp_path,
        sessions_dir=tmp_path / "sessions",
        skills_dir=tmp_path / "skills",
        providers={"mock": provider},
        default_provider="mock",
        default_model="model-a",
        cli=ChannelConfig(enabled=False),
        telegram=ChannelConfig(enabled=False),
    )


def _make_castle(tmp_path: Path) -> Castle:
    config = _build_config(tmp_path)

    default_provider = DummyProvider(provider="mock", model="model-a")

    def _agent_factory(trace: Any) -> Agent:
        return Agent(llm=default_provider, trace=trace)

    session_manager = SessionManager(
        sessions_dir=config.sessions_dir,
        agent_factory=_agent_factory,
    )

    from kcastle.providers import ModelManager

    model_manager = ModelManager(
        config=config,
        session_manager=session_manager,
    )

    return Castle(
        config=config,
        session_manager=session_manager,
        skill_manager=object(),  # type: ignore[arg-type]
        channels=[],
        model_manager=model_manager,
        system_prompt="",
        skill_tools=[],
    )


def test_switch_model_requires_session_id(tmp_path: Path) -> None:
    castle = _make_castle(tmp_path)
    with pytest.raises(TypeError):
        castle.switch_model("mock", "model-b")  # type: ignore[call-arg]


def test_switch_model_only_affects_target_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    castle = _make_castle(tmp_path)
    sm = castle.session_manager

    s1 = sm.create(session_id="s1")
    s2 = sm.create(session_id="s2")

    import kcastle.providers.model_manager as model_manager_module

    def _fake_create_provider(config: ProviderConfig) -> object:
        return DummyProvider(provider=config.provider, model=config.model)

    monkeypatch.setattr(model_manager_module, "create_provider", _fake_create_provider)

    castle.switch_model("mock", "model-b", session_id="s1")

    assert s1.agent.llm.model == "model-b"
    assert s2.agent.llm.model == "model-a"
    assert castle.get_active_model("s1") == ("mock", "model-b")
    assert castle.get_active_model("s2") == ("mock", "model-a")


def test_switch_model_raises_for_unloaded_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    castle = _make_castle(tmp_path)

    import kcastle.providers.model_manager as model_manager_module

    def _fake_create_provider(config: ProviderConfig) -> object:
        return DummyProvider(provider=config.provider, model=config.model)

    monkeypatch.setattr(model_manager_module, "create_provider", _fake_create_provider)

    with pytest.raises(KeyError, match="not loaded"):
        castle.switch_model("mock", "model-b", session_id="missing")


def test_switch_model_persists_across_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    castle = _make_castle(tmp_path)

    import kcastle.providers.model_manager as model_manager_module

    def _fake_create_provider(config: ProviderConfig) -> object:
        return DummyProvider(provider=config.provider, model=config.model)

    monkeypatch.setattr(model_manager_module, "create_provider", _fake_create_provider)

    castle.session_manager.create(session_id="s1")
    castle.switch_model("mock", "model-b", session_id="s1")
    castle.session_manager.suspend("s1")

    castle2 = _make_castle(tmp_path)
    s1_resumed = castle2.session_manager.get_or_create("s1")

    assert castle2.get_active_model("s1") == ("mock", "model-b")
    assert s1_resumed.agent.llm.model == "model-b"


def test_build_agent_hooks_returns_none_when_no_endpoint(tmp_path: Path) -> None:
    config = _build_config(tmp_path)

    assert Castle._build_agent_hooks(config) is None  # pyright: ignore[reportPrivateUsage]


def test_build_agent_hooks_creates_otel_hooks_when_endpoint_set(
    tmp_path: Path,
) -> None:
    config = replace(_build_config(tmp_path), otel_endpoint="http://localhost:4317")

    hooks = Castle._build_agent_hooks(config)  # pyright: ignore[reportPrivateUsage]

    assert isinstance(hooks, OTelHooks)


def test_configure_otel_returns_none_when_no_endpoint(tmp_path: Path) -> None:
    config = _build_config(tmp_path)

    assert Castle._configure_otel(config) == (None, None)  # pyright: ignore[reportPrivateUsage]


def test_configure_otel_sets_up_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        _build_config(tmp_path),
        otel_endpoint="http://otel.example:4317",
    )
    captured: dict[str, object] = {}

    class FakeExporter:
        def __init__(self) -> None:
            captured["exporter"] = self

    class FakeBatchSpanProcessor:
        def __init__(self, exporter: object) -> None:
            captured["processor"] = self

    class FakeProvider:
        def __init__(self, *, resource: object) -> None:
            captured["resource"] = resource
            captured["provider"] = self

        def add_span_processor(self, processor: object) -> None:
            pass

        def shutdown(self) -> None:
            captured["shutdown"] = True

    monkeypatch.setattr("kcastle.otel._create_span_exporter", FakeExporter)
    monkeypatch.setattr("kcastle.otel._create_log_exporter", FakeExporter)
    monkeypatch.setattr("kcastle.otel.BatchSpanProcessor", FakeBatchSpanProcessor)
    monkeypatch.setattr("kcastle.otel.TracerProvider", FakeProvider)

    def _fake_resource_create(attrs: dict[str, str]) -> dict[str, str]:
        return attrs

    monkeypatch.setattr(
        "kcastle.otel.Resource.create",
        staticmethod(_fake_resource_create),
    )

    def _fake_set_provider(provider: object) -> object:
        return captured.setdefault("set_provider", provider)

    monkeypatch.setattr(
        "kcastle.otel.opentelemetry.trace.set_tracer_provider",
        _fake_set_provider,
    )
    # Block logger provider imports so configure_otel falls through to except ImportError.
    import sys

    monkeypatch.setitem(sys.modules, "opentelemetry._logs", None)

    provider, log_provider = Castle._configure_otel(config)  # pyright: ignore[reportPrivateUsage]

    assert provider is captured["provider"]
    assert log_provider is None
    assert captured["resource"] == {"service.name": "kcastle"}
