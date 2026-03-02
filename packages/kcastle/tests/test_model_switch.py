from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest
from kagent import Agent
from kai import Context
from kai.chunk import Chunk

from kcastle.castle import Castle
from kcastle.config import CastleConfig, ChannelConfig, ModelConfig, ProviderConfig
from kcastle.session.manager import SessionManager


class DummyProvider:
    def __init__(self, *, name: str, model: str) -> None:
        self._name = name
        self._model = model

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    async def stream_raw(self, context: Context, **kwargs: Any) -> AsyncIterator[Chunk]:
        if False:
            yield cast(Chunk, None)


def _build_config(tmp_path: Path) -> CastleConfig:
    provider = ProviderConfig(
        name="mock",
        protocol="openai-completions",
        api_key="test-key",
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

    default_provider = DummyProvider(name="mock", model="model-a")

    def _agent_factory(trace: Any) -> Agent:
        return Agent(provider=default_provider, trace=trace)

    session_manager = SessionManager(
        sessions_dir=config.sessions_dir,
        agent_factory=_agent_factory,
    )

    return Castle(
        config=config,
        session_manager=session_manager,
        skill_manager=object(),  # type: ignore[arg-type]
        channels=[],
        provider=default_provider,
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

    import kcastle.castle as castle_module

    def _fake_create_provider(config: CastleConfig) -> object:
        return DummyProvider(name=config.default_provider, model=config.default_model)

    monkeypatch.setattr(castle_module, "_create_provider", _fake_create_provider)

    castle.switch_model("mock", "model-b", session_id="s1")

    assert s1.agent.provider.model == "model-b"
    assert s2.agent.provider.model == "model-a"
    assert castle.get_active_model("s1") == ("mock", "model-b")
    assert castle.get_active_model("s2") == ("mock", "model-a")


def test_switch_model_raises_for_unloaded_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    castle = _make_castle(tmp_path)

    import kcastle.castle as castle_module

    def _fake_create_provider(config: CastleConfig) -> object:
        return DummyProvider(name=config.default_provider, model=config.default_model)

    monkeypatch.setattr(castle_module, "_create_provider", _fake_create_provider)

    with pytest.raises(KeyError, match="not loaded"):
        castle.switch_model("mock", "model-b", session_id="missing")


def test_switch_model_persists_across_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    castle = _make_castle(tmp_path)

    import kcastle.castle as castle_module

    def _fake_create_provider(config: CastleConfig) -> object:
        return DummyProvider(name=config.default_provider, model=config.default_model)

    monkeypatch.setattr(castle_module, "_create_provider", _fake_create_provider)

    castle.session_manager.create(session_id="s1")
    castle.switch_model("mock", "model-b", session_id="s1")
    castle.session_manager.suspend("s1")

    castle2 = _make_castle(tmp_path)
    s1_resumed = castle2.session_manager.get_or_create("s1")

    assert castle2.get_active_model("s1") == ("mock", "model-b")
    assert s1_resumed.agent.provider.model == "model-b"
