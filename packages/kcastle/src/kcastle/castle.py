"""Castle — top-level lifecycle orchestrator.

``Castle`` wires together configuration, session management, skill management,
and channels.  It is the single entry point for running kcastle.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path
from typing import Any

from kagent import Agent, Trace
from kai import Tool

from kcastle.channels import Channel
from kcastle.channels.cli import CLIChannel
from kcastle.channels.telegram import TelegramChannel
from kcastle.config import CastleConfig, load_config
from kcastle.session.manager import SessionManager
from kcastle.skills.manager import SkillManager, find_project_root

_log = logging.getLogger("kcastle")

# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def _create_provider(config: CastleConfig) -> object:
    """Create a kai Provider from the active provider config.

    Looks up ``config.active_provider()``, selects the kai driver class
    based on ``protocol``, and forwards any model-specific options.
    """
    from kai import Anthropic, OpenAICompletions, OpenAIResponses

    provider_cfg = config.active_provider()

    kwargs: dict[str, object] = {"model": config.default_model}
    if provider_cfg.api_key:
        kwargs["api_key"] = provider_cfg.api_key
    if provider_cfg.base_url:
        kwargs["base_url"] = provider_cfg.base_url

    # Merge model-specific options (max_tokens, reasoning, thinking, …)
    model_cfg = provider_cfg.get_model(config.default_model)
    if model_cfg:
        kwargs.update(model_cfg.options)

    protocol = provider_cfg.protocol.lower()
    if protocol in ("openai", "openai-completions"):
        return OpenAICompletions(**kwargs)  # type: ignore[arg-type]
    if protocol == "openai-responses":
        return OpenAIResponses(**kwargs)  # type: ignore[arg-type]
    if protocol == "anthropic":
        return Anthropic(**kwargs)  # type: ignore[arg-type]

    raise ValueError(f"Unknown protocol: {provider_cfg.protocol!r}")


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------


def _build_system_prompt(config: CastleConfig, skill_prompts: str = "") -> str:
    """Assemble the system prompt from composable blocks."""
    from kcastle.prompts import (
        assemble_system_prompt,
        build_runtime_context,
        load_identity_prompt,
        read_workspace_prompt,
    )

    workspace_prompt = read_workspace_prompt(Path.cwd())

    # If user set a custom system_prompt in config, use it as override;
    # otherwise fall back to the built-in identity.
    user_override = config.system_prompt or None

    return assemble_system_prompt(
        identity=load_identity_prompt(),
        runtime_context=build_runtime_context(),
        workspace_prompt=workspace_prompt,
        skill_prompts=skill_prompts or None,
        user_override=user_override,
    )


# ---------------------------------------------------------------------------
# Castle
# ---------------------------------------------------------------------------


class Castle:
    """Top-level orchestrator that wires everything together.

    Usage::

        castle = Castle.create()
        await castle.run()
    """

    def __init__(
        self,
        *,
        config: CastleConfig,
        session_manager: SessionManager,
        skill_manager: SkillManager,
        channels: list[Channel],
        provider: object,
        system_prompt: str,
        skill_tools: list[Tool],
    ) -> None:
        self._config = config
        self._session_manager = session_manager
        self._skill_manager = skill_manager
        self._channels: list[Channel] = channels
        self._shutdown_event = asyncio.Event()
        # Mutable runtime state — updated by switch_model()
        self._provider = provider
        self._system_prompt = system_prompt
        self._skill_tools = skill_tools
        self._active_provider_name = config.default_provider
        self._active_model = config.default_model

    # --- Properties ---

    @property
    def config(self) -> CastleConfig:
        return self._config

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager

    @property
    def skill_manager(self) -> SkillManager:
        return self._skill_manager

    @property
    def active_provider_name(self) -> str:
        return self._active_provider_name

    @property
    def active_model(self) -> str:
        return self._active_model

    # --- Agent factory ---

    def _make_agent_factory(self) -> Any:  # noqa: ANN401
        """Create an agent factory closure using current provider."""
        provider = self._provider
        system_prompt = self._system_prompt
        skill_tools = self._skill_tools
        max_turns = self._config.max_turns

        def factory(trace: Trace) -> Agent:
            return Agent(
                provider=provider,  # type: ignore[arg-type]
                system=system_prompt,
                tools=skill_tools if skill_tools else None,
                trace=trace,
                max_turns=max_turns,
            )

        return factory

    # --- Model switching ---

    def available_models(self) -> list[tuple[str, str]]:
        """Return ``(provider_name, model_id)`` pairs for all active models.

        Only includes providers whose API key is non-empty after env
        expansion.
        """
        result: list[tuple[str, str]] = []
        for pname, pcfg in self._config.providers.items():
            if not pcfg.api_key:
                continue
            for m in pcfg.active_models():
                result.append((pname, m.id))
        return result

    def switch_model(self, provider_name: str, model_id: str) -> None:
        """Switch the active provider and model at runtime.

        Swaps the provider on all live session agents and updates the
        factory so new sessions also use the new model.
        """
        from dataclasses import replace

        # Validate
        if provider_name not in self._config.providers:
            raise ValueError(f"Unknown provider: {provider_name!r}")
        pcfg = self._config.providers[provider_name]
        if pcfg.get_model(model_id) is None:
            raise ValueError(f"Unknown model: {model_id!r} in provider {provider_name!r}")

        # Create new provider
        new_config = replace(
            self._config,
            default_provider=provider_name,
            default_model=model_id,
        )
        self._provider = _create_provider(new_config)

        # Hot-swap provider on all live session agents
        for session in self._session_manager._sessions.values():  # pyright: ignore[reportPrivateUsage]
            session._agent._provider = self._provider  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]

        # Update factory for new sessions
        self._session_manager._agent_factory = self._make_agent_factory()  # pyright: ignore[reportPrivateUsage]

        self._active_provider_name = provider_name
        self._active_model = model_id
        _log.info("Switched to %s / %s", provider_name, model_id)

    # --- Factory ---

    @classmethod
    def create(
        cls,
        config: CastleConfig | None = None,
        *,
        session_id: str | None = None,
        continue_latest: bool = False,
        daemon: bool = False,
    ) -> Castle:
        """Create a Castle from configuration.

        Builds the provider, skill manager, session manager, and channels.
        """
        if config is None:
            config = load_config()

        # Ensure directories exist
        config.home.mkdir(parents=True, exist_ok=True)
        config.sessions_dir.mkdir(parents=True, exist_ok=True)
        config.skills_dir.mkdir(parents=True, exist_ok=True)

        # Skill manager
        project_root = find_project_root(Path.cwd())
        project_skills = project_root / ".skills"
        skill_manager = SkillManager(
            user_skills_dir=config.skills_dir,
            project_skills_dir=project_skills if project_skills.is_dir() else None,
        )
        skill_manager.discover()

        # Build provider, system prompt, skill tools
        provider = _create_provider(config)

        all_skills = skill_manager.all_skills()
        loaded_skills = skill_manager.load_skills([s.id for s in all_skills])
        skill_tools = skill_manager.collect_tools(loaded_skills)
        skill_prompts = skill_manager.collect_prompts(loaded_skills)

        system_prompt = _build_system_prompt(config, skill_prompts)

        # Channels — Telegram only runs in daemon mode to avoid
        # "terminated by other getUpdates request" conflicts.
        channels: list[Channel] = []
        if config.cli.enabled and not daemon:
            channels.append(
                CLIChannel(
                    session_id=session_id,
                    continue_latest=continue_latest,
                )
            )
        if config.telegram.enabled and config.telegram_token and daemon:
            bot_username = config.telegram.options.get("bot_username", "")
            channels.append(
                TelegramChannel(
                    token=config.telegram_token,
                    bot_username=str(bot_username),
                )
            )

        def _init_factory(trace: Trace) -> Agent:
            return Agent(
                provider=provider,  # type: ignore[arg-type]
                system=system_prompt,
                trace=trace,
                max_turns=config.max_turns,
            )

        castle = cls(
            config=config,
            session_manager=SessionManager(
                sessions_dir=config.sessions_dir,
                agent_factory=_init_factory,
            ),
            skill_manager=skill_manager,
            channels=channels,
            provider=provider,
            system_prompt=system_prompt,
            skill_tools=skill_tools,
        )
        castle._session_manager._agent_factory = castle._make_agent_factory()  # pyright: ignore[reportPrivateUsage]
        return castle

    # --- Lifecycle ---

    async def run(self) -> None:
        """Start all channels and wait until shutdown."""
        if not self._channels:
            _log.warning("No channels configured — nothing to do")
            return

        # Install signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._signal_handler)

        _log.info(
            "Castle starting with %d channel(s): %s",
            len(self._channels),
            ", ".join(c.name for c in self._channels),
        )

        # Start all channels concurrently
        tasks = [asyncio.create_task(ch.start(self)) for ch in self._channels]

        try:
            # Wait for either all channels to finish or shutdown signal
            _done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Stop all channels and suspend all sessions."""
        _log.info("Castle shutting down")

        # Stop channels
        for ch in self._channels:
            try:
                await ch.stop()
            except Exception:
                _log.exception("Error stopping channel %s", ch.name)

        # Suspend all sessions
        self._session_manager.suspend_all()

        _log.info("Castle shut down")

    def _signal_handler(self) -> None:
        """Handle SIGINT/SIGTERM."""
        _log.info("Received shutdown signal")
        self._shutdown_event.set()
        # Cancel all running tasks
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()
