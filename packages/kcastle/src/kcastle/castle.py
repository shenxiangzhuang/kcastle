"""Castle — top-level lifecycle orchestrator.

``Castle`` wires together configuration, session management, skill management,
and channels.  It is the single entry point for running kcastle.
"""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from typing import Any

from kagent import Agent, Trace
from kai import ProviderBase, Tool

from kcastle.channels import Channel
from kcastle.channels.cli import CLIChannel
from kcastle.channels.telegram import TelegramChannel
from kcastle.config import CastleConfig, load_config
from kcastle.log import logger
from kcastle.providers import create_provider
from kcastle.session.manager import SessionManager
from kcastle.skills.manager import SkillManager, find_project_root
from kcastle.skills.skill import Skill
from kcastle.tools import create_builtin_tools


def _create_provider(config: CastleConfig) -> ProviderBase:
    """Create a kai Provider from the active provider config."""
    provider_config = config.active_provider_config()
    return create_provider(provider_config)


def _build_system_prompt(config: CastleConfig, skill_prompts: str = "") -> str:
    """Assemble the system prompt from composable blocks."""
    from kcastle.prompts import (
        assemble_system_prompt,
        build_runtime_context,
        load_identity_prompt,
        read_workspace_prompt,
    )

    workspace_prompt = read_workspace_prompt(Path.cwd())

    user_override = config.system_prompt or None

    return assemble_system_prompt(
        identity=load_identity_prompt(),
        runtime_context=build_runtime_context(),
        workspace_prompt=workspace_prompt,
        skill_prompts=skill_prompts or None,
        user_override=user_override,
    )


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
        provider: ProviderBase,
        system_prompt: str,
        skill_tools: list[Tool],
    ) -> None:
        self._config = config
        self._session_manager = session_manager
        self._skill_manager = skill_manager
        self._channels: list[Channel] = channels
        self._provider = provider
        self._system_prompt = system_prompt
        self._skill_tools = skill_tools
        self._active_provider_name = config.default_provider
        self._active_model = config.default_model
        self._session_models: dict[str, tuple[str, str]] = {}

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

    def get_active_model(self, session_id: str | None = None) -> tuple[str, str]:
        """Return active ``(provider_name, model_id)``.

        If ``session_id`` has an override, returns that override; otherwise
        returns the global default runtime model.
        """
        if session_id is not None:
            if session_id in self._session_models:
                return self._session_models[session_id]

            loaded = self._session_manager.get(session_id)
            if loaded is not None and loaded.model_override is not None:
                override = loaded.model_override
                self._session_models[session_id] = override
                try:
                    loaded.agent.replace_llm(self._build_provider(*override))
                except (ValueError, RuntimeError):
                    logger.warning(
                        "Failed to restore session %s model override %s / %s",
                        session_id,
                        override[0],
                        override[1],
                    )
                return override

        return (self._active_provider_name, self._active_model)

    def prepare_user_input(self, user_input: str) -> str:
        """Augment user input with explicitly hinted skill instructions.

        Bub-style progressive disclosure:
        - compact skill metadata is always present in system prompt
        - full skill body is injected only when user references ``$skill-name``
        """
        hints = Skill.extract_hints(user_input)
        if not hints:
            return user_input

        expanded: list[Any] = []
        for hint in hints:
            skill = self._skill_manager.get_skill(hint)
            if skill is None:
                continue
            expanded.append(skill)

        expansion_block = Skill.render_expanded(expanded)
        if not expansion_block:
            return user_input
        return f"{user_input}\n\n{expansion_block}"

    def _build_provider(self, provider_name: str, model_id: str) -> ProviderBase:
        """Validate and build a provider instance for ``provider_name/model_id``."""
        provider_config = self._config.provider_config(provider_name, model_id)
        return create_provider(provider_config)

    def _apply_provider_to_session(self, session_id: str, provider: ProviderBase) -> None:
        """Hot-swap provider for one loaded session."""
        session = self._session_manager.get(session_id)
        if session is None:
            raise KeyError(f"Session {session_id!r} is not loaded")
        session.agent.replace_llm(provider)

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

    def switch_model(
        self,
        provider_name: str,
        model_id: str,
        *,
        session_id: str,
    ) -> None:
        """Switch the active provider and model at runtime.

        Only updates the specified loaded session.
        """
        current_provider, current_model = self.get_active_model(session_id)
        logger.info(
            "Switching session %s model: %s / %s -> %s / %s",
            session_id,
            current_provider,
            current_model,
            provider_name,
            model_id,
        )

        provider = self._build_provider(provider_name, model_id)
        self._apply_provider_to_session(session_id, provider)
        session = self._session_manager.get(session_id)
        if session is None:
            raise KeyError(f"Session {session_id!r} is not loaded")
        session.set_model_override(provider_name, model_id)
        self._session_models[session_id] = (provider_name, model_id)
        logger.info(
            "Switched session %s model to %s / %s",
            session_id,
            provider_name,
            model_id,
        )

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

        config.home.mkdir(parents=True, exist_ok=True)
        config.sessions_dir.mkdir(parents=True, exist_ok=True)
        config.skills_dir.mkdir(parents=True, exist_ok=True)

        project_root = find_project_root(Path.cwd())
        project_skills = project_root / ".agent" / "skills"
        skill_manager = SkillManager(
            user_skills_dir=config.skills_dir,
            project_skills_dir=project_skills,
            builtin_skills_dir=Path(__file__).resolve().parent / "skills",
        )
        skill_manager.discover()

        provider = _create_provider(config)

        all_skills = skill_manager.all_skills()
        skill_tools = create_builtin_tools(
            workspace=Path.cwd(),
            skill_manager=skill_manager,
        )
        skill_prompts = Skill.render_compact(all_skills)

        system_prompt = _build_system_prompt(config, skill_prompts)

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

        def agent_factory(trace: Trace) -> Agent:
            return Agent(
                llm=provider,
                system=system_prompt,
                tools=skill_tools if skill_tools else None,
                trace=trace,
                max_turns=config.max_turns,
            )

        return cls(
            config=config,
            session_manager=SessionManager(
                sessions_dir=config.sessions_dir,
                agent_factory=agent_factory,
            ),
            skill_manager=skill_manager,
            channels=channels,
            provider=provider,
            system_prompt=system_prompt,
            skill_tools=skill_tools,
        )

    async def run(self) -> None:
        """Start all channels and wait until shutdown."""
        if not self._channels:
            logger.warning("No channels configured — nothing to do")
            return

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._signal_handler)

        logger.info(
            "Castle starting with %d channel(s): %s",
            len(self._channels),
            ", ".join(c.name for c in self._channels),
        )

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
        logger.info("Castle shutting down")

        for ch in self._channels:
            try:
                await ch.stop()
            except (RuntimeError, OSError, ValueError, KeyError):
                logger.exception("Error stopping channel %s", ch.name)

        self._session_manager.suspend_all()

        logger.info("Castle shut down")

    def _signal_handler(self) -> None:
        """Handle SIGINT/SIGTERM."""
        logger.info("Received shutdown signal")
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()
