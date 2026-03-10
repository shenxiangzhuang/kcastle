"""Castle — top-level lifecycle orchestrator.

``Castle`` wires together configuration, session management, skill management,
and channels.  It is the single entry point for running kcastle.
"""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path

from kagent import Agent, Trace
from kai import Tool

from kcastle.channels import Channel
from kcastle.channels.cli import CLIChannel
from kcastle.channels.telegram import TelegramChannel
from kcastle.config import CastleConfig, load_config
from kcastle.log import logger
from kcastle.providers import ModelManager, create_provider
from kcastle.session.manager import SessionManager
from kcastle.skills.manager import SkillManager, find_project_root
from kcastle.skills.skill import render_compact_skills
from kcastle.tools import create_builtin_tools


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
        model_manager: ModelManager,
        system_prompt: str,
        skill_tools: list[Tool],
    ) -> None:
        self._config = config
        self._session_manager = session_manager
        self._skill_manager = skill_manager
        self._channels: list[Channel] = channels
        self._model_manager = model_manager
        self._system_prompt = system_prompt
        self._skill_tools = skill_tools

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
    def model_manager(self) -> ModelManager:
        return self._model_manager

    @property
    def active_provider_name(self) -> str:
        return self._model_manager.active_provider_name

    @property
    def active_model(self) -> str:
        return self._model_manager.active_model

    def get_active_model(self, session_id: str | None = None) -> tuple[str, str]:
        """Return active ``(provider_name, model_id)``.

        Delegates to :class:`~kcastle.providers.ModelManager`.
        """
        return self._model_manager.get_active_model(session_id)

    def available_models(self) -> list[tuple[str, str]]:
        """Return ``(provider_name, model_id)`` pairs for all active models.

        Delegates to :class:`~kcastle.providers.ModelManager`.
        """
        return self._model_manager.available_models()

    def switch_model(
        self,
        provider_name: str,
        model_id: str,
        *,
        session_id: str,
    ) -> None:
        """Switch the active provider and model for a single loaded session.

        Delegates to :class:`~kcastle.providers.ModelManager`.
        """
        self._model_manager.switch_model(provider_name, model_id, session_id=session_id)

    def prepare_user_input(self, user_input: str) -> str:
        """Augment user input with explicitly hinted skill instructions.

        Delegates to :meth:`~kcastle.skills.SkillManager.expand_hints`.
        """
        return self._skill_manager.expand_hints(user_input)

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

        provider_config = config.active_provider_config()
        provider = create_provider(provider_config)

        all_skills = skill_manager.all_skills()
        skill_tools = create_builtin_tools(
            workspace=Path.cwd(),
            skill_manager=skill_manager,
        )
        skill_prompts = render_compact_skills(all_skills)

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

        session_manager = SessionManager(
            sessions_dir=config.sessions_dir,
            agent_factory=agent_factory,
        )

        model_manager = ModelManager(
            config=config,
            session_manager=session_manager,
        )

        return cls(
            config=config,
            session_manager=session_manager,
            skill_manager=skill_manager,
            channels=channels,
            model_manager=model_manager,
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
