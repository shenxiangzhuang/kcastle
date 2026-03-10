"""ModelManager — per-session LLM provider and model management.

Owns all mutable model-selection state so that ``Castle`` can stay focused
on session and channel lifecycle.

Responsibilities
----------------
- Expose the default ``(provider_name, model_id)`` from configuration.
- Track per-session model overrides (in-memory cache + persisted via
  :attr:`~kcastle.session.session.Session.model_override`).
- Build :class:`~kai.ProviderBase` instances from configuration.
- Switch the model of a loaded session at runtime.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kai import ProviderBase

from kcastle.providers.factory import create_provider

if TYPE_CHECKING:
    from kcastle.config import CastleConfig
    from kcastle.session.manager import SessionManager

logger = logging.getLogger("kcastle.model_manager")


class ModelManager:
    """Manages LLM provider selection and per-session model overrides.

    Usage::

        model_manager = ModelManager(config=config, session_manager=sm)

        # Query active model (with optional session override)
        provider_name, model_id = model_manager.get_active_model(session_id)

        # Switch model for a specific session
        model_manager.switch_model("openai", "gpt-4o", session_id="s1")

        # Enumerate all usable models
        for provider_name, model_id in model_manager.available_models():
            print(provider_name, model_id)
    """

    def __init__(
        self,
        *,
        config: CastleConfig,
        session_manager: SessionManager,
    ) -> None:
        self._config = config
        self._session_manager = session_manager
        self._active_provider_name = config.default_provider
        self._active_model = config.default_model
        self._session_models: dict[str, tuple[str, str]] = {}

    @property
    def active_provider_name(self) -> str:
        """Name of the globally active provider."""
        return self._active_provider_name

    @property
    def active_model(self) -> str:
        """ID of the globally active model."""
        return self._active_model

    def get_active_model(self, session_id: str | None = None) -> tuple[str, str]:
        """Return the active ``(provider_name, model_id)`` for *session_id*.

        If *session_id* has an override (from a previous call to
        :meth:`switch_model` or a persisted
        :attr:`~kcastle.session.session.Session.model_override`), that
        override is returned.  Otherwise the global default is used.
        """
        if session_id is not None:
            if session_id in self._session_models:
                return self._session_models[session_id]

            loaded = self._session_manager.get(session_id)
            if loaded is not None and loaded.model_override is not None:
                override = loaded.model_override
                self._session_models[session_id] = override
                try:
                    loaded.agent.replace_llm(self.build_provider(*override))
                except (ValueError, RuntimeError):
                    logger.warning(
                        "Failed to restore session %s model override %s / %s",
                        session_id,
                        override[0],
                        override[1],
                    )
                return override

        return (self._active_provider_name, self._active_model)

    def available_models(self) -> list[tuple[str, str]]:
        """Return ``(provider_name, model_id)`` pairs for all active models.

        Only includes providers whose API key is non-empty after environment
        variable expansion.
        """
        result: list[tuple[str, str]] = []
        for pname, pcfg in self._config.providers.items():
            if not pcfg.api_key:
                continue
            for m in pcfg.active_models():
                result.append((pname, m.id))
        return result

    def build_provider(self, provider_name: str, model_id: str) -> ProviderBase:
        """Validate configuration and build a provider instance.

        Raises:
            ValueError: If *provider_name* or *model_id* is not in config.
        """
        provider_config = self._config.provider_config(provider_name, model_id)
        return create_provider(provider_config)

    def switch_model(
        self,
        provider_name: str,
        model_id: str,
        *,
        session_id: str,
    ) -> None:
        """Switch the active provider and model for a single loaded session.

        Hot-swaps the provider on the loaded session and persists the
        override so it survives session suspension and resumption.

        Raises:
            KeyError: If *session_id* is not currently loaded.
            ValueError: If *provider_name/model_id* is not in configuration.
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

        provider = self.build_provider(provider_name, model_id)
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

    def _apply_provider_to_session(self, session_id: str, provider: ProviderBase) -> None:
        """Hot-swap *provider* on a single loaded session."""
        session = self._session_manager.get(session_id)
        if session is None:
            raise KeyError(f"Session {session_id!r} is not loaded")
        session.agent.replace_llm(provider)
