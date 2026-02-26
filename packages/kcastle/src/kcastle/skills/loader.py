"""Skill loading — converts a skill directory into tools + prompt fragments.

``SkillLoader`` reads a skill's entry module (``tools.py``) and extracts
``kai.Tool`` subclasses from it via dynamic import.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from kai import Tool

from kcastle.skills.schema import SkillMeta

_log = logging.getLogger("kcastle.skills")


# ---------------------------------------------------------------------------
# LoadedSkill
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadedSkill:
    """A skill loaded into memory — tools extracted, prompt ready."""

    meta: SkillMeta
    tools: list[Tool] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    prompt_fragment: str = ""


# ---------------------------------------------------------------------------
# SkillLoader
# ---------------------------------------------------------------------------


class SkillLoader:
    """Loads skills from their directories into ``LoadedSkill`` instances."""

    def load(self, meta: SkillMeta) -> LoadedSkill:
        """Load a single skill by its metadata.

        Extracts ``Tool`` subclass instances from the entry module.
        """
        tools = self._load_tools(meta)
        return LoadedSkill(
            meta=meta,
            tools=tools,
            prompt_fragment=meta.prompt_fragment,
        )

    def _load_tools(self, meta: SkillMeta) -> list[Tool]:
        """Dynamically import tools from the skill's entry module."""
        entry_path = meta.path / meta.entry
        if not entry_path.is_file():
            _log.debug("No entry module %s for skill %s", entry_path, meta.id)
            return []

        try:
            module = self._import_module(meta.id, entry_path)
        except Exception:
            _log.exception("Failed to import tools from skill %s", meta.id)
            return []

        tools: list[Tool] = []
        for attr_name in dir(module):
            attr: object = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, Tool) and attr is not Tool:
                try:
                    instance: Tool = attr()  # type: ignore[call-arg]
                    tools.append(instance)
                except Exception:
                    _log.exception("Failed to instantiate tool %s in skill %s", attr_name, meta.id)

        _log.debug("Loaded %d tools from skill %s", len(tools), meta.id)
        return tools

    @staticmethod
    def _import_module(skill_id: str, path: Path) -> object:
        """Import a Python module from an arbitrary path."""
        module_name = f"kcastle._skills_runtime.{skill_id}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
