"""Built-in runtime tools for kcastle."""

from __future__ import annotations

from pathlib import Path

from kai import Tool

from kcastle.skills import SkillManager
from kcastle.tools.core import create_core_tools
from kcastle.tools.emoji_reactor import EmojiReactor, emoji_reactor
from kcastle.tools.skills import create_skill_tools


def create_builtin_tools(*, workspace: Path, skill_manager: SkillManager) -> list[Tool]:
    """Create all built-in runtime tools.

    Includes:
    - Core coding tools (file/search/bash)
    - Skill discovery tools (skills_list)
    """
    tools: list[Tool] = []
    tools.extend(create_core_tools(workspace=workspace))
    tools.extend(create_skill_tools(manager=skill_manager))
    return tools


__all__ = ["create_builtin_tools", "EmojiReactor", "emoji_reactor"]
