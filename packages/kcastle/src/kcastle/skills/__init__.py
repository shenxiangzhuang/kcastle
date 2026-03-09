"""kcastle.skills — Skill management for kcastle.

Discover, search, and render skills across three layers:
builtin (``kcastle/skills``), user (``~/.agent/skills``), and
project (``<root>/.agent/skills``).
"""

from kcastle.skills.manager import SkillManager
from kcastle.skills.skill import Skill

__all__ = [
    "Skill",
    "SkillManager",
]
