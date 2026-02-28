"""kcastle.skills — Skill management for kcastle.

Discover, search, and render skills across three layers:
builtin (``kcastle/skills``), user (``~/.kcastle/skills``), and
project (``<root>/.skills``).
"""

from kcastle.skills.manager import SkillManager, SkillMatch, find_project_root
from kcastle.skills.skill import Skill

__all__ = [
    "Skill",
    "SkillManager",
    "SkillMatch",
    "find_project_root",
]
