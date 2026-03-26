"""kcastle.skills — Skill management for kcastle.

Discover, search, and render skills across three layers:
builtin (``kcastle/skills``), user (``~/.agents/skills``), and
project (``<root>/.agents/skills``).
"""

from kcastle.skills.manager import SkillManager
from kcastle.skills.skill import (
    Skill,
    extract_skill_hints,
    render_compact_skills,
    render_expanded_skills,
)

__all__ = [
    "Skill",
    "SkillManager",
    "extract_skill_hints",
    "render_compact_skills",
    "render_expanded_skills",
]
