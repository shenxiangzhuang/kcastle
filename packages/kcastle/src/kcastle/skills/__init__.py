"""kcastle.skills — Skill management for kcastle.

Discover, search, load, create, and update skills across three layers:
builtin, user (``~/.kcastle/skills``), and project (``<root>/.skills``).
"""

from kcastle.skills.loader import LoadedSkill, SkillLoader
from kcastle.skills.manager import SkillManager
from kcastle.skills.resolver import SkillMatch, SkillResolver
from kcastle.skills.schema import SkillMeta, load_skill_meta, write_skill_yaml

__all__ = [
    "LoadedSkill",
    "SkillLoader",
    "SkillManager",
    "SkillMatch",
    "SkillMeta",
    "SkillResolver",
    "load_skill_meta",
    "write_skill_yaml",
]
