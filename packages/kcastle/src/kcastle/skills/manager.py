"""Skill lifecycle management — discover and search.

``SkillManager`` is the main entry point for all skill operations in kcastle.
It handles layered discovery (builtin → user → project), override resolution,
and keyword search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from kcastle.log import logger
from kcastle.skills.skill import Skill

_WORD_RE = re.compile(r"[a-z0-9]+")


def find_project_root(cwd: Path) -> Path:
    """Find the project root by walking up from *cwd*.

    Checks for ``.git/`` first, then ``pyproject.toml``, else returns *cwd*.
    """
    current = cwd.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").is_dir():
            return parent
        if (parent / "pyproject.toml").is_file():
            return parent
    return current


@dataclass(frozen=True, slots=True)
class SkillMatch:
    """A skill matched against a query with a relevance score."""

    skill: Skill
    score: float


class SkillManager:
    """Manages the skill lifecycle: discover and search.

    Skills are discovered from three layers (lowest → highest priority):
    1. Builtin skills (shipped with kcastle)
    2. User skills (``~/.agent/skills``)
    3. Project skills (``<project_root>/.agent/skills``)

    Same-name skills in higher-priority layers override lower ones.
    """

    def __init__(
        self,
        *,
        user_skills_dir: Path,
        project_skills_dir: Path | None = None,
        builtin_skills_dir: Path | None = None,
        top_k: int = 3,
    ) -> None:
        self._user_dir = user_skills_dir
        self._project_dir = project_skills_dir
        self._builtin_dir = builtin_skills_dir
        self._top_k = top_k
        self._skills: dict[str, Skill] = {}

    def discover(self) -> list[Skill]:
        """Scan all skill layers and build the merged skill index.

        Override order: project > user > builtin (same name).
        Returns the final merged list of discovered skills.
        """
        merged: dict[str, Skill] = {}

        if self._builtin_dir and self._builtin_dir.is_dir():
            for skill in self._scan_dir(self._builtin_dir, "builtin"):
                merged[skill.name] = skill

        if self._user_dir.is_dir():
            for skill in self._scan_dir(self._user_dir, "user"):
                merged[skill.name] = skill

        if self._project_dir and self._project_dir.is_dir():
            for skill in self._scan_dir(self._project_dir, "project"):
                merged[skill.name] = skill

        self._skills = merged
        logger.info("Discovered %d skills", len(merged))
        return list(merged.values())

    def search(self, query: str) -> list[SkillMatch]:
        """Search for skills matching the query.  Returns top-K results."""
        if not query.strip():
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: list[SkillMatch] = []
        for skill in self._skills.values():
            score = _score(skill, query_tokens)
            if score > 0:
                scored.append(SkillMatch(skill=skill, score=score))

        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[: self._top_k]

    def all_skills(self) -> list[Skill]:
        """Return all discovered skills."""
        return list(self._skills.values())

    def get_skill(self, name: str) -> Skill | None:
        """Get a single skill by name."""
        return self._skills.get(name)

    @staticmethod
    def _scan_dir(directory: Path, source: str) -> list[Skill]:
        """Scan a directory for skill sub-directories."""
        results: list[Skill] = []
        if not directory.is_dir():
            return results
        for child in sorted(directory.iterdir()):
            if not child.is_dir():
                continue
            skill = Skill.load(child, source=source)
            if skill is not None:
                results.append(skill)
            else:
                logger.debug("Skipping %s — not a valid skill directory", child)
        return results


def _tokenize(text: str) -> set[str]:
    normalized = text.lower().replace("-", " ").replace("_", " ").replace(".", " ")
    return set(_WORD_RE.findall(normalized))


def _score(skill: Skill, query_tokens: set[str]) -> float:
    """Score a skill against query tokens (keyword overlap)."""
    searchable = " ".join(
        [
            skill.name.lower(),
            skill.description.lower(),
            " ".join(t.lower() for t in skill.tags),
        ]
    )
    searchable_tokens = _tokenize(searchable)
    overlap = query_tokens & searchable_tokens
    if not overlap:
        return 0.0
    return len(overlap) / len(query_tokens)
