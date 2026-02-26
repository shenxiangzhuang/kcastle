"""Skill lifecycle management — discover, search, create, update.

``SkillManager`` is the main entry point for all skill operations in kcastle.
It handles layered discovery (builtin → user → project), override resolution,
and CRUD operations for skill directories.
"""

from __future__ import annotations

import logging
from pathlib import Path

from kai import Tool

from kcastle.skills.loader import LoadedSkill, SkillLoader
from kcastle.skills.resolver import SkillMatch, SkillResolver
from kcastle.skills.schema import SkillMeta, load_skill_meta, write_skill_yaml

_log = logging.getLogger("kcastle.skills")


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


class SkillManager:
    """Manages the full skill lifecycle: discover, search, load, create, update.

    Skills are discovered from three layers (lowest → highest priority):
    1. Builtin skills (shipped with kcastle)
    2. User skills (``~/.kcastle/skills``)
    3. Project skills (``<project_root>/.skills``)

    Same-id skills in higher-priority layers override lower ones.
    """

    def __init__(
        self,
        *,
        user_skills_dir: Path,
        project_skills_dir: Path | None = None,
        builtin_skills_dir: Path | None = None,
    ) -> None:
        self._user_dir = user_skills_dir
        self._project_dir = project_skills_dir
        self._builtin_dir = builtin_skills_dir
        self._loader = SkillLoader()
        self._resolver = SkillResolver()
        self._skills: dict[str, SkillMeta] = {}

    # --- Discovery ---

    def discover(self) -> list[SkillMeta]:
        """Scan all skill layers and build the merged skill index.

        Override order: project > user > builtin (same id).
        Returns the final merged list of discovered skills.
        """
        merged: dict[str, SkillMeta] = {}

        # Layer 1: builtin (lowest priority)
        if self._builtin_dir and self._builtin_dir.is_dir():
            for meta in self._scan_dir(self._builtin_dir, "builtin"):
                merged[meta.id] = meta

        # Layer 2: user
        if self._user_dir.is_dir():
            for meta in self._scan_dir(self._user_dir, "user"):
                merged[meta.id] = meta

        # Layer 3: project (highest priority)
        if self._project_dir and self._project_dir.is_dir():
            for meta in self._scan_dir(self._project_dir, "project"):
                merged[meta.id] = meta

        self._skills = merged
        skills_list = list(merged.values())
        self._resolver.index(skills_list)
        _log.info("Discovered %d skills", len(skills_list))
        return skills_list

    # --- Search ---

    def search(self, query: str) -> list[SkillMatch]:
        """Search for skills matching the query."""
        return self._resolver.search(query)

    def all_skills(self) -> list[SkillMeta]:
        """Return all discovered skills."""
        return list(self._skills.values())

    def get_skill(self, skill_id: str) -> SkillMeta | None:
        """Get a single skill by ID."""
        return self._skills.get(skill_id)

    # --- Loading ---

    def load_skill(self, skill_id: str) -> LoadedSkill | None:
        """Load a skill (tools + prompt) by its ID."""
        meta = self._skills.get(skill_id)
        if meta is None:
            return None
        return self._loader.load(meta)

    def load_skills(self, skill_ids: list[str]) -> list[LoadedSkill]:
        """Load multiple skills by their IDs."""
        results: list[LoadedSkill] = []
        for sid in skill_ids:
            loaded = self.load_skill(sid)
            if loaded is not None:
                results.append(loaded)
        return results

    def collect_tools(self, loaded: list[LoadedSkill]) -> list[Tool]:
        """Flatten all tools from loaded skills."""
        return [tool for skill in loaded for tool in skill.tools]

    def collect_prompts(self, loaded: list[LoadedSkill]) -> str:
        """Concatenate prompt fragments from loaded skills."""
        fragments = [s.prompt_fragment for s in loaded if s.prompt_fragment]
        return "\n\n".join(fragments)

    # --- Create / Update ---

    def create_skill(
        self,
        skill_id: str,
        *,
        name: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
        prompt_fragment: str = "",
        target: str = "user",
    ) -> SkillMeta:
        """Create a new skill directory in the target layer.

        Args:
            skill_id: Unique skill identifier (becomes directory name).
            name: Human-readable name (defaults to skill_id).
            description: Short description.
            tags: Searchable tags.
            prompt_fragment: System prompt guidance.
            target: Layer to create in (``user`` or ``project``).

        Returns:
            The created ``SkillMeta``.

        Raises:
            ValueError: If the target layer directory is not configured or
                a skill with this ID already exists in the target.
        """
        target_dir = self._resolve_target_dir(target)
        skill_dir = target_dir / skill_id
        if skill_dir.exists():
            raise ValueError(f"Skill '{skill_id}' already exists at {skill_dir}")

        meta = SkillMeta(
            id=skill_id,
            name=name or skill_id,
            description=description,
            tags=tags or [],
            prompt_fragment=prompt_fragment,
            source=target,
            path=skill_dir.resolve(),
        )
        write_skill_yaml(skill_dir, meta)

        # Create empty tools.py
        tools_path = skill_dir / meta.entry
        if not tools_path.exists():
            tools_path.write_text(f'"""Tools for skill: {skill_id}."""\n', encoding="utf-8")

        # Re-discover to update index
        self.discover()
        _log.info("Created skill %s in %s layer", skill_id, target)
        return meta

    def update_skill(
        self,
        skill_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        prompt_fragment: str | None = None,
    ) -> SkillMeta:
        """Update an existing skill's metadata.

        Only non-None fields are updated.

        Raises:
            KeyError: If the skill does not exist.
            PermissionError: If the skill is builtin (read-only).
        """
        meta = self._skills.get(skill_id)
        if meta is None:
            raise KeyError(f"Skill '{skill_id}' not found")
        if meta.source == "builtin":
            raise PermissionError(f"Cannot update builtin skill '{skill_id}'")

        # Build updated meta
        from dataclasses import replace

        updates: dict[str, object] = {}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description
        if tags is not None:
            updates["tags"] = tags
        if prompt_fragment is not None:
            updates["prompt_fragment"] = prompt_fragment

        updated = replace(meta, **updates)
        write_skill_yaml(meta.path, updated)

        # Re-discover
        self.discover()
        _log.info("Updated skill %s", skill_id)
        return updated

    # --- Internal ---

    def _resolve_target_dir(self, target: str) -> Path:
        """Resolve the directory for a given target layer."""
        if target == "user":
            return self._user_dir
        if target == "project":
            if self._project_dir is None:
                raise ValueError("No project skills directory configured")
            return self._project_dir
        raise ValueError(f"Invalid target layer: {target!r} (expected 'user' or 'project')")

    @staticmethod
    def _scan_dir(directory: Path, source: str) -> list[SkillMeta]:
        """Scan a directory for skill sub-directories."""
        results: list[SkillMeta] = []
        if not directory.is_dir():
            return results
        for child in sorted(directory.iterdir()):
            if not child.is_dir():
                continue
            meta = load_skill_meta(child, source=source)
            if meta is not None:
                results.append(meta)
            else:
                _log.debug("Skipping %s — not a valid skill directory", child)
        return results
