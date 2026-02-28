"""Skill — metadata, loading, and prompt rendering.

Canonical format follows anthropics/skills exactly:

- ``<skill-dir>/SKILL.md`` (required)
- YAML frontmatter with ``name`` and ``description`` (required)
- Markdown body as instructions
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from kcastle.log import logger

_SKILL_MD = "SKILL.md"
_FM_SEP = "---"
_HINT_RE = re.compile(r"\$([A-Za-z0-9_.-]+)")


@dataclass(frozen=True, slots=True)
class Skill:
    """A single skill discovered from a ``SKILL.md`` file.

    ``name`` from frontmatter serves as the unique identifier.
    ``path`` is the skill directory; ``file_path`` is derived as ``path / SKILL.md``.
    """

    name: str
    """Skill name — unique identifier from frontmatter."""

    description: str
    """Short description of what this skill does."""

    path: Path
    """Absolute path to the skill directory."""

    tags: list[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    """Optional searchable tags."""

    instructions: str = ""
    """Instructions body loaded from ``SKILL.md``."""

    source: str = "unknown"
    """Source layer: ``builtin``, ``user``, or ``project``."""

    @property
    def file_path(self) -> Path:
        """Absolute path to ``SKILL.md`` (derived from ``path``)."""
        return self.path / _SKILL_MD

    @classmethod
    def load(cls, skill_dir: Path, *, source: str = "unknown") -> Skill | None:
        """Load from a skill directory containing ``SKILL.md``.

        Returns ``None`` if the directory lacks a valid ``SKILL.md``.
        """
        skill_md = skill_dir / _SKILL_MD
        if not skill_md.is_file():
            return None

        raw = skill_md.read_text(encoding="utf-8")
        frontmatter, body = _parse_frontmatter(raw)

        name = str(frontmatter.get("name", "")).strip()
        if not name:
            logger.warning("Skill %s missing name in SKILL.md frontmatter", skill_dir)
            return None

        description = str(frontmatter.get("description", "")).strip()
        if not description:
            logger.warning("Skill %s missing description in SKILL.md frontmatter", skill_dir)
            return None

        tags_raw: object = frontmatter.get("tags", [])
        tags = [str(t) for t in cast(list[object], tags_raw)] if isinstance(tags_raw, list) else []

        return cls(
            name=name,
            description=description,
            path=skill_dir.resolve(),
            tags=tags,
            instructions=body,
            source=source,
        )

    def save(self, skill_dir: Path | None = None) -> None:
        """Write anthropics-style ``SKILL.md``.

        Defaults to ``self.path`` if *skill_dir* is not provided.
        """
        target = skill_dir or self.path
        target.mkdir(parents=True, exist_ok=True)

        fm_data: dict[str, object] = {"name": self.name, "description": self.description}
        if self.tags:
            fm_data["tags"] = list(self.tags)

        fm = yaml.safe_dump(fm_data, sort_keys=False, allow_unicode=True).strip()
        body = self.instructions.strip()
        content = f"---\n{fm}\n---\n\n{body}\n" if body else f"---\n{fm}\n---\n\n# {self.name}\n"
        (target / _SKILL_MD).write_text(content, encoding="utf-8")

    @staticmethod
    def render_compact(skills: list[Skill]) -> str:
        """Render compact skill metadata for system prompt injection."""
        if not skills:
            return ""

        lines = ["<skills>"]
        for skill in skills:
            lines.append(f"- {skill.name} ({skill.source}): {skill.description}")
        lines.append("</skills>")
        return "\n".join(lines)

    @staticmethod
    def render_expanded(skills: list[Skill]) -> str:
        """Render full instruction bodies for explicitly hinted skills."""
        if not skills:
            return ""

        lines = ["<skill_expansion>"]
        for skill in skills:
            lines.append(f"=== [{skill.name}] ({skill.source}) ===")
            if skill.instructions.strip():
                lines.append(skill.instructions.strip())
            lines.append("")
        lines.append("</skill_expansion>")
        return "\n".join(lines).strip()

    @staticmethod
    def extract_hints(text: str) -> list[str]:
        """Extract unique ``$skill`` style hints from free text."""
        hints: list[str] = []
        seen: set[str] = set()

        for match in _HINT_RE.finditer(text):
            raw = match.group(1).strip().lower()
            if not raw:
                continue
            normalized = raw.replace("_", "-")
            if normalized in seen:
                continue
            seen.add(normalized)
            hints.append(normalized)

        return hints


def _parse_frontmatter(markdown: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter and markdown body from ``SKILL.md``."""
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != _FM_SEP:
        return {}, markdown.strip()

    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == _FM_SEP:
            payload = "\n".join(lines[1:idx])
            meta = _load_yaml(payload)
            body = "\n".join(lines[idx + 1 :]).strip()
            return meta, body
    return {}, markdown.strip()


def _load_yaml(text: str) -> dict[str, Any]:
    """Parse YAML text safely."""
    try:
        data: object = yaml.safe_load(text)
    except yaml.YAMLError:
        logger.warning("Invalid YAML frontmatter in SKILL.md")
        return {}
    return dict(data) if isinstance(data, dict) else {}  # type: ignore[arg-type]
