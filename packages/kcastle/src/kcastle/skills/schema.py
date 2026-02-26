"""Skill metadata schema and validation.

Defines ``SkillMeta`` — the typed representation of a ``skill.yaml`` file,
and helpers for reading/writing skill metadata from disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger("kcastle.skills")


# We use a pure-stdlib YAML subset parser to avoid a hard pyyaml dependency.
# skill.yaml is simple enough for a lightweight approach, but for robustness
# we try to import yaml and fall back to a TOML-style manual parser.
def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML/JSON file.  Tries PyYAML first, falls back to JSON."""
    try:
        import yaml  # type: ignore[import-untyped]

        with path.open(encoding="utf-8") as f:
            data: object = yaml.safe_load(f)
        return dict(data) if isinstance(data, dict) else {}  # type: ignore[arg-type]
    except ImportError:
        pass

    import json as _json

    try:
        with path.open(encoding="utf-8") as f:
            data_j: object = _json.load(f)
        return dict(data_j) if isinstance(data_j, dict) else {}  # type: ignore[arg-type]
    except Exception:
        _log.warning("Cannot parse %s — install PyYAML for full YAML support", path)
        return {}


# ---------------------------------------------------------------------------
# SkillMeta
# ---------------------------------------------------------------------------

_SKILL_YAML = "skill.yaml"
_PROMPT_MD = "prompt.md"


@dataclass(frozen=True, slots=True)
class SkillMeta:
    """Typed representation of a skill's ``skill.yaml``."""

    id: str
    """Unique skill identifier (defaults to directory name)."""

    name: str
    """Human-readable skill name."""

    description: str = ""
    """Short description of what this skill does."""

    version: str = "0.1.0"
    """Skill version string."""

    tags: list[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    """Searchable tags for skill discovery."""

    entry: str = "tools.py"
    """Entry-point module for tool implementations."""

    prompt_fragment: str = ""
    """System prompt guidance loaded from ``prompt.md`` (populated at load time)."""

    source: str = "unknown"
    """Source layer: ``builtin``, ``user``, or ``project``."""

    path: Path = field(default_factory=lambda: Path("."))
    """Absolute path to the skill directory."""


def load_skill_meta(skill_dir: Path, source: str = "unknown") -> SkillMeta | None:
    """Load a ``SkillMeta`` from a skill directory.

    Returns ``None`` if the directory is not a valid skill (missing ``skill.yaml``).
    """
    yaml_path = skill_dir / _SKILL_YAML
    if not yaml_path.is_file():
        return None

    data = _load_yaml(yaml_path)
    if not data:
        _log.warning("Empty or invalid skill.yaml in %s", skill_dir)
        return None

    skill_id = str(data.get("id", skill_dir.name))
    name = str(data.get("name", skill_id))

    # Load optional prompt fragment
    prompt = ""
    prompt_path = skill_dir / _PROMPT_MD
    if prompt_path.is_file():
        prompt = prompt_path.read_text(encoding="utf-8").strip()

    tags_raw: object = data.get("tags", [])
    tags: list[str] = (
        [str(t) for t in list(tags_raw)]  # pyright: ignore[reportUnknownArgumentType,reportUnknownVariableType]
        if isinstance(tags_raw, list)
        else []
    )

    return SkillMeta(
        id=skill_id,
        name=name,
        description=str(data.get("description", "")),
        version=str(data.get("version", "0.1.0")),
        tags=tags,
        entry=str(data.get("entry", "tools.py")),
        prompt_fragment=prompt,
        source=source,
        path=skill_dir.resolve(),
    )


def write_skill_yaml(skill_dir: Path, meta: SkillMeta) -> None:
    """Write a ``skill.yaml`` to a skill directory.

    Uses JSON format as the universal fallback (valid YAML).
    """
    import json

    skill_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "id": meta.id,
        "name": meta.name,
        "description": meta.description,
        "version": meta.version,
        "tags": meta.tags,
        "entry": meta.entry,
    }
    yaml_path = skill_dir / _SKILL_YAML
    yaml_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    if meta.prompt_fragment:
        prompt_path = skill_dir / _PROMPT_MD
        prompt_path.write_text(meta.prompt_fragment + "\n", encoding="utf-8")
