from __future__ import annotations

from pathlib import Path

from kcastle.skills.skill import Skill

# --- Loading / Saving ---


def test_load_requires_skill_md(tmp_path: Path) -> None:
    skill_dir = tmp_path / "legacy-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "skill.yaml").write_text("name: legacy\ndescription: old\n", encoding="utf-8")

    assert Skill.load(skill_dir, source="project") is None


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    skill_dir = tmp_path / "new-skill"
    skill = Skill(
        name="new-skill",
        description="Do a specific workflow.",
        path=skill_dir,
        tags=["demo"],
        instructions="# New Skill\n\nUse tools first.",
        source="project",
    )

    skill.save()
    loaded = Skill.load(skill_dir, source="project")

    assert loaded is not None
    assert loaded.name == "new-skill"
    assert loaded.description == "Do a specific workflow."
    assert "Use tools first." in loaded.instructions


def test_load_preserves_frontmatter_name(tmp_path: Path) -> None:
    skill_dir = tmp_path / "human-name"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: Friendly Skill Name\ndescription: Helpful workflow\n---\n\n# body\n",
        encoding="utf-8",
    )

    loaded = Skill.load(skill_dir, source="project")

    assert loaded is not None
    assert loaded.name == "Friendly Skill Name"


def test_save_handles_yaml_special_chars(tmp_path: Path) -> None:
    skill_dir = tmp_path / "special-chars"
    skill = Skill(
        name="Skill: Creator",
        description="Create: skills # safely",
        path=skill_dir,
        tags=["build", "safe:mode"],
        instructions="Use this with care.",
        source="project",
    )

    skill.save()
    loaded = Skill.load(skill_dir, source="project")

    assert loaded is not None
    assert loaded.name == "Skill: Creator"
    assert loaded.description == "Create: skills # safely"
    assert loaded.tags == ["build", "safe:mode"]


# --- Rendering ---


def test_extract_hints_normalizes_and_deduplicates() -> None:
    text = "Use $skill_creator then $skill-creator and also $skill-installer"
    assert Skill.extract_hints(text) == ["skill-creator", "skill-installer"]


def test_render_expanded_includes_instruction_body() -> None:
    skill = Skill(
        name="skill-creator",
        description="Create skills",
        path=Path("/tmp/skill-creator"),
        instructions="# creator\n\nDo things.",
        source="builtin",
    )

    block = Skill.render_expanded([skill])
    assert "<skill_expansion>" in block
    assert "[skill-creator] (builtin)" in block
    assert "Do things." in block
