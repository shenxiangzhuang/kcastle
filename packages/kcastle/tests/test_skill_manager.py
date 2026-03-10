from __future__ import annotations

from pathlib import Path

from kcastle.skills.manager import SkillManager, find_project_root


def _write_skill(skill_dir: Path, *, name: str, description: str, tags: str = "") -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    tag_line = f"\ntags: [{tags}]" if tags else ""
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}{tag_line}\n---\n\n# body\n",
        encoding="utf-8",
    )


def test_find_project_root_uses_git_marker(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    nested = project / "a" / "b"
    (project / ".git").mkdir(parents=True, exist_ok=True)
    nested.mkdir(parents=True, exist_ok=True)

    assert find_project_root(nested) == project


def test_find_project_root_falls_back_to_pyproject(tmp_path: Path) -> None:
    project = tmp_path / "project"
    nested = project / "pkg" / "src"
    project.mkdir(parents=True, exist_ok=True)
    (project / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    nested.mkdir(parents=True, exist_ok=True)

    assert find_project_root(nested) == project


def test_discover_prefers_project_over_user_over_builtin(tmp_path: Path) -> None:
    builtin_dir = tmp_path / "builtin"
    user_dir = tmp_path / "user"
    project_dir = tmp_path / "project"

    _write_skill(
        builtin_dir / "shared-skill",
        name="shared-skill",
        description="builtin layer",
    )
    _write_skill(
        user_dir / "shared-skill",
        name="shared-skill",
        description="user layer",
    )
    _write_skill(
        project_dir / "shared-skill",
        name="shared-skill",
        description="project layer",
    )

    manager = SkillManager(
        user_skills_dir=user_dir,
        project_skills_dir=project_dir,
        builtin_skills_dir=builtin_dir,
    )

    manager.discover()
    skill = manager.get_skill("shared-skill")

    assert skill is not None
    assert skill.source == "project"
    assert skill.name == "shared-skill"
    assert skill.description == "project layer"


def test_search_matches_hyphenated_and_underscored_queries(tmp_path: Path) -> None:
    user_dir = tmp_path / "user"
    _write_skill(
        user_dir / "skill-creator",
        name="skill-creator",
        description="Create reusable skills",
        tags="skill_creator, automation",
    )

    manager = SkillManager(user_skills_dir=user_dir, top_k=5)
    manager.discover()

    matches_dash = manager.search("skill-creator")
    matches_underscore = manager.search("skill_creator")

    assert matches_dash
    assert matches_underscore
    assert matches_dash[0].skill.name == "skill-creator"
    assert matches_underscore[0].skill.name == "skill-creator"


def test_expand_hints_injects_skill_instructions(tmp_path: Path) -> None:
    user_dir = tmp_path / "user"
    _write_skill(
        user_dir / "my-tool",
        name="my-tool",
        description="Performs a useful task",
    )
    (user_dir / "my-tool" / "SKILL.md").write_text(
        (
            "---\nname: my-tool\ndescription: Performs a useful task\n---\n\n"
            "# My Tool\n\nDo the thing.\n"
        ),
        encoding="utf-8",
    )

    manager = SkillManager(user_skills_dir=user_dir)
    manager.discover()

    result = manager.expand_hints("Please use $my-tool for this.")

    assert "<skill_expansion>" in result
    assert "[my-tool]" in result
    assert "Do the thing." in result


def test_expand_hints_returns_input_unchanged_when_no_hints(tmp_path: Path) -> None:
    manager = SkillManager(user_skills_dir=tmp_path / "user")
    manager.discover()

    user_input = "No hints here, just a plain message."
    assert manager.expand_hints(user_input) == user_input


def test_expand_hints_ignores_unknown_hints(tmp_path: Path) -> None:
    manager = SkillManager(user_skills_dir=tmp_path / "user")
    manager.discover()

    user_input = "Use $nonexistent-skill please."
    assert manager.expand_hints(user_input) == user_input
