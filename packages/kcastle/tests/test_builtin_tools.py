from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from kai import Tool

from kcastle.skills import SkillManager
from kcastle.tools import create_builtin_tools


def _tool_map(tools: list[Tool]) -> dict[str, Tool]:
    return {t.name: t for t in tools}


@pytest.mark.asyncio
async def test_builtin_toolset_contains_core_and_skill_tools(tmp_path: Path) -> None:
    manager = SkillManager(
        user_skills_dir=tmp_path / "user",
        project_skills_dir=tmp_path / "project",
    )
    manager.discover()

    tools = create_builtin_tools(workspace=tmp_path, skill_manager=manager)
    names = {t.name for t in tools}

    assert "read_file" in names
    assert "write_file" in names
    assert "edit_file" in names
    assert "run_bash" in names
    assert "create_skill" in names
    assert "update_skill" in names
    assert "list_skills" in names


@pytest.mark.asyncio
async def test_core_file_tools_roundtrip(tmp_path: Path) -> None:
    manager = SkillManager(
        user_skills_dir=tmp_path / "user",
        project_skills_dir=tmp_path / "project",
    )
    manager.discover()
    tools = _tool_map(create_builtin_tools(workspace=tmp_path, skill_manager=manager))

    write_tool = tools["write_file"]
    read_tool = tools["read_file"]
    edit_tool = tools["edit_file"]

    write_result = await write_tool.execute(
        cast(Any, write_tool).Params(path="notes.txt", content="hello\nworld")
    )
    assert not write_result.is_error

    read_result = await read_tool.execute(cast(Any, read_tool).Params(path="notes.txt"))
    assert not read_result.is_error
    assert "hello" in read_result.output

    edit_result = await edit_tool.execute(
        cast(Any, edit_tool).Params(path="notes.txt", old="world", new="kcastle")
    )
    assert not edit_result.is_error

    read_result2 = await read_tool.execute(cast(Any, read_tool).Params(path="notes.txt"))
    assert "kcastle" in read_result2.output


@pytest.mark.asyncio
async def test_create_skill_tool_creates_skill_files(tmp_path: Path) -> None:
    user_dir = tmp_path / "user"
    project_dir = tmp_path / "project"
    manager = SkillManager(user_skills_dir=user_dir, project_skills_dir=project_dir)
    manager.discover()

    tools = _tool_map(create_builtin_tools(workspace=tmp_path, skill_manager=manager))
    create_tool = tools["create_skill"]

    result = await create_tool.execute(
        cast(Any, create_tool).Params(
            skill_id="my-skill",
            description="demo",
            instructions="# My Skill\n\nDo x.",
        )
    )
    assert not result.is_error
    assert (user_dir / "my-skill" / "SKILL.md").is_file()
    assert not (project_dir / "my-skill" / "SKILL.md").exists()


@pytest.mark.asyncio
async def test_create_skill_tool_rejects_project_target(tmp_path: Path) -> None:
    manager = SkillManager(
        user_skills_dir=tmp_path / "user",
        project_skills_dir=tmp_path / "project",
    )
    manager.discover()

    tools = _tool_map(create_builtin_tools(workspace=tmp_path, skill_manager=manager))
    create_tool = tools["create_skill"]

    result = await create_tool.execute(
        cast(Any, create_tool).Params(
            skill_id="my-skill",
            description="demo",
            instructions="# My Skill\n\nDo x.",
            target="project",
        )
    )
    assert result.is_error


def test_builtin_skills_are_discoverable() -> None:
    package_root = Path(__file__).resolve().parents[1] / "src" / "kcastle"
    builtin_dir = package_root / "skills"
    manager = SkillManager(
        user_skills_dir=Path("/tmp/none-user"),
        project_skills_dir=Path("/tmp/none-project"),
        builtin_skills_dir=builtin_dir,
    )

    skills = manager.discover()
    ids = {s.id for s in skills}
    assert "skill-creator" in ids
    assert "skill-installer" in ids
