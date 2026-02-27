from __future__ import annotations

import shutil
import uuid
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
    assert "skills_list" in names


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
async def test_core_file_tools_allow_user_skill_dir_with_tilde(
    tmp_path: Path,
) -> None:
    manager = SkillManager(
        user_skills_dir=tmp_path / "user",
        project_skills_dir=tmp_path / "project",
    )
    manager.discover()

    tools = _tool_map(create_builtin_tools(workspace=tmp_path / "workspace", skill_manager=manager))
    write_tool = tools["write_file"]
    read_tool = tools["read_file"]

    skill_id = f"kcastle-test-{uuid.uuid4().hex[:8]}"
    target = f"~/.kcastle/skills/{skill_id}/SKILL.md"
    content = "---\nname: demo-skill\ndescription: demo\n---\n"

    try:
        write_result = await write_tool.execute(
            cast(Any, write_tool).Params(path=target, content=content)
        )
        assert not write_result.is_error

        read_result = await read_tool.execute(cast(Any, read_tool).Params(path=target))
        assert not read_result.is_error
        assert "name: demo-skill" in read_result.output
    finally:
        shutil.rmtree(Path.home() / ".kcastle" / "skills" / skill_id, ignore_errors=True)


@pytest.mark.asyncio
async def test_skills_list_tool_lists_discovered_skills(tmp_path: Path) -> None:
    user_dir = tmp_path / "user"
    skill_dir = user_dir / "demo-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo skill for listing\n---\n\n# demo\n",
        encoding="utf-8",
    )

    manager = SkillManager(
        user_skills_dir=user_dir,
        project_skills_dir=tmp_path / "project",
    )
    manager.discover()

    tools = _tool_map(create_builtin_tools(workspace=tmp_path, skill_manager=manager))
    list_tool = tools["skills_list"]

    result = await list_tool.execute(cast(Any, list_tool).Params(query="demo", max_results=10))
    assert not result.is_error
    assert "demo-skill" in result.output


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
