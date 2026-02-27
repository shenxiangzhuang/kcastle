"""Built-in skill tools."""

from __future__ import annotations

from kai import Tool, ToolResult
from pydantic import BaseModel, Field, PrivateAttr

from kcastle.skills import SkillManager


class _SkillTool(Tool):
    _manager: SkillManager = PrivateAttr()

    @classmethod
    def for_manager(cls, manager: SkillManager) -> _SkillTool:
        tool = cls.model_construct()
        tool._manager = manager
        return tool


class ListSkillsTool(_SkillTool):
    name: str = "skills_list"
    description: str = "List discovered skills across builtin, user, and project layers."

    class Params(BaseModel):
        query: str = Field(default="", description="Optional search query.")
        max_results: int = Field(default=50, ge=1, le=500, description="Result cap.")

    async def execute(self, params: ListSkillsTool.Params) -> ToolResult:
        if params.query.strip():
            matches = self._manager.search(params.query)
            rows = [
                f"{m.meta.id} | {m.meta.source} | score={m.score:.2f} | {m.meta.description}"
                for m in matches[: params.max_results]
            ]
            return ToolResult(output="\n".join(rows) if rows else "(no matches)")

        rows = [
            f"{s.id} | {s.source} | {s.description}"
            for s in self._manager.all_skills()[: params.max_results]
        ]
        return ToolResult(output="\n".join(rows) if rows else "(no skills)")


def create_skill_tools(*, manager: SkillManager) -> list[Tool]:
    """Create built-in read-only skill tools."""
    return [ListSkillsTool.for_manager(manager)]
