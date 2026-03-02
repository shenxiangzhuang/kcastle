"""Built-in core coding tools for kcastle.

These tools are always available to the agent and provide a minimal,
workspace-scoped coding toolbox.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from kai import Tool, ToolResult
from pydantic import BaseModel, Field, PrivateAttr

_MAX_OUTPUT_CHARS = 12_000
_MAX_READ_CHARS = 50_000


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    clipped = text[:limit]
    return f"{clipped}\n\n...[truncated {len(text) - limit} chars]"


def _safe_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class _WorkspaceTool(Tool):
    _workspace: Path = PrivateAttr()
    _user_skills_dir: Path = PrivateAttr()

    @classmethod
    def for_workspace(cls, workspace: Path) -> _WorkspaceTool:
        tool = cls.model_construct()
        tool._workspace = workspace.resolve()
        tool._user_skills_dir = (Path.home() / ".kcastle" / "skills").resolve(strict=False)
        return tool

    def _resolve(self, user_path: str) -> Path:
        candidate = Path(user_path).expanduser()
        absolute = (
            candidate.resolve(strict=False)
            if candidate.is_absolute()
            else (self._workspace / candidate).resolve(strict=False)
        )

        is_workspace_path = absolute.is_relative_to(self._workspace)
        is_user_skill_path = absolute.is_relative_to(self._user_skills_dir)
        if not is_workspace_path and not is_user_skill_path:
            raise ValueError(f"Path escapes workspace: {user_path}")
        return absolute

    def _display_path(self, absolute: Path) -> str:
        if absolute.is_relative_to(self._workspace):
            return absolute.relative_to(self._workspace).as_posix()
        if absolute.is_relative_to(self._user_skills_dir):
            return f"~/.kcastle/skills/{absolute.relative_to(self._user_skills_dir).as_posix()}"
        return absolute.as_posix()


class ReadFileTool(_WorkspaceTool):
    name: str = "read_file"
    description: str = "Read a UTF-8 text file from the workspace."

    class Params(BaseModel):
        path: str = Field(description="Path to file, relative to workspace preferred.")
        start_line: int = Field(default=1, ge=1, description="1-based inclusive start line.")
        end_line: int | None = Field(default=None, ge=1, description="1-based inclusive end line.")

    async def execute(self, params: ReadFileTool.Params) -> ToolResult:
        try:
            path = self._resolve(params.path)
            if not path.is_file():
                return ToolResult.error(f"Not a file: {params.path}")
            content = _safe_text(path)
            lines = content.splitlines()
            start = params.start_line - 1
            end = params.end_line if params.end_line is not None else len(lines)
            selected = "\n".join(lines[start:end])
            return ToolResult(output=_truncate(selected, _MAX_READ_CHARS))
        except (OSError, ValueError, UnicodeError) as e:
            return ToolResult.error(str(e))


class WriteFileTool(_WorkspaceTool):
    name: str = "write_file"
    description: str = "Write UTF-8 text content to a workspace file."

    class Params(BaseModel):
        path: str = Field(description="Path to file.")
        content: str = Field(description="Full file content to write.")
        create_dirs: bool = Field(default=True, description="Create parent directories if missing.")

    async def execute(self, params: WriteFileTool.Params) -> ToolResult:
        try:
            path = self._resolve(params.path)
            if params.create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(params.content, encoding="utf-8")
            return ToolResult(output=f"Wrote {params.path}")
        except (OSError, ValueError) as e:
            return ToolResult.error(str(e))


class EditFileTool(_WorkspaceTool):
    name: str = "edit_file"
    description: str = "Replace text in a UTF-8 workspace file."

    class Params(BaseModel):
        path: str = Field(description="Path to file.")
        old: str = Field(description="Text to replace.")
        new: str = Field(description="Replacement text.")
        replace_all: bool = Field(
            default=False,
            description="Replace all matches instead of first.",
        )

    async def execute(self, params: EditFileTool.Params) -> ToolResult:
        try:
            path = self._resolve(params.path)
            if not path.is_file():
                return ToolResult.error(f"Not a file: {params.path}")
            content = _safe_text(path)
            if params.old not in content:
                return ToolResult.error("Old text not found")
            count = content.count(params.old)
            new_content = (
                content.replace(params.old, params.new)
                if params.replace_all
                else content.replace(params.old, params.new, 1)
            )
            path.write_text(new_content, encoding="utf-8")
            replaced = count if params.replace_all else 1
            return ToolResult(output=f"Edited {params.path} (replaced {replaced} occurrence(s))")
        except (OSError, ValueError, UnicodeError) as e:
            return ToolResult.error(str(e))


class ListDirTool(_WorkspaceTool):
    name: str = "list_dir"
    description: str = "List files and directories under a workspace path."

    class Params(BaseModel):
        path: str = Field(default=".", description="Directory path.")
        recursive: bool = Field(default=False, description="Recursively list files.")
        max_entries: int = Field(default=200, ge=1, le=2000, description="Entry cap.")

    async def execute(self, params: ListDirTool.Params) -> ToolResult:
        try:
            base = self._resolve(params.path)
            if not base.is_dir():
                return ToolResult.error(f"Not a directory: {params.path}")
            entries = sorted(base.rglob("*")) if params.recursive else sorted(base.iterdir())
            out: list[str] = []
            for item in entries[: params.max_entries]:
                shown = self._display_path(item)
                out.append(f"{shown}/" if item.is_dir() else shown)
            return ToolResult(output="\n".join(out) if out else "(empty)")
        except (OSError, ValueError) as e:
            return ToolResult.error(str(e))


class FindFilesTool(_WorkspaceTool):
    name: str = "find_files"
    description: str = "Find files by glob pattern within the workspace."

    class Params(BaseModel):
        pattern: str = Field(default="**/*", description="Glob pattern.")
        path: str = Field(default=".", description="Base directory.")
        max_results: int = Field(default=200, ge=1, le=2000, description="Result cap.")

    async def execute(self, params: FindFilesTool.Params) -> ToolResult:
        try:
            base = self._resolve(params.path)
            if not base.is_dir():
                return ToolResult.error(f"Not a directory: {params.path}")
            results: list[str] = []
            for item in base.glob(params.pattern):
                shown = self._display_path(item)
                results.append(f"{shown}/" if item.is_dir() else shown)
                if len(results) >= params.max_results:
                    break
            return ToolResult(output="\n".join(results) if results else "(no matches)")
        except (OSError, ValueError) as e:
            return ToolResult.error(str(e))


class GrepTool(_WorkspaceTool):
    name: str = "grep_text"
    description: str = "Search for text or regex matches in workspace files."

    class Params(BaseModel):
        query: str = Field(description="Text or regex query.")
        path: str = Field(default=".", description="Base directory.")
        include_pattern: str = Field(default="**/*", description="File glob to include.")
        is_regex: bool = Field(default=False, description="Interpret query as regex.")
        max_results: int = Field(default=200, ge=1, le=2000, description="Result cap.")

    async def execute(self, params: GrepTool.Params) -> ToolResult:
        try:
            base = self._resolve(params.path)
            if not base.is_dir():
                return ToolResult.error(f"Not a directory: {params.path}")

            needle = re.compile(params.query) if params.is_regex else None
            matches: list[str] = []
            for file_path in base.glob(params.include_pattern):
                if not file_path.is_file():
                    continue
                try:
                    text = _safe_text(file_path)
                except (OSError, UnicodeError):
                    continue
                for line_no, line in enumerate(text.splitlines(), start=1):
                    ok = bool(needle.search(line)) if needle else params.query in line
                    if ok:
                        shown = self._display_path(file_path)
                        matches.append(f"{shown}:{line_no}: {line}")
                        if len(matches) >= params.max_results:
                            return ToolResult(output="\n".join(matches))
            return ToolResult(output="\n".join(matches) if matches else "(no matches)")
        except (OSError, ValueError, re.error) as e:
            return ToolResult.error(str(e))


class BashTool(_WorkspaceTool):
    name: str = "run_bash"
    description: str = "Run a shell command inside the workspace directory."

    class Params(BaseModel):
        command: str = Field(description="Shell command to execute.")
        timeout_seconds: int = Field(default=30, ge=1, le=600, description="Execution timeout.")

    async def execute(self, params: BashTool.Params) -> ToolResult:
        dangerous = [r"\brm\s+-rf\s+/", r"\bshutdown\b", r"\breboot\b", r":\(\)\s*\{"]
        if any(re.search(rule, params.command) for rule in dangerous):
            return ToolResult.error("Blocked potentially destructive command")

        try:
            proc = await asyncio.create_subprocess_shell(
                params.command,
                cwd=str(self._workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=float(params.timeout_seconds)
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult.error(f"Command timed out after {params.timeout_seconds}s")

            out = (stdout or b"").decode("utf-8", errors="replace")
            err = (stderr or b"").decode("utf-8", errors="replace")
            text = out if not err else f"{out}\n{err}".strip()
            text = _truncate(text, _MAX_OUTPUT_CHARS)
            if proc.returncode and proc.returncode != 0:
                return ToolResult.error(f"exit={proc.returncode}\n{text}")
            return ToolResult(output=text or "(no output)")
        except (OSError, ValueError, RuntimeError) as e:
            return ToolResult.error(str(e))


def create_core_tools(*, workspace: Path) -> list[Tool]:
    """Create built-in workspace-scoped coding tools."""
    return [
        ReadFileTool.for_workspace(workspace),
        WriteFileTool.for_workspace(workspace),
        EditFileTool.for_workspace(workspace),
        ListDirTool.for_workspace(workspace),
        FindFilesTool.for_workspace(workspace),
        GrepTool.for_workspace(workspace),
        BashTool.for_workspace(workspace),
    ]
