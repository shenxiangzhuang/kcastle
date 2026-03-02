"""Built-in prompt fragments for kcastle.

The system prompt is assembled from ordered blocks:

1. **Identity** — built-in ``identity.md`` or user override from config
2. **Runtime context** — current time, OS, Python version (auto-generated)
3. **Workspace rules** — ``AGENTS.md`` from the project root (if present)
4. **Skill descriptions** — from active skills (auto-generated)

If the user sets ``system_prompt`` in ``config.yaml``, it replaces the
built-in identity entirely.
"""

from __future__ import annotations

import datetime
import platform
from importlib import resources
from pathlib import Path


def load_identity_prompt() -> str:
    """Load the built-in identity prompt for agent k."""
    ref = resources.files("kcastle.prompts").joinpath("identity.md")
    return ref.read_text(encoding="utf-8")


def build_runtime_context() -> str:
    """Build dynamic runtime context block."""
    now = datetime.datetime.now(datetime.UTC).astimezone()
    lines = [
        "## Runtime Context",
        "",
        f"- Current time: {now.strftime('%Y-%m-%d %H:%M %Z')}",
        f"- OS: {platform.system()} {platform.release()}",
        f"- Python: {platform.python_version()}",
    ]
    return "\n".join(lines)


def read_workspace_prompt(workspace: Path) -> str | None:
    """Read ``AGENTS.md`` from a workspace root, if present."""
    agents_file = workspace / "AGENTS.md"
    if agents_file.is_file():
        text = agents_file.read_text(encoding="utf-8").strip()
        return text or None
    return None


def assemble_system_prompt(
    *,
    identity: str | None = None,
    runtime_context: str | None = None,
    workspace_prompt: str | None = None,
    skill_prompts: str | None = None,
    user_override: str | None = None,
) -> str:
    """Assemble the full system prompt from fragments.

    Order:
        1. Identity (built-in or user override)
        2. Runtime context (auto-generated)
        3. Workspace rules (AGENTS.md)
        4. Skill descriptions (from active skills)

    If *user_override* is provided it replaces the built-in identity.
    """
    blocks: list[str] = []

    # 1 — Identity
    if user_override:
        blocks.append(user_override)
    elif identity:
        blocks.append(identity)
    else:
        blocks.append(load_identity_prompt())

    # 2 — Runtime context
    if runtime_context:
        blocks.append(runtime_context)

    # 3 — Workspace rules
    if workspace_prompt:
        blocks.append(f"## Workspace Rules\n\n{workspace_prompt}")

    # 4 — Skill guidance
    if skill_prompts:
        blocks.append(f"## Skill Guidance\n\n{skill_prompts}")

    return "\n\n".join(blocks)
