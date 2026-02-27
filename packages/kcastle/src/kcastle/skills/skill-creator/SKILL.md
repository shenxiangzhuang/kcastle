---
name: skill-creator
description: Create or update kcastle skills in SKILL.md format when user asks to add capabilities or automate repeated workflows.
---

# skill-creator

Use this skill when the user asks k to create, modify, or improve a skill.

Rules:
- Execute real changes via tools (`create_skill`, `update_skill`, `write_file`, `edit_file`).
- Do not return large code blocks unless user explicitly asks for the full file content.
- Every skill must live in `<skill-id>/SKILL.md` with valid frontmatter (`name`, `description`).
- Agent-created skills must be managed in user scope (`~/.kcastle/skills`).
- Treat project scope (`.skills`) as user-maintained; do not write there automatically.
- Keep instructions concise and operational.

Creation flow:
1. Choose skill id in lowercase-hyphen format.
2. Write frontmatter with precise trigger description.
3. Add minimal instructions body focused on repeatable workflow.
4. If needed, update existing skill with `append_instructions`.
