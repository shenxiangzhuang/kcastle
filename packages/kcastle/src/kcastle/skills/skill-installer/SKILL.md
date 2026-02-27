---
name: skill-installer
description: Discover and install or scaffold skills into project or user skill directories when user asks to add available skills.
---

# skill-installer

Use this skill when user asks what skills are available or asks to install/add one.

Rules:
- Use `list_skills` first when intent is ambiguous.
- Agent-created skills must be managed in user scope (`~/.kcastle/skills`).
- Treat project scope (`.skills`) as user-maintained and read-only for autonomous skill writes.
- After creating/updating a skill, remind the user that runtime restart may be needed to refresh loaded prompts.
