---
name: skill-creator
description: Create new skills and iteratively improve existing skills in SKILL.md format when users ask to automate repeated workflows, add reusable capabilities, or tune skill triggering behavior.
---

# skill-creator

Use this skill when the user asks to create, modify, optimize, or evaluate a skill.

## Operating rules

- Discover candidate skills via `skills_list`; perform edits via file tools (`write_file`, `edit_file`).
- Keep responses concise; avoid dumping long file contents unless explicitly requested.
- Every skill must be `<skill-id>/SKILL.md` with valid frontmatter (`name`, `description`).
- Agent-created skills must be managed under `~/.kcastle/skills`.
- Use one folder per skill: `~/.kcastle/skills/<skill-id>/SKILL.md`.
- A skill folder may also include `scripts/` for reusable helper scripts.
- Use lowercase-hyphen IDs only (`^[a-z0-9-]{1,64}$`).
- New or updated skills should be prepared as file edits and confirmed with the user before write.
- Treat project scope (`.skills`) as user-managed unless user explicitly asks for direct edits.
- Prioritize clear intent and trigger conditions in `description`; avoid vague wording.

## Workflow

### 1) Capture intent

Before writing, clarify:

1. What capability should the skill enable?
2. In what user contexts should it trigger?
3. What outputs/artifacts should the skill produce?
4. Are there constraints (tooling, format, style, safety, scope)?

If the conversation already contains these details, summarize assumptions and ask for confirmation.

### 2) Draft or update the skill

When creating:

1. Choose a stable skill ID.
2. Write frontmatter:
	- `name`: exact skill ID
	- `description`: what it does + when to trigger (be specific and slightly proactive)
3. Write body instructions in imperative style.

When updating:

- Preserve user intent and existing useful structure.
- Prefer targeted edits over full rewrites unless the current version is fundamentally broken.
- Explain key changes briefly.

### 3) Include practical guidance in SKILL.md

The body should usually contain:

- Trigger interpretation guidance
- Step-by-step execution flow
- Output format expectations
- Edge cases / fallback behavior

Keep SKILL.md focused and operational. If content grows too large, split details into referenced companion files.

Recommended skill layout:

- `<skill-id>/SKILL.md` (required)
- `<skill-id>/scripts/` (optional, executable helpers)

If `scripts/` is present, reference script names and expected inputs/outputs in SKILL.md so execution is predictable.

### 4) Create lightweight eval prompts

After drafting, propose 2-3 realistic prompts to validate behavior.

- For objective tasks, include expected checks (structure, fields, transformations).
- For subjective tasks, focus on qualitative review criteria.

Use results to identify missing instructions, over-constraints, or ambiguous triggers.

### 5) Iterate

After feedback:

1. Generalize from failures (avoid overfitting to one example).
2. Remove instructions that add complexity without improving outcomes.
3. Strengthen rationale-oriented wording (explain why, not just rigid rules).
4. Update the skill and re-test with revised prompts.

Repeat until the user is satisfied or changes stop producing meaningful gains.

## Description optimization

When the user asks to improve triggering:

1. Draft a mixed set of should-trigger and should-not-trigger prompts.
2. Prefer realistic near-miss negatives over obviously unrelated prompts.
3. Tighten description wording to improve precision without becoming too narrow.
4. Show before/after description and explain expected trigger behavior changes.

## Output contract

When finishing a creation/update cycle, report:

- Skill path modified
- Summary of what changed
- Suggested test prompts
- Any assumptions or open questions
