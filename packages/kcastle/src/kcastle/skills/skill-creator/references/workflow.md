# Workflow

## Intake checklist

Capture these four items before writing:

1. Capability: what behavior the skill should enable.
2. Trigger context: which user requests should invoke the skill.
3. Deliverable: what outputs/artifacts should be produced.
4. Constraints: tools, style, safety, and scope limits.

If the conversation already provides most answers, summarize assumptions and ask for confirmation once.

## Authoring rules

Use this baseline layout:

```text
<skill-folder>/
├── SKILL.md
├── scripts/        # optional deterministic helpers
└── references/     # optional deep guides
```

Frontmatter must include:

- `name`: stable lookup key.
- `description`: what the skill does + when it should trigger.

Body should contain:

- Decision/trigger interpretation guidance.
- Ordered execution steps.
- Expected output format.
- Edge cases and fallback behavior.

Prefer imperative language with rationale. Avoid long "MUST"-style lists unless strictness is required.

## Creation flow

1. Choose a stable skill `name`.
2. Draft a focused `description` with clear trigger hints.
3. Write `SKILL.md` skeleton and minimal executable workflow.
4. Add references/scripts only where needed.
5. Propose 2-3 realistic eval prompts.

## Update flow

1. Identify the weakest part from feedback (keyword matching, output quality, consistency).
2. Make a targeted patch first; avoid broad rewrites.
3. Re-run the same eval prompts (see `references/evals.md` for how) before adding new cases.
4. Generalize successful changes; remove overfit instructions.

## Anti-patterns

- Overfitting to one prompt instead of describing intent.
- Encoding too many brittle formatting rules.
- Repeating shell/tool boilerplate inside the prompt when a `scripts/` helper should exist.
- Adding "negative boundary" words in `description` (they become matchable tokens and attract false positives).
- Referencing tools, scripts, or infrastructure that does not exist in the current environment.
