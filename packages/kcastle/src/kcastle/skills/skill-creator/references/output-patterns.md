# Output Patterns

## Template pattern

Use when the skill must produce output in a specific format:

```markdown
## Output format

Produce output matching this template:

\```
TITLE: <descriptive title>
STATUS: <pass|fail|partial>
FINDINGS:
- <finding 1>
- <finding 2>
RECOMMENDATION: <one-sentence action item>
\```
```

Provide a complete example alongside the template so K can see both
structure and realistic content.

## Example-driven pattern

Use when showing concrete examples is more effective than describing rules:

```markdown
## Examples

### Good output
<complete realistic example>

### Bad output (avoid)
<example of what NOT to produce, with explanation of why>
```

Include 1-2 good examples and optionally 1 bad example.
More examples cost tokens — add them only if the format is genuinely ambiguous.

## Quality checklist pattern

Use when output must meet specific criteria:

```markdown
## Quality checklist

Before delivering output, verify:
- [ ] All required sections present
- [ ] No placeholder text remaining
- [ ] Code examples are syntactically valid
- [ ] File paths use correct format
```

Keep checklists to 3-6 items. Longer lists get ignored.

## Progressive output pattern

Use when the skill produces output in stages:

```markdown
## Output stages

1. **Quick summary** — 1-2 sentences of what was found/done.
2. **Detailed results** — Full output with supporting data.
3. **Next steps** — Actionable recommendations.
```

This lets K front-load the most useful information.
