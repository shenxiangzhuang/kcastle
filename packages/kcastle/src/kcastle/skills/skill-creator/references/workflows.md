# Workflows

## Sequential workflow pattern

Use when steps must execute in a fixed order:

```markdown
## Workflow

1. **Validate inputs** — Check required files exist, format is correct.
2. **Transform** — Apply the main operation.
3. **Verify output** — Confirm result matches expected shape/schema.
4. **Report** — Summarize what was done.
```

Keep steps short and imperative. Add sub-steps only when the main step is genuinely complex.

## Conditional workflow pattern

Use when the next action depends on input characteristics:

```markdown
## Decision tree

- Is the input a single file or a directory?
  - **Single file** → proceed to "Process single file"
  - **Directory** → proceed to "Batch processing"
- Does the file contain images?
  - **Yes** → include OCR step
  - **No** → skip to text extraction
```

Place the decision tree early in SKILL.md so K can route quickly.

## Iterative workflow pattern

Use when quality improves through repeated passes:

```markdown
## Workflow

1. **Draft** — Produce initial output.
2. **Self-review** — Check against quality criteria.
3. **Refine** — Fix issues found in review.
4. **Final check** — One last pass; if still failing, report remaining issues.
```

Limit iterations (2-3 max) to avoid infinite loops.

## Error handling

Always specify what K should do when a step fails:

```markdown
If step 2 fails:
- Log the error with context
- Attempt fallback approach X
- If fallback also fails, report to user with actionable next steps
```
