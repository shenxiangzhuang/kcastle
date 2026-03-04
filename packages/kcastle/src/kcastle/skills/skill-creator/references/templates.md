# Templates

## Improvement report

Produce this after each edit-and-rerun cycle.

```markdown
## Skill iteration report

- **Skill path**: `~/.kcastle/skills/example-skill/`
- **What changed**: tightened step 3 output format; added error-handling edge case.
- **Why**: eval prompt produced malformed YAML.
- **Eval results**: 3/3 checks pass; qualitative feedback pending.
- **Risks**: new constraint may be too strict for freeform prompts.
- **Next focus**: run broader prompt set to verify no regression.
```
