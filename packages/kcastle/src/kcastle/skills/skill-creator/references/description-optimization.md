# Description optimization

`description` is the primary signal for skill discovery. Optimize it for keyword coverage.

## How skill matching works in kcastle

`SkillManager.search(query)` scores each skill by **token overlap**:

1. Tokenize the query: lowercase, split on `-`, `_`, `.`, extract `[a-z0-9]+` words.
2. Tokenize the skill's `name + description + tags` the same way.
3. Score = `|overlap| / |query_tokens|`. Higher is better; zero means no match.

Additionally, users can explicitly invoke a skill with `$skill-name` in their message,
which bypasses search and directly injects the full skill body.

**Implication**: every word in `description` becomes a matchable token. Adding irrelevant
words (even in negative phrases like "not for data migration") pollutes the token set and
can cause false positives.

## Build a keyword coverage checklist

Instead of a large trigger eval set, build a focused list:

1. List 5-8 **target keywords** — words a user would realistically use when they need
   this skill (include synonyms and near-equivalents).
2. List 3-5 **confusable keywords** — words that appear in similar but different skills,
   and that should NOT appear in this description.

Example for `skill-creator`:

| Target keywords | Confusable keywords |
|---|---|
| skill, create, evolve, improve, eval, trigger, description | install, discover, list, manage |

## Optimization steps

1. **Tokenize the current description** mentally or via Python:
   ```python
   import re
   tokens = set(re.findall(r"[a-z0-9]+", description.lower()))
   ```
2. **Check target coverage**: every target keyword (or a close variant) should appear in
   the token set. Missing keywords = potential false negatives.
3. **Check for pollution**: no confusable keywords should appear. Remove or rephrase any
   sentence that introduces them.
4. **Add synonyms sparingly**: if users might say "build" instead of "create", include
   both — but only if genuine.
5. **Test with `SkillManager.search()`** if you have access to a running kcastle instance,
   or simulate by computing overlap manually for 3-5 representative queries.

## Writing pattern

A strong description encodes positive signal only:

```text
<What this skill does> when users ask to <verb A>, <verb B>, or <verb C>,
including <synonym/near phrase>.
```

**Do NOT add negative boundary phrases** like "Do not use for X" — the word "X" becomes a
matchable token and attracts false positives. Instead, ensure the competing skill's own
description covers those keywords better.

## Output expectations

When reporting optimization results, include:

- Previous description (full text).
- New description (full text).
- Added/removed keywords and why.
- Remaining coverage gaps, if any.
