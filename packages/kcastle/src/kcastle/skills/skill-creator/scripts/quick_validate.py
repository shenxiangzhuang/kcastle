#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pyyaml>=6.0",
# ]
# ///
"""
Quick validation script for skills - minimal version
"""

import re
import sys
from pathlib import Path
from typing import Any, cast

import yaml

MAX_SKILL_NAME_LENGTH = 64


def validate_skill(skill_path: str) -> tuple[bool, str]:
    """Basic validation of a skill."""
    path = Path(skill_path)

    skill_md = path / "SKILL.md"
    if not skill_md.exists():
        return False, "SKILL.md not found"

    content = skill_md.read_text()
    if not content.startswith("---"):
        return False, "No YAML frontmatter found"

    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return False, "Invalid frontmatter format"

    frontmatter_text = match.group(1)

    try:
        raw = yaml.safe_load(frontmatter_text)
        if not isinstance(raw, dict):
            return False, "Frontmatter must be a YAML dictionary"
    except yaml.YAMLError as e:
        return False, f"Invalid YAML in frontmatter: {e}"

    frontmatter = cast(dict[str, Any], raw)
    allowed_properties = {"name", "description"}

    unexpected_keys = set(frontmatter.keys()) - allowed_properties
    if unexpected_keys:
        allowed = ", ".join(sorted(allowed_properties))
        unexpected = ", ".join(sorted(unexpected_keys))
        return (
            False,
            f"Unexpected key(s) in SKILL.md frontmatter: {unexpected}. "
            f"Allowed properties are: {allowed}",
        )

    if "name" not in frontmatter:
        return False, "Missing 'name' in frontmatter"
    if "description" not in frontmatter:
        return False, "Missing 'description' in frontmatter"

    raw_name: object = frontmatter.get("name", "")
    if not isinstance(raw_name, str):
        return False, f"Name must be a string, got {type(raw_name).__name__}"
    name = raw_name
    name = name.strip()
    if name:
        if not re.match(r"^[a-z0-9-]+$", name):
            return (
                False,
                f"Name '{name}' should be hyphen-case "
                "(lowercase letters, digits, and hyphens only)",
            )
        if name.startswith("-") or name.endswith("-") or "--" in name:
            return (
                False,
                f"Name '{name}' cannot start/end with hyphen or contain consecutive hyphens",
            )
        if len(name) > MAX_SKILL_NAME_LENGTH:
            return (
                False,
                f"Name is too long ({len(name)} characters). "
                f"Maximum is {MAX_SKILL_NAME_LENGTH} characters.",
            )

    raw_desc: object = frontmatter.get("description", "")
    if not isinstance(raw_desc, str):
        return False, f"Description must be a string, got {type(raw_desc).__name__}"
    description = raw_desc
    description = description.strip()
    if description:
        if "<" in description or ">" in description:
            return False, "Description cannot contain angle brackets (< or >)"
        if len(description) > 1024:
            return (
                False,
                f"Description is too long ({len(description)} characters). "
                "Maximum is 1024 characters.",
            )

    return True, "Skill is valid!"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run scripts/quick_validate.py <skill_directory>")
        sys.exit(1)

    valid, message = validate_skill(sys.argv[1])
    print(message)
    sys.exit(0 if valid else 1)
