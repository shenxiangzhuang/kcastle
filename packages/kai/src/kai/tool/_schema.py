"""JSON Schema utilities for Pydantic-based tool parameters.

Converts a Pydantic ``BaseModel`` class into a clean JSON Schema dict
suitable for LLM tool calling APIs. Handles ``$ref`` dereferencing,
``$defs`` removal, and ``title`` stripping.

``deref_json_schema`` is adapted from the kosong library:
  https://github.com/MoonshotAI/kimi-cli
  Copyright 2025 Moonshot AI — Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import copy
from typing import Any, cast

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema

type JsonType = None | int | float | str | bool | list[JsonType] | dict[str, JsonType]
type JsonDict = dict[str, JsonType]


class _NoTitleJsonSchema(GenerateJsonSchema):
    """Custom JSON schema generator that omits titles."""

    def field_title_should_be_set(self, schema: Any) -> bool:  # noqa: ANN401
        return False

    def _update_class_schema(
        self,
        json_schema: dict[str, Any],
        cls: type[Any],
        config: Any,  # noqa: ANN401
    ) -> None:
        super()._update_class_schema(json_schema, cls, config)
        json_schema.pop("title", None)


# ---------------------------------------------------------------------------
# Adapted from kosong (https://github.com/MoonshotAI/kimi-cli)
# Copyright 2025 Moonshot AI — Apache License 2.0
# ---------------------------------------------------------------------------


def deref_json_schema(schema: JsonDict) -> JsonDict:
    """Expand local ``$ref`` entries in a JSON Schema without infinite recursion."""
    # Work on a deep copy so we never mutate the caller's schema.
    full_schema: JsonDict = copy.deepcopy(schema)

    def resolve_pointer(root: JsonDict, pointer: str) -> JsonType:
        """Resolve a JSON Pointer (e.g. ``#/$defs/User``) inside the schema."""
        parts = pointer.lstrip("#/").split("/")
        current: JsonType = root
        try:
            for part in parts:
                if isinstance(current, dict):
                    current = current[part]
                else:
                    raise ValueError
            return current
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Unable to resolve reference path: {pointer}") from None

    def traverse(node: JsonType, root: JsonDict) -> JsonType:
        """Recursively traverse every node to inline local references."""
        if isinstance(node, dict):
            # Replace local ``$ref`` entries with their referenced payload.
            if "$ref" in node and isinstance(node["$ref"], str):
                ref_path = node["$ref"]
                if ref_path.startswith("#"):
                    # Resolve the local reference target.
                    target = resolve_pointer(root, ref_path)
                    # Recursively inline the target in case it contains more refs.
                    ref = traverse(target, root)
                    if not isinstance(ref, dict):
                        msg = "Local $ref must resolve to a JSON object"
                        raise TypeError(msg)
                    node.pop("$ref")
                    node.update(ref)
                    return node
                else:
                    # Ignore remote references such as http://...
                    return node

            # Traverse the remaining mapping entries.
            return {k: traverse(v, root) for k, v in node.items()}

        elif isinstance(node, list):
            # Traverse list members (e.g. allOf, oneOf, items).
            return [traverse(item, root) for item in node]

        else:
            return node

    # Remove definition buckets to keep the resolved schema minimal.
    resolved = cast(JsonDict, traverse(full_schema, full_schema))

    # Remove $defs / definitions buckets.
    resolved.pop("$defs", None)
    resolved.pop("definitions", None)

    return resolved


def params_to_json_schema(params_cls: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic BaseModel class into a clean JSON Schema dict.

    1. Generates the schema using a custom generator that strips titles.
    2. Dereferences any ``$ref`` / ``$defs`` for a flat result.

    Args:
        params_cls: The Pydantic model class to convert.

    Returns:
        A JSON Schema dict ready for LLM tool calling APIs.
    """
    raw = params_cls.model_json_schema(schema_generator=_NoTitleJsonSchema)
    return cast(dict[str, Any], deref_json_schema(cast(JsonDict, raw)))
