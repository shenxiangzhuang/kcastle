"""Provider-agnostic error type for kai."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum


class ErrorKind(StrEnum):
    """Stable error categories for caller decisions."""

    CONNECTION = "connection"
    TIMEOUT = "timeout"
    STATUS = "status"
    EMPTY_RESPONSE = "empty_response"
    PROVIDER = "provider"


@dataclass(frozen=True)
class KaiError(Exception):
    """Single error type for all kai errors."""

    kind: ErrorKind
    message: str
    cause: Exception | None = None

    def __str__(self) -> str:
        return f"[{self.kind}] {self.message}"

    def with_cause(self, cause: Exception) -> KaiError:
        return replace(self, cause=cause)
