"""Runtime environment made available to executable tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Env:
    cwd: Path
