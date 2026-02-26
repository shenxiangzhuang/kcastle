"""Skill resolution — search and rank skills for a given query.

PoC implementation uses keyword matching against skill name, description,
and tags.  BM25 or embedding-based ranking can be added later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from kcastle.skills.schema import SkillMeta

_log = logging.getLogger("kcastle.skills")


@dataclass(frozen=True, slots=True)
class SkillMatch:
    """A skill matched against a query with a relevance score."""

    meta: SkillMeta
    score: float


class SkillResolver:
    """Resolves user queries to ranked skill candidates.

    PoC: simple keyword overlap scoring.
    """

    def __init__(self, *, top_k: int = 3) -> None:
        self._top_k = top_k
        self._skills: list[SkillMeta] = []

    def index(self, skills: list[SkillMeta]) -> None:
        """(Re)build the searchable index from discovered skills."""
        self._skills = list(skills)
        _log.debug("Indexed %d skills for resolution", len(self._skills))

    def search(self, query: str) -> list[SkillMatch]:
        """Search for skills matching the query.  Returns top-K results."""
        if not query.strip():
            return []

        query_tokens = set(query.lower().split())
        scored: list[SkillMatch] = []

        for meta in self._skills:
            score = self._score(meta, query_tokens)
            if score > 0:
                scored.append(SkillMatch(meta=meta, score=score))

        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[: self._top_k]

    def all_skills(self) -> list[SkillMeta]:
        """Return all indexed skills."""
        return list(self._skills)

    @staticmethod
    def _score(meta: SkillMeta, query_tokens: set[str]) -> float:
        """Score a skill against query tokens (keyword overlap)."""
        searchable = " ".join(
            [
                meta.name.lower(),
                meta.description.lower(),
                " ".join(t.lower() for t in meta.tags),
            ]
        )
        searchable_tokens = set(searchable.split())

        overlap = query_tokens & searchable_tokens
        if not overlap:
            return 0.0

        # Normalize by query length
        return len(overlap) / len(query_tokens)
