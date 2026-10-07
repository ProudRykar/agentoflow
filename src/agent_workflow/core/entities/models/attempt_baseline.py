"""The research state a run started from.

Kept so a retry can return to it. A run's own progress is its own; a
retry that inherits the discarded attempt's evidence satisfies the
research contract with material the user rejected, and there is then
no way to tell a contract that was met from one that was never
attempted.

Deliberately narrow. Only research is rolled back: the plan's completed
steps survive, because a retry of the wording should not also discard
the research that made the first answer correct.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class AttemptBaseline:
    """Research progress as it stood when an attempt began."""

    evidence_ids: tuple[str, ...]

    fetched_urls: frozenset[str]
    failed_urls: frozenset[str]
    discovered_urls: frozenset[str]

    max_depth_reached: int
    total_bytes: int

    research_completed: bool
