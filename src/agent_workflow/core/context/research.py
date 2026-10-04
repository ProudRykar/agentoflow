from __future__ import annotations

from dataclasses import dataclass

from agent_workflow.core.entities.models.research_contract import (
    ResearchContract,
    ResearchCoverage,
)


@dataclass(slots=True, frozen=True)
class ResearchContext:
    """Read-view over the research subsystem.

    contract = WHAT to investigate (from TaskContract).
    coverage = HOW MUCH is covered (snapshot copy).
    """

    contract: ResearchContract | None
    coverage: ResearchCoverage

    @classmethod
    def snapshot(
        cls,
        contract: ResearchContract | None,
        coverage: ResearchCoverage,
    ) -> ResearchContext:
        return cls(
            contract=contract,
            coverage=ResearchCoverage(
                fetched_urls=set(
                    coverage.fetched_urls,
                ),
                failed_urls=set(
                    coverage.failed_urls,
                ),
                discovered_urls=set(
                    coverage.discovered_urls,
                ),
                max_depth_reached=(
                    coverage.max_depth_reached
                ),
                total_bytes=coverage.total_bytes,
            ),
        )
