from __future__ import annotations

from dataclasses import dataclass

from agent_workflow.core.entities.models.agent_phase import AgentPhase
from agent_workflow.core.entities.models.task_contract import (
    TaskContract,
    TaskProgress,
)


@dataclass(slots=True, frozen=True)
class CompletionGateResult:
    """
    Result of checking whether the runtime may complete the task.
    """

    allowed: bool
    next_phase: AgentPhase | None
    missing: tuple[str, ...]
    reason: str


class CompletionGate:
    """
    Deterministic completion policy.

    The gate does not inspect model output and does not infer progress
    from tool names. It only evaluates the explicit TaskContract against
    the explicit TaskProgress maintained by the runtime.
    """

    def check(
        self,
        contract: TaskContract,
        progress: TaskProgress,
    ) -> CompletionGateResult:
        missing: list[str] = []
        next_phase: AgentPhase | None = None

        if (
            contract.requires_research
            and not progress.research_completed
        ):
            missing.append("research")

            if next_phase is None:
                next_phase = AgentPhase.RESEARCH

        if (
            contract.requires_verification
            and not progress.verification_completed
        ):
            missing.append("verification")

            if next_phase is None:
                next_phase = AgentPhase.VERIFICATION

        if (
            contract.requires_reflection
            and not progress.reflection_completed
        ):
            missing.append("reflection")

            if next_phase is None:
                next_phase = AgentPhase.REFLECTION

        if missing:
            return CompletionGateResult(
                allowed=False,
                next_phase=next_phase,
                missing=tuple(missing),
                reason=(
                    "completion blocked: missing "
                    + ", ".join(missing)
                ),
            )

        return CompletionGateResult(
            allowed=True,
            next_phase=None,
            missing=(),
            reason="completion gate satisfied",
        )