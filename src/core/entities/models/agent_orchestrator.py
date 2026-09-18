from __future__ import annotations

from dataclasses import dataclass

from core.entities.models.agent_phase import AgentPhase
from core.entities.models.agent_state import AgentState
from core.entities.models.completion_gate import (
    CompletionGate,
    CompletionGateResult,
)
from core.entities.models.research_contract import (
    ResearchResult,
)
from core.entities.models.task_contract import (
    TaskContract,
    TaskProgress,
)


@dataclass(slots=True, frozen=True)
class PhaseTransition:
    previous_phase: AgentPhase
    phase: AgentPhase
    reason: str

    def __iter__(self):
        yield self.previous_phase
        yield self.phase
        yield self.reason

    def __getitem__(self, index: int):
        values = (
            self.previous_phase,
            self.phase,
            self.reason,
        )

        return values[index]

    def __len__(self) -> int:
        return 3

    def __eq__(
        self,
        other: object,
    ) -> bool:
        if isinstance(
            other,
            PhaseTransition,
        ):
            return (
                self.previous_phase
                == other.previous_phase
                and self.phase
                == other.phase
                and self.reason
                == other.reason
            )

        if isinstance(
            other,
            tuple,
        ):
            return (
                len(other) == 3
                and other[0]
                == self.previous_phase
                and other[1]
                == self.phase
                and other[2]
                == self.reason
            )

        return NotImplemented


class AgentOrchestrator:
    """
    Runtime state machine for one agent run.

    Observable execution phases belong here.

    Semantic task completion is delegated to
    TaskContract + TaskProgress + CompletionGate.
    """

    _ALLOWED_TRANSITIONS: dict[
        AgentPhase,
        frozenset[AgentPhase],
    ] = {
        AgentPhase.IDLE: frozenset({
            AgentPhase.PLANNING,
            AgentPhase.EXECUTION,
            AgentPhase.BLOCKED,
        }),

        AgentPhase.PLANNING: frozenset({
            AgentPhase.RESEARCH,
            AgentPhase.EXECUTION,
            AgentPhase.DEBUGGING,
            AgentPhase.VERIFICATION,
            AgentPhase.REFLECTION,
            AgentPhase.SYNTHESIS,
            AgentPhase.WAITING_APPROVAL,
            AgentPhase.BLOCKED,
        }),

        AgentPhase.RESEARCH: frozenset({
            AgentPhase.RESEARCH,
            AgentPhase.EXECUTION,
            AgentPhase.DEBUGGING,
            AgentPhase.VERIFICATION,
            AgentPhase.REFLECTION,
            AgentPhase.SYNTHESIS,
            AgentPhase.WAITING_APPROVAL,
            AgentPhase.BLOCKED,
        }),

        AgentPhase.EXECUTION: frozenset({
            AgentPhase.RESEARCH,
            AgentPhase.EXECUTION,
            AgentPhase.DEBUGGING,
            AgentPhase.VERIFICATION,
            AgentPhase.REFLECTION,
            AgentPhase.SYNTHESIS,
            AgentPhase.WAITING_APPROVAL,
            AgentPhase.BLOCKED,
        }),

        AgentPhase.DEBUGGING: frozenset({
            AgentPhase.RESEARCH,
            AgentPhase.EXECUTION,
            AgentPhase.DEBUGGING,
            AgentPhase.VERIFICATION,
            AgentPhase.REFLECTION,
            AgentPhase.SYNTHESIS,
            AgentPhase.WAITING_APPROVAL,
            AgentPhase.BLOCKED,
        }),

        AgentPhase.VERIFICATION: frozenset({
            AgentPhase.RESEARCH,
            AgentPhase.EXECUTION,
            AgentPhase.DEBUGGING,
            AgentPhase.VERIFICATION,
            AgentPhase.REFLECTION,
            AgentPhase.SYNTHESIS,
            AgentPhase.WAITING_APPROVAL,
            AgentPhase.BLOCKED,
        }),

        AgentPhase.REFLECTION: frozenset({
            AgentPhase.RESEARCH,
            AgentPhase.EXECUTION,
            AgentPhase.DEBUGGING,
            AgentPhase.VERIFICATION,
            AgentPhase.REFLECTION,
            AgentPhase.SYNTHESIS,
            AgentPhase.WAITING_APPROVAL,
            AgentPhase.BLOCKED,
        }),

        AgentPhase.SYNTHESIS: frozenset({
            AgentPhase.RESEARCH,
            AgentPhase.VERIFICATION,
            AgentPhase.REFLECTION,
            AgentPhase.COMPLETED,
            AgentPhase.BLOCKED,
        }),

        AgentPhase.WAITING_APPROVAL: frozenset({
            AgentPhase.RESEARCH,
            AgentPhase.EXECUTION,
            AgentPhase.DEBUGGING,
            AgentPhase.BLOCKED,
        }),

        AgentPhase.BLOCKED: frozenset({
            AgentPhase.REFLECTION,
            AgentPhase.RESEARCH,
            AgentPhase.EXECUTION,
        }),

        AgentPhase.COMPLETED: frozenset(),
    }

    _RESEARCH_TOOLS = frozenset({
        "web_fetch",
        "web_crawl",
    })

    def __init__(
        self,
        state: AgentState | None = None,
    ) -> None:
        self.state = (
            state
            or AgentState()
        )

        self._task_contract = (
            TaskContract()
        )

        self._task_progress = (
            TaskProgress()
        )

        self._completion_gate = (
            CompletionGate()
        )

    # ------------------------------------------------------------------
    # Task contract / progress
    # ------------------------------------------------------------------

    @property
    def task_contract(self) -> TaskContract:
        return self._task_contract

    @property
    def task_progress(self) -> TaskProgress:
        return self._task_progress

    def set_task_contract(
        self,
        contract: TaskContract,
    ) -> None:
        self._task_contract = contract

        self._task_progress.reset()

    def mark_research_completed(self) -> None:
        self._task_progress.mark_research_completed()

    def mark_verification_completed(self) -> None:
        self._task_progress.mark_verification_completed()

    def mark_reflection_completed(self) -> None:
        self._task_progress.mark_reflection_completed()

    def record_research_result(
        self,
        result: ResearchResult,
    ) -> bool:
        """
        Consume trusted research evidence.

        Research completion is determined from the result contents
        and ResearchContract, never from the name of the tool that
        produced the result.
        """

        self._task_progress.record_research_result(
            result
        )

        if (
            not self._task_contract.requires_research
        ):
            return False

        research_contract = (
            self._task_contract.research
        )

        if research_contract is None:
            # Backward-compatible legacy meaning:
            # some real research evidence is enough.
            satisfied = bool(
                self._task_progress
                .research_coverage
                .fetched_urls
            )
        else:
            satisfied = (
                research_contract.is_satisfied(
                    self._task_progress
                    .research_coverage
                )
            )

        if satisfied:
            self.mark_research_completed()

        return satisfied

    def check_completion(
        self,
    ) -> CompletionGateResult:
        return self._completion_gate.check(
            contract=self._task_contract,
            progress=self._task_progress,
        )

    # ------------------------------------------------------------------
    # Phase state
    # ------------------------------------------------------------------

    def reset_for_run(self) -> None:
        self.state.reset()

    def transition(
        self,
        target: AgentPhase,
        *,
        reason: str = "",
    ) -> PhaseTransition | None:
        current = self.state.phase

        if current is target:
            self.state.phase_reason = reason
            return None

        allowed = self._ALLOWED_TRANSITIONS[
            current
        ]

        if target not in allowed:
            raise RuntimeError(
                "Invalid agent phase transition: "
                f"{current.value} -> "
                f"{target.value}"
            )

        self.state.phase = target

        self.state.phase_reason = reason

        return PhaseTransition(
            previous_phase=current,
            phase=target,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Runtime events
    # ------------------------------------------------------------------

    def on_agent_started(
        self,
    ) -> PhaseTransition | None:
        self.reset_for_run()

        self._task_progress.reset()

        self.state.started = True

        self.state.finished = False

        return self.transition(
            AgentPhase.PLANNING,
            reason="agent run started",
        )

    def on_llm_requested(
        self,
        iteration: int,
    ) -> PhaseTransition | None:
        self.state.iteration = iteration

        if (
            self.state.phase
            is AgentPhase.IDLE
        ):
            return self.transition(
                AgentPhase.PLANNING,
                reason="planning next step",
            )

        return None

    def on_tool_started(
        self,
        tool_name: str,
    ) -> PhaseTransition | None:
        self.state.tool_calls += 1

        self.state.last_tool_name = tool_name

        self.state.last_tool_succeeded = None

        self.state.last_error = None

        if (
            tool_name
            in self._RESEARCH_TOOLS
        ):
            return self.transition(
                AgentPhase.RESEARCH,
                reason=(
                    "research tool started: "
                    f"{tool_name}"
                ),
            )

        return self.transition(
            AgentPhase.EXECUTION,
            reason=(
                "tool started: "
                f"{tool_name}"
            ),
        )

    def on_tool_finished(
        self,
        *,
        error_code: str | None,
        error_message: str | None,
    ) -> PhaseTransition | None:
        failed = error_code is not None

        self.state.last_tool_succeeded = (
            not failed
        )

        self.state.last_error = (
            error_message
        )

        if failed:
            return self.transition(
                AgentPhase.DEBUGGING,
                reason=(
                    "tool failed: "
                    f"{error_code}"
                ),
            )

        if (
            self.state.last_tool_name
            in self._RESEARCH_TOOLS
        ):
            return self.transition(
                AgentPhase.RESEARCH,
                reason=(
                    "research tool completed: "
                    f"{self.state.last_tool_name}"
                ),
            )

        return self.transition(
            AgentPhase.EXECUTION,
            reason=(
                "tool completed: "
                f"{self.state.last_tool_name}"
            ),
        )

    # ------------------------------------------------------------------
    # Explicit phase entry
    # ------------------------------------------------------------------

    def begin_verification(
        self,
        reason: str = "verification started",
    ) -> PhaseTransition | None:
        return self.transition(
            AgentPhase.VERIFICATION,
            reason=reason,
        )

    def begin_reflection(
        self,
        reason: str = "reflection required",
    ) -> PhaseTransition | None:
        return self.transition(
            AgentPhase.REFLECTION,
            reason=reason,
        )

    def begin_synthesis(
        self,
        reason: str = (
            "candidate final response produced"
        ),
    ) -> PhaseTransition | None:
        return self.transition(
            AgentPhase.SYNTHESIS,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def complete(self) -> PhaseTransition:
        completion = self.check_completion()

        if not completion.allowed:
            if completion.next_phase is None:
                raise RuntimeError(
                    "Completion gate rejected completion "
                    "without a next phase"
                )

            return self.transition(
                completion.next_phase,
                reason=completion.reason,
            )

        transition = self.transition(
            AgentPhase.COMPLETED,
            reason="agent run completed",
        )

        if transition is None:
            raise RuntimeError(
                "Agent is already in COMPLETED phase"
            )

        self.state.finished = True

        return transition

    def block(
        self,
        reason: str,
    ) -> PhaseTransition | None:
        return self.transition(
            AgentPhase.BLOCKED,
            reason=reason,
        )