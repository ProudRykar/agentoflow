from __future__ import annotations

from dataclasses import dataclass

from agent_workflow.core.context.evidence import (
    EvidenceReceipt,
    EvidenceStore,
)
from agent_workflow.core.context.execution_context import ExecutionContext
from agent_workflow.core.context.research import ResearchContext
from agent_workflow.core.context.task_anchor import TaskAnchor
from agent_workflow.core.context.task_state import TaskState
from agent_workflow.core.entities.models.agent_phase import AgentPhase
from agent_workflow.core.entities.models.agent_plan import AgentPlan
from agent_workflow.core.entities.models.agent_state import AgentState
from agent_workflow.core.entities.models.completion_gate import (
    CompletionGate,
    CompletionGateResult,
)
from agent_workflow.core.entities.models.plan_step import (
    PlanStep,
    PlanStepStatus,
)
from agent_workflow.core.entities.models.planner import Planner
from agent_workflow.core.entities.models.research_contract import ResearchResult
from agent_workflow.core.entities.models.task_contract import (
    TaskContract,
    TaskProgress,
)
from agent_workflow.core.entities.models.task_plan import TaskPlan


@dataclass(slots=True, frozen=True)
class PhaseTransition:
    previous_phase: AgentPhase
    phase: AgentPhase
    reason: str

    def __iter__(self):
        yield self.previous_phase
        yield self.phase
        yield self.reason

    def __getitem__(
        self,
        index: int,
    ):
        return (
            self.previous_phase,
            self.phase,
            self.reason,
        )[index]

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
    Runtime orchestrator.

    Responsibilities:

    - own AgentState;
    - own TaskContract;
    - own TaskProgress;
    - own the runtime AgentPlan;
    - coordinate phase transitions;
    - coordinate semantic progress;
    - coordinate CompletionGate;
    - revise the execution plan when completion is rejected.

    Planner is responsible for constructing TaskPlan and AgentPlan.

    The preferred lifecycle is:

        Planner
            ↓
        TaskPlan
            ↓
        TaskContract
            ↓
        AgentOrchestrator
            ↓
        AgentPlan
            ↓
        Runtime

    TaskPlan is therefore supplied to the orchestrator by the caller.

    The orchestrator keeps a compatibility fallback: if no TaskPlan has
    been supplied before initialization and a prompt is available, the
    orchestrator may create the TaskPlan itself.

    Conversation lifetime and execution-run lifetime are different.

    A conversation can contain many execution runs:

        conversation
            ├── run #1
            │     └── COMPLETED
            │
            ├── run #2
            │     └── COMPLETED
            │
            └── run #3
                  └── RUNNING

    A new user message starts a new execution run.

    on_agent_resumed() is reserved for resuming an unfinished run.
    """

    _ALLOWED_TRANSITIONS: dict[
        AgentPhase,
        frozenset[AgentPhase],
    ] = {
        AgentPhase.IDLE: frozenset({
            AgentPhase.PLANNING,
            AgentPhase.RESEARCH,
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
            AgentPhase.EXECUTION,
            AgentPhase.DEBUGGING,
            AgentPhase.VERIFICATION,
            AgentPhase.REFLECTION,
            AgentPhase.SYNTHESIS,
            AgentPhase.BLOCKED,
            AgentPhase.COMPLETED,
        }),

        AgentPhase.WAITING_APPROVAL: frozenset({
            AgentPhase.RESEARCH,
            AgentPhase.EXECUTION,
            AgentPhase.DEBUGGING,
            AgentPhase.VERIFICATION,
            AgentPhase.REFLECTION,
            AgentPhase.SYNTHESIS,
            AgentPhase.BLOCKED,
        }),

        AgentPhase.BLOCKED: frozenset({
            AgentPhase.PLANNING,
            AgentPhase.RESEARCH,
            AgentPhase.EXECUTION,
            AgentPhase.DEBUGGING,
            AgentPhase.VERIFICATION,
            AgentPhase.REFLECTION,
        }),

        AgentPhase.COMPLETED: frozenset(),
    }

    def __init__(
        self,
        state: AgentState | None = None,
        *,
        planner: Planner | None = None,
        completion_gate: CompletionGate | None = None,
    ) -> None:
        self.state = (
            state
            if state is not None
            else AgentState()
        )

        self._planner = (
            planner
            if planner is not None
            else Planner()
        )

        self._completion_gate = (
            completion_gate
            if completion_gate is not None
            else CompletionGate()
        )

        self._task_contract = TaskContract()
        self._task_progress = TaskProgress()

        # Append-only task-owned storage for research evidence.
        # Survives run resets; page bodies live here, never in
        # the conversation.
        self._evidence_store = EvidenceStore()

        # Immutable source of truth for WHAT the user asked.
        # Created once per task, never mutated afterwards.
        self._task_anchor: TaskAnchor | None = None

        # Run identity for ExecutionContext snapshots.
        # Synced from Agent before each run.
        self._run_id: str = ""

        # Prepared immutable task description produced by Planner.
        self._task_plan: TaskPlan | None = None

        # Whether the currently stored TaskPlan belongs to the next
        # execution run and should be consumed by initialize_plan().
        self._task_plan_prepared = False

        # Mutable runtime execution plan.
        self._plan: AgentPlan | None = None

    # ==================================================================
    # Public state
    # ==================================================================

    @property
    def task_contract(self) -> TaskContract:
        return self._task_contract

    @property
    def task_progress(self) -> TaskProgress:
        return self._task_progress

    @property
    def task_plan(self) -> TaskPlan | None:
        return self._task_plan

    @property
    def plan(self) -> AgentPlan | None:
        return self._plan

    @property
    def current_step(self) -> PlanStep | None:
        if self._plan is None:
            return None

        return self._plan.current_step()

    @property
    def task_anchor(self) -> TaskAnchor | None:
        return self._task_anchor

    @property
    def has_prepared_task(self) -> bool:
        return (
            self._task_plan_prepared
            and self._task_plan is not None
        )

    @property
    def evidence_store(self) -> EvidenceStore:
        return self._evidence_store

    @property
    def research_context(self) -> ResearchContext:
        """
        Read-view over the research subsystem: contract,
        coverage snapshot, and stored evidence ids.
        """

        return ResearchContext.snapshot(
            contract=self._task_contract.research,
            coverage=(
                self._task_progress.research_coverage
            ),
            evidence_ids=self._evidence_store.ids,
        )

    def store_research_evidence(
        self,
        result: ResearchResult,
    ) -> EvidenceReceipt:
        """
        Register fetched pages in the EvidenceStore and merge
        coverage into TaskProgress.

        Returns a receipt for the conversation. The full page
        bodies stay in the store and never enter the dialogue.
        """

        evidence_ids: list[str] = []

        for page in result.pages:
            evidence = self._evidence_store.append_page(
                page,
                result.root_url,
            )
            evidence_ids.append(evidence.evidence_id)

        satisfied = self.record_research_result(result)

        return EvidenceReceipt(
            evidence_ids=tuple(evidence_ids),
            page_count=len(result.pages),
            total_bytes=result.total_bytes,
            max_depth_reached=result.max_depth_reached,
            satisfied=satisfied,
        )

    @property
    def task_state(self) -> TaskState:
        """
        Immutable snapshot of WHAT the task is.

        Read-view over the mutable runtime fields. Later progress
        never mutates an already taken snapshot.
        """

        if self._task_anchor is None:
            raise RuntimeError(
                "Cannot build TaskState without a TaskAnchor",
            )

        if self._task_plan is None:
            raise RuntimeError(
                "Cannot build TaskState without a TaskPlan",
            )

        return TaskState.snapshot(
            anchor=self._task_anchor,
            plan=self._task_plan,
            contract=self._task_contract,
            coverage=(
                self._task_progress.research_coverage
            ),
        )

    @property
    def execution_context(self) -> ExecutionContext:
        """
        Immutable snapshot of the current execution run.

        Read-view over AgentState and the runtime plan. The
        ContextAssembler must depend on this snapshot, never on
        the orchestrator's internal fields.
        """

        current = self.current_step

        blocked_reason: str | None = None

        if (
            self.state.phase is AgentPhase.BLOCKED
        ):
            blocked_reason = (
                self.state.last_error
            )

        return ExecutionContext(
            run_id=self._run_id,
            iteration=self.state.iteration,
            phase=self.state.phase,
            current_step_id=(
                current.id
                if current is not None
                else None
            ),
            tool_calls_used=self.state.tool_calls,
            blocked_reason=blocked_reason,
        )

    def set_run_id(
        self,
        run_id: str,
    ) -> None:
        self._run_id = run_id

    def ensure_task_anchor(
        self,
        prompt: str,
        task_plan: TaskPlan | None = None,
        *,
        force_new: bool = False,
    ) -> TaskAnchor:
        """
        Return the TaskAnchor for the current task.

        Rules:

        - prepared task exists and no conflicting plan given
          and not force_new -> reuse the existing anchor;
        - otherwise -> create a new anchor (new task).

        The anchor objective comes from the task plan when one
        is available, otherwise from a fallback Planner pass
        over the prompt.
        """

        if not isinstance(prompt, str) or not prompt:
            raise ValueError(
                "prompt must be a non-empty string",
            )

        if (
            not force_new
            and self._task_anchor is not None
            and self._task_plan_prepared
            and self._task_plan is not None
            and (
                task_plan is None
                or task_plan == self._task_plan
            )
        ):
            return self._task_anchor

        reference_plan = (
            task_plan
            if task_plan is not None
            else self._task_plan
        )

        if reference_plan is not None:
            objective = reference_plan.objective
        else:
            objective = self._planner.plan(
                prompt,
            ).objective

        self._task_anchor = TaskAnchor.create(
            original_prompt=prompt,
            objective=objective,
        )

        return self._task_anchor

    # ==================================================================
    # Task preparation
    # ==================================================================

    def set_task_plan(
        self,
        task_plan: TaskPlan,
    ) -> None:
        """
        Supply the TaskPlan produced by the Planner.

        This is the preferred way to initialize a new task.

        The orchestrator does not call Planner.plan() again when a prepared
        TaskPlan exists.
        """

        if not isinstance(
            task_plan,
            TaskPlan,
        ):
            raise TypeError(
                "task_plan must be a TaskPlan",
            )

        self._task_plan = task_plan
        self._task_plan_prepared = True

        # A new TaskPlan always requires a fresh runtime plan.
        self._plan = None

    def set_task_contract(
        self,
        contract: TaskContract,
    ) -> None:
        """
        Set requirements for the current execution run.

        This method intentionally does NOT clear _task_plan.

        The TaskPlan and TaskContract are separate artifacts and may have
        been produced by the same Planner invocation.
        """

        if not isinstance(
            contract,
            TaskContract,
        ):
            raise TypeError(
                "contract must be a TaskContract",
            )

        self._task_contract = contract

        self._task_progress.reset()

        # Contract changes invalidate the executable AgentPlan.
        self._plan = None

    def prepare_task(
        self,
        task_plan: TaskPlan,
        contract: TaskContract,
        *,
        prompt: str | None = None,
        anchor: TaskAnchor | None = None,
    ) -> None:
        """
        Preferred atomic API for preparing a new execution run.

        Both TaskPlan and TaskContract are installed before the runtime
        starts.

        When an explicit anchor is supplied, it is installed as-is.
        When a prompt is supplied, a fresh anchor is created for the
        new task. Otherwise the anchor is invalidated and will be
        created lazily by ensure_task_anchor() at run() time.
        """

        self.set_task_plan(
            task_plan,
        )

        self.set_task_contract(
            contract,
        )

        if anchor is not None:
            if not isinstance(
                anchor,
                TaskAnchor,
            ):
                raise TypeError(
                    "anchor must be a TaskAnchor",
                )

            self._task_anchor = anchor
        elif prompt is not None:
            self.ensure_task_anchor(
                prompt,
                task_plan=task_plan,
                force_new=True,
            )
        else:
            self._task_anchor = None

    # ==================================================================
    # Semantic progress
    # ==================================================================

    def mark_research_completed(self) -> None:
        self._task_progress.mark_research_completed()

        step = self.current_step

        if (
            step is not None
            and step.phase is AgentPhase.RESEARCH
        ):
            step.complete(
                "research requirements satisfied",
            )

    def mark_verification_completed(self) -> None:
        self._task_progress.mark_verification_completed()

        step = self.current_step

        if (
            step is not None
            and step.phase is AgentPhase.VERIFICATION
        ):
            step.complete(
                "verification completed",
            )

    def mark_reflection_completed(self) -> None:
        self._task_progress.mark_reflection_completed()

        step = self.current_step

        if (
            step is not None
            and step.phase is AgentPhase.REFLECTION
        ):
            step.complete(
                "reflection completed",
            )

    def record_research_result(
        self,
        result: ResearchResult,
    ) -> bool:
        self._task_progress.record_research_result(
            result,
        )

        if not self._task_contract.requires_research:
            return False

        research_contract = (
            self._task_contract.research
        )

        if research_contract is None:
            satisfied = bool(
                self._task_progress
                .research_coverage
                .fetched_urls
            )
        else:
            satisfied = (
                research_contract.is_satisfied(
                    self._task_progress
                    .research_coverage,
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

    # ==================================================================
    # Planning
    # ==================================================================

    def initialize_plan(
        self,
        prompt: str | None = None,
    ) -> AgentPlan:
        """
        Build the mutable runtime AgentPlan.

        Preferred path:

            existing TaskPlan supplied by caller
                ↓
            build_execution_plan()

        Compatibility path:

            no TaskPlan
                ↓
            Planner.plan(prompt)
                ↓
            build_execution_plan()

        The preferred path never invokes Planner.plan() twice.
        """

        if self._plan is not None:
            return self._plan

        # --------------------------------------------------------------
        # Preferred path: consume the already prepared TaskPlan.
        # --------------------------------------------------------------

        if (
            self._task_plan is None
            or not self._task_plan_prepared
        ):
            if prompt is None:
                raise RuntimeError(
                    "Cannot initialize execution plan without "
                    "a prepared TaskPlan or prompt",
                )

            self._task_plan = self._planner.plan(
                prompt,
            )

            self._task_plan_prepared = True

        self._plan = (
            self._planner.build_execution_plan(
                self._task_plan,
                self._task_contract,
            )
        )

        self._activate_first_pending_step()

        return self._plan

    def _activate_first_pending_step(
        self,
    ) -> PlanStep | None:
        if self._plan is None:
            return None

        current = self._plan.current_step()

        if current is not None:
            if (
                current.status
                is PlanStepStatus.PENDING
            ):
                current.start()

            return current

        return self._plan.activate_next()

    def replan(
        self,
        phase: AgentPhase,
        *,
        reason: str,
    ) -> PhaseTransition | None:
        if self._plan is None:
            return self.transition(
                phase,
                reason=reason,
            )

        current = self.current_step

        if (
            current is not None
            and current.phase is AgentPhase.SYNTHESIS
            and current.status is PlanStepStatus.ACTIVE
        ):
            current.status = (
                PlanStepStatus.PENDING
            )

            current.result = None
            current.error = None

        step = (
            self._planner.revise_for_missing_phase(
                self._plan,
                phase,
                reason,
            )
        )

        self._plan.current_step_id = (
            step.id
        )

        if step.status in {
            PlanStepStatus.PENDING,
            PlanStepStatus.FAILED,
        }:
            step.start()

        return self.transition(
            step.phase,
            reason=(
                f"plan revised: {reason}"
            ),
        )

    def advance_plan(
        self,
        *,
        reason: str,
    ) -> PhaseTransition | None:
        if self._plan is None:
            return None

        current = self.current_step

        if (
            current is not None
            and current.status
            is not PlanStepStatus.COMPLETED
        ):
            current.complete()

        next_step = (
            self._plan.activate_next()
        )

        if next_step is None:
            return None

        return self.transition(
            next_step.phase,
            reason=reason,
        )

    def retry_current_step(
        self,
        *,
        reason: str,
    ) -> PhaseTransition | None:
        if self._plan is None:
            return None

        step = self.current_step

        if step is None:
            return None

        step.start()

        return self.transition(
            step.phase,
            reason=reason,
        )

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def reset_for_run(self) -> None:
        """
        Reset execution state for a NEW run.

        Important:

        - AgentState is reset;
        - TaskProgress is reset;
        - AgentPlan is rebuilt;
        - prepared TaskPlan is preserved;
        - TaskAnchor is preserved;
        - EvidenceStore is preserved.

        The TaskPlan is preserved because the caller may have already
        supplied the result of Planner.plan() before Agent.run() or
        Agent.continue_run(). The TaskAnchor belongs to the task,
        not to the run, so it survives run resets and is replaced
        only when a new task begins.
        """

        self.state.reset()

        self._task_progress.reset()

        self._plan = None

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

        allowed = self._ALLOWED_TRANSITIONS.get(
            current,
            frozenset(),
        )

        if target not in allowed:
            raise RuntimeError(
                "Invalid agent phase transition: "
                f"{current.value} -> {target.value}",
            )

        self.state.phase = target
        self.state.phase_reason = reason

        return PhaseTransition(
            previous_phase=current,
            phase=target,
            reason=reason,
        )

    def on_agent_started(
        self,
        prompt: str | None = None,
        *,
        run_id: str = "",
    ) -> PhaseTransition | None:
        """
        Start a NEW execution run.

        The caller may already have prepared a TaskPlan using Planner.

        In that case:

            set_task_plan(...)
                ↓
            set_task_contract(...)
                ↓
            on_agent_started(...)
                ↓
            initialize_plan()

        and Planner.plan() is NOT called again.

        When a prompt is supplied and no anchor exists yet for the
        prepared task, a TaskAnchor is ensured here. An already
        prepared anchor is reused, never recreated.
        """

        if run_id:
            self._run_id = run_id

        if prompt is not None:
            self.ensure_task_anchor(prompt)

        self.reset_for_run()

        self.state.started = True
        self.state.finished = False

        planning_transition = self.transition(
            AgentPhase.PLANNING,
            reason="agent run started",
        )

        self.initialize_plan(
            prompt,
        )

        current = self.current_step

        if current is None:
            return planning_transition

        transition = self.transition(
            current.phase,
            reason=(
                "initial plan step activated: "
                f"{current.id}"
            ),
        )

        return (
            transition
            or planning_transition
        )

    def on_agent_resumed(
        self,
    ) -> PhaseTransition | None:
        """
        Resume an EXISTING unfinished run.

        Existing:

            TaskPlan
            TaskContract
            TaskProgress
            AgentPlan

        are preserved.
        """

        if self.state.finished:
            raise RuntimeError(
                "Cannot resume a completed agent run",
            )

        if (
            self.state.phase
            is AgentPhase.COMPLETED
        ):
            raise RuntimeError(
                "Cannot resume a completed agent run",
            )

        if self._plan is None:
            raise RuntimeError(
                "Cannot resume agent run without "
                "an existing execution plan",
            )

        self.state.started = True
        self.state.finished = False

        current = self.current_step

        if current is None:
            return None

        return self.transition(
            current.phase,
            reason="agent run resumed",
        )

    # ==================================================================
    # LLM
    # ==================================================================

    def on_llm_requested(
        self,
        iteration: int,
    ) -> PhaseTransition | None:
        self.state.iteration = iteration

        current = self.current_step

        if current is not None:
            if (
                self.state.phase
                is AgentPhase.DEBUGGING
                and current.phase
                is not AgentPhase.DEBUGGING
            ):
                return self.transition(
                    current.phase,
                    reason=(
                        "resuming current plan step "
                        "after debugging"
                    ),
                )

            if (
                self.state.phase
                is AgentPhase.PLANNING
            ):
                return self.transition(
                    current.phase,
                    reason=(
                        "executing current plan step: "
                        f"{current.id}"
                    ),
                )

            if (
                self.state.phase
                is AgentPhase.IDLE
            ):
                return self.transition(
                    current.phase,
                    reason=(
                        "executing current plan step: "
                        f"{current.id}"
                    ),
                )

        return None

    # ==================================================================
    # Tools
    # ==================================================================

    def on_tool_started(
        self,
        tool_name: str,
    ) -> PhaseTransition | None:
        self.state.tool_calls += 1
        self.state.last_tool_name = tool_name
        self.state.last_tool_succeeded = None
        self.state.last_error = None

        current = self.current_step

        if current is None:
            return self.transition(
                AgentPhase.EXECUTION,
                reason=(
                    "tool started without active "
                    f"plan step: {tool_name}"
                ),
            )

        return self.transition(
            current.phase,
            reason=(
                "tool started for plan step "
                f"{current.id}: {tool_name}"
            ),
        )

    def on_tool_finished(
        self,
        *,
        error_code: str | None,
        error_message: str | None,
    ) -> PhaseTransition | None:
        failed = (
            error_code is not None
        )

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
                    f"tool failed: {error_code}"
                ),
            )

        current = self.current_step

        if current is not None:
            return self.transition(
                current.phase,
                reason=(
                    "tool completed for plan step "
                    f"{current.id}: "
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

    # ==================================================================
    # Explicit phases
    # ==================================================================

    def begin_verification(
        self,
        reason: str = "verification started",
    ) -> PhaseTransition | None:
        return self._activate_phase(
            AgentPhase.VERIFICATION,
            reason,
        )

    def begin_reflection(
        self,
        reason: str = "reflection required",
    ) -> PhaseTransition | None:
        return self._activate_phase(
            AgentPhase.REFLECTION,
            reason,
        )

    def begin_synthesis(
        self,
        reason: str = (
            "candidate final response produced"
        ),
    ) -> PhaseTransition | None:
        """
        Enter synthesis.

        Execution can be completed here because a candidate response
        exists.

        Verification/reflection are not completed implicitly.
        """

        if self._plan is not None:
            current = self.current_step

            if (
                current is not None
                and current.phase
                is AgentPhase.EXECUTION
                and current.status
                is not PlanStepStatus.COMPLETED
            ):
                current.complete(
                    "candidate response produced",
                )

            synthesis = (
                self._find_synthesis_step()
            )

            if synthesis is not None:
                self._plan.current_step_id = (
                    synthesis.id
                )

                if (
                    synthesis.status
                    is not PlanStepStatus.ACTIVE
                ):
                    synthesis.start()

        return self.transition(
            AgentPhase.SYNTHESIS,
            reason=reason,
        )

    def _activate_phase(
        self,
        phase: AgentPhase,
        reason: str,
    ) -> PhaseTransition | None:
        if self._plan is not None:
            current = self.current_step

            if (
                current is not None
                and current.phase is phase
            ):
                if (
                    current.status
                    is not PlanStepStatus.ACTIVE
                ):
                    current.start()

                return self.transition(
                    phase,
                    reason=reason,
                )

            for candidate in (
                self._plan.steps
            ):
                if (
                    candidate.phase is phase
                    and candidate.status
                    in {
                        PlanStepStatus.PENDING,
                        PlanStepStatus.FAILED,
                    }
                ):
                    self._plan.activate(
                        candidate,
                    )

                    return self.transition(
                        phase,
                        reason=reason,
                    )

        return self.transition(
            phase,
            reason=reason,
        )

    def _find_synthesis_step(
        self,
    ) -> PlanStep | None:
        if self._plan is None:
            return None

        for step in self._plan.steps:
            if (
                step.phase
                is AgentPhase.SYNTHESIS
            ):
                return step

        return None

    # ==================================================================
    # Completion
    # ==================================================================

    def complete(self) -> PhaseTransition:
        completion = self.check_completion()

        if not completion.allowed:
            if completion.next_phase is None:
                raise RuntimeError(
                    "Completion gate rejected completion "
                    "without a next phase",
                )

            transition = self.replan(
                completion.next_phase,
                reason=completion.reason,
            )

            if transition is None:
                raise RuntimeError(
                    "Completion gate requested replan, "
                    "but no transition was produced",
                )

            return transition

        if self._plan is not None:
            current = self.current_step

            if (
                current is not None
                and current.phase
                is AgentPhase.SYNTHESIS
            ):
                current.complete(
                    "final response accepted",
                )

        transition = self.transition(
            AgentPhase.COMPLETED,
            reason="agent run completed",
        )

        if transition is None:
            raise RuntimeError(
                "Agent is already in COMPLETED phase",
            )

        self.state.finished = True

        # The TaskPlan is a consumed plan. A subsequent new run must
        # provide another TaskPlan instead of accidentally reusing this one.
        self._task_plan_prepared = False

        return transition

    # ==================================================================
    # Blocking
    # ==================================================================

    def block(
        self,
        reason: str,
    ) -> PhaseTransition | None:
        if self._plan is not None:
            current = self.current_step

            if current is not None:
                current.fail(
                    reason,
                )

        self.state.last_error = reason

        return self.transition(
            AgentPhase.BLOCKED,
            reason=reason,
        )