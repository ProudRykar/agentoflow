from core.entities.models.agent_phase import AgentPhase
from core.entities.models.plan_step import (
    DelegationHint,
    PlanStep,
    PlanStepStatus,
)
from core.entities.models.planner import Planner
from core.entities.models.subagent import SubagentPower
from core.entities.models.task_contract import TaskContract


def test_research_step_has_hint() -> None:
    planner = Planner()
    plan = planner.plan("Research https://example.com docs")
    contract = TaskContract(
        requires_research=True,
        research=plan.research,
    )

    execution = planner.build_execution_plan(plan, contract)
    research = execution.get_step("research")

    assert research is not None
    assert research.delegation_hint == DelegationHint(
        role="researcher",
        power=SubagentPower.LOW,
        reason=(
            "independent source traversal with "
            "isolated context"
        ),
    )


def test_synthesis_never_has_hint() -> None:
    planner = Planner()
    plan = planner.plan("Read test.txt")

    execution = planner.build_execution_plan(
        plan,
        TaskContract(),
    )
    synthesis = execution.get_step("synthesis")

    assert synthesis is not None
    assert synthesis.delegation_hint is None


def test_execution_coder_hint_only_for_code() -> None:
    planner = Planner()

    coding = planner.build_execution_plan(
        planner.plan("Напиши код парсера"),
        TaskContract(),
    )
    hint = coding.get_step("execution").delegation_hint

    assert hint is not None
    assert hint.role == "coder"
    assert hint.power is SubagentPower.MEDIUM

    reading = planner.build_execution_plan(
        planner.plan("Прочитай файл test.txt"),
        TaskContract(),
    )

    assert (
        reading.get_step("execution").delegation_hint
        is None
    )


def test_verification_has_hint_when_required() -> None:
    planner = Planner()
    plan = planner.plan("Напиши код парсера")
    contract = TaskContract(requires_verification=True)

    execution = planner.build_execution_plan(plan, contract)
    verification = execution.get_step("verification")

    assert verification is not None
    assert verification.delegation_hint is not None
    assert verification.delegation_hint.role == "reviewer"


def test_reflection_has_no_hint() -> None:
    planner = Planner()
    plan = planner.plan("Read test.txt")
    contract = TaskContract(requires_reflection=True)

    execution = planner.build_execution_plan(plan, contract)

    assert (
        execution.get_step("reflection").delegation_hint
        is None
    )


def test_revise_keeps_old_and_hints_new() -> None:
    planner = Planner()
    plan = planner.plan("Research https://example.com docs")
    contract = TaskContract(
        requires_research=True,
        research=plan.research,
    )

    execution = planner.build_execution_plan(plan, contract)
    old_hint = execution.get_step("research").delegation_hint

    assert old_hint is not None

    # Force a missing verification phase.
    step = planner.revise_for_missing_phase(
        execution,
        AgentPhase.VERIFICATION,
        "gate rejected",
    )

    assert step.delegation_hint is not None
    assert step.delegation_hint.role == "reviewer"

    # Existing hints survive replanning.
    assert (
        execution.get_step("research").delegation_hint
        == old_hint
    )


def test_hint_requires_role_and_reason() -> None:
    import pytest

    with pytest.raises(ValueError):
        DelegationHint(
            role="",
            power=SubagentPower.LOW,
            reason="reason",
        )

    with pytest.raises(ValueError):
        DelegationHint(
            role="researcher",
            power=SubagentPower.LOW,
            reason="",
        )


def test_plan_step_defaults_to_no_hint() -> None:
    step = PlanStep(
        id="execution",
        description="do work",
        phase=AgentPhase.EXECUTION,
        status=PlanStepStatus.PENDING,
    )

    assert step.delegation_hint is None
