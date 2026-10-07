"""The plan must read as a checklist, not a phase diagram.

Two things made the old rendering useless to a reader: the markers
were status words in brackets, and the one step anyone looks at --
execution -- carried the objective verbatim, so it said nothing about
what would count as doing it.
"""

from __future__ import annotations

import pytest

from agent_workflow.core.entities.models.agent_orchestrator import (
    AgentOrchestrator,
)
from agent_workflow.core.context.context_assembler import ContextAssembler
from agent_workflow.core.context.context_budget import ContextBudget
from agent_workflow.core.entities.models.agent_phase import AgentPhase
from agent_workflow.core.entities.models.agent_plan import AgentPlan
from agent_workflow.core.entities.models.plan_step import PlanStepStatus
from agent_workflow.core.entities.models.planner import Planner
from agent_workflow.core.entities.models.task_contract import TaskContract
from agent_workflow.core.entities.models.task_plan import TaskPlan


@pytest.mark.parametrize(
    ("status", "marker"),
    [
        (PlanStepStatus.COMPLETED, "[✓]"),
        (PlanStepStatus.ACTIVE, "[•]"),
        (PlanStepStatus.PENDING, "[○]"),
        (PlanStepStatus.FAILED, "[✗]"),
        (PlanStepStatus.SKIPPED, "[-]"),
    ],
)
def test_every_status_has_a_distinct_marker(
    status: PlanStepStatus,
    marker: str,
) -> None:
    assert status.todo_marker == marker


def test_markers_are_distinguishable() -> None:
    """A tick and a cross must not collapse into one glyph pair."""

    markers = {
        status.todo_marker
        for status in PlanStepStatus
    }

    assert len(markers) == len(list(PlanStepStatus))


def test_research_step_names_its_sources() -> None:
    planner = Planner()

    task = planner.plan(
        "Изучи документацию по https://example.com/a, "
        "минимум 5 страниц, глубина 2",
    )

    plan = planner.build_execution_plan(
        task,
        TaskContract(
            requires_research=True,
            research=task.research,
        ),
    )

    research = next(
        step
        for step in plan.steps
        if step.phase is AgentPhase.RESEARCH
    )

    # An end state that can be checked, rather than "do research".
    assert "https://example.com/a" in research.description
    assert "5" in research.description
    assert "2" in research.description


def test_execution_step_is_not_the_bare_objective() -> None:
    planner = Planner()

    task = TaskPlan(objective="сравнить два API")

    plan = planner.build_execution_plan(
        task,
        TaskContract(),
    )

    execution = next(
        step
        for step in plan.steps
        if step.id == "execution"
    )

    # Restating the objective in the step that acts on it told the
    # reader nothing about what doing it would look like.
    assert execution.description != "сравнить два API"
    assert "сравнить два API" in execution.description
    assert execution.description.startswith("Выполнить задачу:")


def test_execution_step_does_not_repeat_another_checklist_item() -> None:
    """Verification is already its own item; asking twice is noise."""

    planner = Planner()

    plan = planner.build_execution_plan(
        TaskPlan(objective="найти источник"),
        TaskContract(requires_verification=True),
    )

    execution = next(
        step
        for step in plan.steps
        if step.id == "execution"
    )

    assert "проверить результат" not in execution.description.lower()

    assert any(
        "проверить результат" in step.description.lower()
        for step in plan.steps
    )


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (
            "Найди https://example.com/x, at least 3 pages",
            # The verb is not the goal, and the coverage requirement
            # belongs to the research contract.
            "Исследовать указанные в источниках материалы.",
        ),
        ("Сравни API на https://a.com и https://b.com", "API"),
        ("Напиши тест для парсера", "тест для парсера"),
    ],
)
def test_objective_is_not_left_dangling(
    prompt: str,
    expected: str,
) -> None:
    assert Planner().plan(prompt).objective == expected


def test_plan_renders_as_a_todo_list() -> None:
    planner = Planner()

    task = planner.plan("Напиши тест для парсера")

    plan = planner.build_execution_plan(
        task,
        TaskContract(requires_verification=True),
    )

    text = _render_task_state(plan)

    assert "▼Todo" in text

    # Markers, not status words in brackets.
    assert "[active]" not in text
    assert "[•]" in text
    assert "[○]" in text

    # The phase name is redundant next to the transition that already
    # announced it.
    assert "(execution)" not in text


def _render_task_state(plan: AgentPlan) -> str:
    """Render the plan block the model is actually shown."""

    orchestrator = AgentOrchestrator()
    orchestrator.prepare_task(
        TaskPlan(objective=plan.objective),
        TaskContract(),
        prompt=plan.objective,
    )
    orchestrator.on_agent_started(plan.objective)
    orchestrator.initialize_plan(plan)

    assembler = ContextAssembler(
        budget=ContextBudget(
            maximum_tokens=8000,
            reserved_system=200,
            reserved_task=200,
            reserved_output=400,
        ),
    )

    request = assembler.build(
        anchor=orchestrator.task_anchor,
        task_state=orchestrator.task_state,
        execution=orchestrator.execution_context,
        conversation_recent=(),
        evidence_selected=(),
        execution_plan=orchestrator.plan,
    )

    return "\n".join(
        message["content"]
        for message in request.messages
    )


# ======================================================================
# Silent truncation of a fetched page
# ======================================================================


def test_a_truncated_page_says_so() -> None:
    """
    Found by auditing a dead local: the truncation flag was computed
    and dropped, so byte_limit_reached reported "no limit reached" for
    a page that had been cut.
    """

    from agent_workflow.core.entities.models.research_contract import (
        ResearchPage,
        ResearchResult,
    )

    page = ResearchPage(
        url="https://example.com/a",
        depth=0,
        title="t",
        content="prefix",
        links=(),
        content_bytes=999_999,
        truncated=True,
    )

    result = ResearchResult(
        root_url=page.url,
        pages=(page,),
        discovered_urls=(),
        failed_urls=(),
        max_depth_reached=0,
        total_bytes=999_999,
        page_limit_reached=False,
        byte_limit_reached=False,
    )

    payload = result.to_json()

    # content_bytes is the full length, so the reader needs to be told
    # the content is only a prefix of it.
    assert '"truncated":true' in payload
    assert '"byte_limit_reached":false' in payload


def test_a_whole_page_is_not_reported_as_truncated() -> None:
    from agent_workflow.core.entities.models.research_contract import (
        ResearchPage,
    )

    page = ResearchPage(
        url="https://example.com/a",
        depth=0,
        title="t",
        content="whole",
        links=(),
        content_bytes=5,
    )

    assert page.truncated is False
    assert page.http_status is None
