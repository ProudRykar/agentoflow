from core.context.synthesis import (
    CITATION_RULE,
    GROUNDING_RULE,
    REQUIREMENTS_RULE,
    build_synthesis_guide,
)
from core.entities.models.agent_orchestrator import AgentOrchestrator
from core.entities.models.planner import Planner
from core.entities.models.task_contract import TaskContract


def _research_state() -> object:
    orchestrator = AgentOrchestrator()
    plan = Planner().plan("Research https://example.com docs")
    orchestrator.prepare_task(
        plan,
        TaskContract(
            requires_research=True,
            research=plan.research,
        ),
        prompt="Research https://example.com docs",
    )
    orchestrator.on_agent_started(
        "Research https://example.com docs"
    )

    return orchestrator.task_state


def _plain_state() -> object:
    orchestrator = AgentOrchestrator()
    orchestrator.prepare_task(
        Planner().plan("Read test.txt"),
        TaskContract(),
        prompt="Read test.txt",
    )
    orchestrator.on_agent_started("Read test.txt")

    return orchestrator.task_state


def test_research_guide_has_sections_and_rules() -> None:
    guide = build_synthesis_guide(_research_state())  # type: ignore[arg-type]

    assert len(guide.sections) == 4
    assert any("Resources" in s for s in guide.sections)
    assert GROUNDING_RULE in guide.rules
    assert CITATION_RULE in guide.rules
    assert REQUIREMENTS_RULE in guide.rules

    text = guide.render()

    assert "[SYNTHESIS GUIDE]" in text
    assert "source URL" in text


def test_plain_guide_is_minimal() -> None:
    guide = build_synthesis_guide(_plain_state())  # type: ignore[arg-type]

    assert guide.sections == ()
    assert guide.rules == (GROUNDING_RULE,)


def test_research_contract_without_flag_still_guides() -> None:
    orchestrator = AgentOrchestrator()
    plan = Planner().plan("Research https://example.com docs")
    orchestrator.prepare_task(
        plan,
        TaskContract(research=plan.research),
        prompt="Research https://example.com docs",
    )
    orchestrator.on_agent_started(
        "Research https://example.com docs"
    )

    # TaskContract auto-sets requires_research with research.
    assert orchestrator.task_contract.requires_research is True

    guide = build_synthesis_guide(orchestrator.task_state)

    assert len(guide.sections) == 4
