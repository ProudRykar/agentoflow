"""The plan has to survive a round trip to a browser and back.

Two things can go wrong and neither is visible in a unit test of the
panel alone: the event can fail to serialise, so the client receives
``unknown`` and silently shows nothing, or it can fail to replay, so a
reconnected session comes back with no plan at all.
"""

from __future__ import annotations

import pytest

from agent_workflow.core.application.event_store import (
    StoredEvent,
    decode_event,
    encode_event,
)
from agent_workflow.core.entities.models.agent_phase import AgentPhase
from agent_workflow.core.entities.models.agent_plan import AgentPlan
from agent_workflow.core.entities.models.agent_trace import PlanUpdated
from agent_workflow.core.entities.models.plan_event import plan_event
from agent_workflow.core.entities.models.plan_step import (
    PlanStep,
    PlanStepStatus,
)
from agent_workflow.core.entities.models.subagent import SubagentPower
from agent_workflow.web.serialization import event_type, to_wire


@pytest.fixture
def plan() -> AgentPlan:
    return AgentPlan(
        objective="find her other accounts",
        steps=[
            PlanStep(
                id="research",
                description="read the pages on file",
                phase=AgentPhase.RESEARCH,
                status=PlanStepStatus.COMPLETED,
                attempts=1,
            ),
            PlanStep(
                id="execution",
                description="expand the sources",
                phase=AgentPhase.EXECUTION,
                status=PlanStepStatus.ACTIVE,
                attempts=2,
                error=None,
                delegation_hint=None,
            ),
        ],
        current_step_id="execution",
        revision=2,
    )


def test_the_event_has_a_stable_wire_name(plan: AgentPlan) -> None:
    # The frontend switches on this string, so deriving it from the
    # class name would silently break the client.
    assert event_type(plan_event(plan)) == "plan.updated"


def test_every_field_a_panel_renders_is_present(plan: AgentPlan) -> None:
    event = plan_event(plan)

    assert event.objective == "find her other accounts"
    assert event.revision == 2
    assert event.current_step_id == "execution"
    assert [s["id"] for s in event.steps] == ["research", "execution"]

    active = event.steps[1]

    assert active["status"] == "active"
    assert active["attempts"] == 2
    # Present as null rather than absent: a client deciding whether to
    # draw "attempt 2" needs to tell zero from unknown.
    assert "result" in active
    assert "error" in active
    assert active["delegation"] is None


def test_enums_are_flat_strings_on_the_wire(plan: AgentPlan) -> None:
    wire = to_wire(plan_event(plan), session_id="s1", seq=4)

    assert wire["type"] == "plan.updated"
    assert wire["seq"] == 4
    assert wire["data"]["steps"][1]["phase"] == "execution"
    assert wire["data"]["steps"][1]["status"] == "active"


def test_a_delegation_hint_reaches_the_client() -> None:
    from agent_workflow.core.entities.models.plan_step import (
        DelegationHint,
    )

    plan = AgentPlan(
        objective="read a lot",
        steps=[
            PlanStep(
                id="research",
                description="look",
                phase=AgentPhase.RESEARCH,
                delegation_hint=DelegationHint(
                    role="explorer",
                    power=SubagentPower.LOW,
                    reason="many files",
                ),
            )
        ],
    )

    delegation = plan_event(plan).steps[0]["delegation"]

    assert delegation is not None
    assert delegation["role"] == "explorer"


def test_the_plan_survives_durable_replay(plan: AgentPlan) -> None:
    """A reconnected session must still have its plan.

    Without this the panel is empty after any reconnect, and the plan
    looks like it was never made.
    """

    event = plan_event(plan)

    restored = decode_event(encode_event(event))

    assert isinstance(restored, PlanUpdated)
    assert restored.objective == plan.objective
    assert restored.revision == 2
    assert restored.current_step_id == "execution"
    assert restored.steps[1]["status"] == "active"
    assert restored.steps[1]["attempts"] == 2


def test_a_plan_event_is_stored_like_any_other(plan: AgentPlan) -> None:
    stored = StoredEvent(seq=1, event=plan_event(plan))

    assert type(stored.event).__name__ == "PlanUpdated"
