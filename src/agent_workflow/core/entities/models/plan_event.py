"""Turning a runtime ``AgentPlan`` into a wire event.

Kept separate from the plan itself so that converting to a
JSON-friendly shape is one place, and so the mapping can be tested
without constructing an orchestrator or an agent.
"""

from __future__ import annotations

from agent_workflow.core.entities.models.agent_plan import AgentPlan
from agent_workflow.core.entities.models.agent_trace import PlanUpdated
from agent_workflow.core.entities.models.plan_step import PlanStep
from agent_workflow.core.entities.models.todo_list import TodoList


def step_payload(step: PlanStep) -> dict[str, object]:
    """One step, as a flat dict.

    Flat rather than nested so the payload needs no schema knowledge on
    replay, and every field a panel renders is present even when empty:
    a missing ``attempts`` and a zero are different things to a client
    deciding whether to show "attempt 2".
    """

    hint = step.delegation_hint

    return {
        "id": step.id,
        "description": step.description,
        "phase": str(step.phase),
        "status": str(step.status),
        "attempts": step.attempts,
        "result": step.result,
        "error": step.error,
        "delegation": (
            {
                "role": hint.role,
                "power": str(hint.power),
                "reason": hint.reason,
            }
            if hint is not None
            else None
        ),
    }


def todo_payload(todo: object) -> dict[str, object]:
    """One checklist item, in the same flat shape as a plan step.

    Same keys where the meaning matches, so a client renders one list
    component. ``status`` is the model's own vocabulary -- in_progress
    rather than active -- which is why the panel maps it rather than
    trusting the string.
    """

    return {
        "id": f"todo-{getattr(todo, 'content', '')}",
        "description": str(getattr(todo, "content", "")),
        "phase": "todo",
        "status": str(getattr(todo, "status", "pending")),
        "attempts": 0,
        "result": None,
        "error": None,
        "delegation": None,
    }


def plan_event(
    plan: AgentPlan,
    run_id: str = "",
    parent_run_id: str | None = None,
    todo_list: TodoList | None = None,
) -> PlanUpdated:
    """Snapshot the plan into an event."""

    return PlanUpdated(
        objective=plan.objective,
        steps=[step_payload(step) for step in plan.steps],
        revision=plan.revision,
        current_step_id=plan.current_step_id,
        run_id=run_id,
        parent_run_id=parent_run_id,
        todos=(
            None
            if todo_list is None or todo_list.is_empty()
            else [
                todo_payload(item)
                for item in todo_list.items
            ]
        ),
    )
