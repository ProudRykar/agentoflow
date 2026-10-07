"""Pinning a skill must put its text in the prompt without a tool call.

The problem being solved is an agent that never loads the skill which
would have told it what to do. Leaving that to a tool call means the
decision to ask is the very decision that gets skipped, so the reader
pins it from the UI instead.

These tests go through the HTTP surface because the interesting
failures are at the edges: a typo must not look like success, and the
pin has to survive a restart.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


SKILL = """---
name: stash-research
description: How to query the Stash media library without guessing.
version: 1.0.0
---

# Stash Research

Always check the schema before writing a query.
"""


def _make_runtime_factory(skills_dir: Path):
    """A stub runtime whose agent actually has a skill manager."""

    async def factory(approval_handler=None, working_directory=None, **kwargs):
        from agent_workflow.core.application.runtime import AgentRuntime
        from agent_workflow.core.entities.models.agent import Agent
        from agent_workflow.core.entities.models.skills.manager import (
            SkillManager,
        )
        from agent_workflow.core.entities.models.tool import ToolContext
        from agent_workflow.core.entities.models.tool_executor import (
            ToolExecutor,
        )
        from agent_workflow.core.entities.models.tool_registry import (
            ToolRegistry,
        )
        from agent_workflow.core.entities.models.tool_toggle import (
            ToolToggle,
        )
        from agent_workflow.core.infrastructure.config import AgentConfig, LLMConfig
        from agent_workflow.core.infrastructure.paths import (
            AgentWorkflowPaths,
        )

        registry = ToolRegistry()

        class NoopStore:
            def close(self) -> None:
                return None

        return AgentRuntime(
            agent=Agent(
                llm=None,
                registry=registry,
                executor=ToolExecutor(registry),
                run_id="run-stub",
                skill_manager=SkillManager(skills_dir),
            ),
            context=ToolContext(
                working_directory=working_directory or skills_dir,
                environment={},
                allowed_path=(working_directory or skills_dir,),
                permissions=frozenset(),
            ),
            store=NoopStore(),  # type: ignore[arg-type]
            paths=AgentWorkflowPaths(home=skills_dir.parent),
            config=type(
                "StubConfig",
                (),
                {
                    "llm": LLMConfig(model="stub-model"),
                    "agent": AgentConfig(),
                },
            )(),
            plugin_manager=None,
            tool_toggle=ToolToggle(registry),
        )

    return factory


@pytest.fixture
def skills_client(tmp_path: Path):
    from agent_workflow.core.application import session as session_module

    skills_dir = tmp_path / "skills" / "stash-research"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(SKILL, encoding="utf-8")

    original = session_module.create_runtime

    session_module.create_runtime = _make_runtime_factory(
        tmp_path / "skills",
    )

    from agent_workflow.web.app import create_app

    try:
        with TestClient(create_app()) as client:
            yield client
    finally:
        session_module.create_runtime = original


def test_listing_shows_the_skill(skills_client: TestClient) -> None:
    session_id = skills_client.post("/api/sessions").json()["session_id"]

    listed = skills_client.get(f"/api/skills/{session_id}").json()

    assert [item["name"] for item in listed] == ["stash-research"]
    assert listed[0]["preloaded"] is False
    assert listed[0]["active"] is False


def test_preload_activates_the_skill(skills_client: TestClient) -> None:
    session_id = skills_client.post("/api/sessions").json()["session_id"]

    response = skills_client.post(
        f"/api/skills/{session_id}/preload",
        json={"skills": ["stash-research"]},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["preloaded"] == ["stash-research"]
    assert "stash-research" in body["active"]

    listed = skills_client.get(f"/api/skills/{session_id}").json()

    # Active means the text is already in the next prompt, which is the
    # whole reason the toggle exists.
    assert listed[0]["active"] is True
    assert listed[0]["preloaded"] is True


def test_preload_rejects_a_typo_instead_of_silently_doing_nothing(
    skills_client: TestClient,
) -> None:
    session_id = skills_client.post("/api/sessions").json()["session_id"]

    response = skills_client.post(
        f"/api/skills/{session_id}/preload",
        json={"skills": ["stash-reasearch"]},
    )

    assert response.status_code == 404
    assert "stash-reasearch" in response.json()["detail"]

    listed = skills_client.get(f"/api/skills/{session_id}").json()

    assert listed[0]["preloaded"] is False


def test_preload_can_be_cleared(skills_client: TestClient) -> None:
    session_id = skills_client.post("/api/sessions").json()["session_id"]

    skills_client.post(
        f"/api/skills/{session_id}/preload",
        json={"skills": ["stash-research"]},
    )

    response = skills_client.post(
        f"/api/skills/{session_id}/preload",
        json={"skills": []},
    )

    assert response.status_code == 200
    assert response.json()["preloaded"] == []

    listed = skills_client.get(f"/api/skills/{session_id}").json()

    assert listed[0]["active"] is False
    assert listed[0]["preloaded"] is False


def test_the_state_store_records_the_preload(tmp_path: Path) -> None:
    from agent_workflow.core.application.session_state_store import (
        SessionStateStore,
    )

    store = SessionStateStore(tmp_path / "state.db")

    store.save("s1", preloaded_skills=("stash-research",))

    assert store.preloaded_skills("s1") == ("stash-research",)