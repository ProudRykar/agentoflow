from types import SimpleNamespace
from typing import Any

import pytest

from cli.approval import ApprovalController
from cli.ui.app import AgentUI
from cli.ui.state import UIState
from core.entities.models.agent_trace import AgentStarted


def _app() -> AgentUI:
    app = AgentUI(
        runtime=SimpleNamespace(
            agent=None,
            config=None,
            context=None,
        ),
        approval=ApprovalController(),
    )
    app._invalidate_ui = lambda **kwargs: None  # type: ignore[method-assign]

    return app


def test_start_run_stores_model_and_role() -> None:
    state = UIState()

    run = state.start_run(
        run_id="sub",
        parent_run_id="main",
        model="e2b",
        agent_id="subagent:researcher",
        role="researcher",
    )

    assert run.model == "e2b"
    assert run.role == "researcher"
    assert state.active_subagent_run_id == "sub"


def test_start_run_updates_existing() -> None:
    state = UIState()
    state.start_run(run_id="sub", parent_run_id="main")

    run = state.start_run(
        run_id="sub",
        parent_run_id="main",
        model="e2b",
        agent_id="subagent:researcher",
        role="researcher",
    )

    assert run.model == "e2b"
    assert run.role == "researcher"


@pytest.mark.asyncio
async def test_agent_started_records_subagent() -> None:
    app = _app()

    await app.handle_event(
        AgentStarted(
            prompt="do work",
            run_id="main",
            parent_run_id=None,
        )
    )
    await app.handle_event(
        AgentStarted(
            prompt="research",
            run_id="sub",
            parent_run_id="main",
            agent_id="subagent:researcher",
            role="researcher",
            model="e2b",
        )
    )

    run = app.state.get_run("sub")

    assert run is not None
    assert run.model == "e2b"
    assert run.role == "researcher"
    assert app.state.active_subagent_run_id == "sub"


def test_render_subagent_title_shows_role_model() -> None:
    app = _app()
    run = app.state.start_run(
        run_id="ef18af9f",
        parent_run_id="main",
        model="gemma3n:e2b-it-qat",
        agent_id="subagent:researcher",
        role="researcher",
    )

    panel = app._render_subagent(run)
    text = str(panel.title)

    assert "researcher" in text
    assert "gemma3n:e2b-it-qat" in text
    assert "running" in text


def test_render_subagent_title_falls_back_to_id() -> None:
    app = _app()
    run = app.state.start_run(
        run_id="ef18af9f86e4",
        parent_run_id="main",
    )

    text = str(app._render_subagent(run).title)

    assert "ef18af9f" in text


class _FakeBar:
    def __init__(self) -> None:
        self.text = ""

    def update(self, text: Any) -> None:
        self.text = str(text)


def test_status_line_shows_active_subagent() -> None:
    app = _app()
    app.status_bar = _FakeBar()  # type: ignore[assignment]
    app.state.running = True
    app.state.iteration = 3
    app.state.start_run(
        run_id="sub",
        parent_run_id="main",
        model="e2b",
        agent_id="subagent:researcher",
        role="researcher",
    )

    app._render_status()

    assert "researcher/e2b" in app.status_bar.text


def test_status_line_without_subagent_unchanged() -> None:
    app = _app()
    app.status_bar = _FakeBar()  # type: ignore[assignment]
    app.state.running = True
    app.state.iteration = 3

    app._render_status()

    assert "→" not in app.status_bar.text
