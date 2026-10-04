from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_workflow.core.application.settings_service import (
    SettingsService,
)
from agent_workflow.core.application.skills_service import (
    SkillsService,
)
from agent_workflow.core.infrastructure.paths import (
    AgentWorkflowPaths,
)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """A client whose settings and skills live under tmp_path."""

    from agent_workflow.core.application.session import SessionManager
    from agent_workflow.core.application.session_store import (
        SessionStore,
    )
    from agent_workflow.core.application.tool_stats_store import (
        ToolStatsStore,
    )
    from agent_workflow.web.app import create_app

    paths = AgentWorkflowPaths(home=tmp_path)
    paths.ensure()

    app = create_app(
        session_manager=SessionManager(
            store=SessionStore(tmp_path / "sessions.json"),
            stats_store=ToolStatsStore(tmp_path / "usage.db"),
        )
    )

    app.state.settings_service = SettingsService(paths=paths)
    app.state.skills_library = SkillsService(
        skills_directory=tmp_path / "skills"
    )

    with TestClient(app) as test_client:
        yield test_client


# ======================================================================
# Settings
# ======================================================================


def test_list_editable_files(client: TestClient) -> None:
    response = client.get("/api/settings")

    assert response.status_code == 200
    assert set(response.json()) == {"config.toml", "models.toml"}


def test_read_config(client: TestClient) -> None:
    response = client.get("/api/settings/config.toml")

    assert response.status_code == 200

    body = response.json()

    assert body["exists"] is True
    assert "agent" in body["data"]
    assert "[agent]" in body["text"]


def test_read_rejects_unknown_file(client: TestClient) -> None:
    assert (
        client.get("/api/settings/other.toml").status_code == 400
    )


def test_write_config(client: TestClient) -> None:
    response = client.put(
        "/api/settings/config.toml",
        json={
            "data": {
                "agent": {"max_iterations": 99},
                "llm": {"model": "custom:1b"},
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["data"]["agent"]["max_iterations"] == 99

    reread = client.get("/api/settings/config.toml").json()

    assert reread["data"]["llm"]["model"] == "custom:1b"


def test_write_invalid_config_is_422(
    client: TestClient,
) -> None:
    response = client.put(
        "/api/settings/config.toml",
        json={"data": {"agent": "nope"}},
    )

    assert response.status_code == 422
    assert "would not load" in response.json()["detail"]


def test_write_requires_payload(client: TestClient) -> None:
    assert (
        client.put("/api/settings/config.toml", json={}).status_code
        == 422
    )


def test_write_text_editor(client: TestClient) -> None:
    response = client.put(
        "/api/settings/config.toml",
        json={"text": "[agent]\nmax_iterations = 5\n"},
    )

    assert response.status_code == 200

    body = client.get("/api/settings/config.toml").json()

    assert body["data"]["agent"]["max_iterations"] == 5


def test_write_text_rejects_bad_toml(client: TestClient) -> None:
    response = client.put(
        "/api/settings/config.toml",
        json={"text": "= = bad"},
    )

    assert response.status_code == 422


def test_secrets_are_redacted_over_the_api(
    client: TestClient,
) -> None:
    client.put(
        "/api/settings/config.toml",
        json={
            "data": {
                "mcp": {
                    "enabled": True,
                    "servers": [
                        {
                            "name": "s",
                            "command": "x",
                            "headers": {
                                "authorization": "Bearer topsecret"
                            },
                        }
                    ],
                }
            }
        },
    )

    body = client.get("/api/settings/config.toml")

    assert "topsecret" not in body.text
    assert "topsecret" not in body.json()["text"]


# ======================================================================
# Models
# ======================================================================


def test_models_start_empty(client: TestClient) -> None:
    response = client.get("/api/settings/models/list")

    assert response.status_code == 200
    assert response.json() == []


def test_add_and_list_model(client: TestClient) -> None:
    created = client.put(
        "/api/settings/models/entry",
        json={
            "name": "qwen3:8b",
            "description": "balanced",
            "capabilities": {"reasoning": 3},
            "requirements": {"context_size": 32768},
        },
    )

    assert created.status_code == 200

    [model] = client.get(
        "/api/settings/models/list"
    ).json()

    assert model["name"] == "qwen3:8b"
    assert model["capabilities"] == {"reasoning": 3}
    assert model["requirements"]["context_size"] == 32768


def test_model_update_replaces(client: TestClient) -> None:
    for description in ("one", "two"):
        client.put(
            "/api/settings/models/entry",
            json={"name": "m", "description": description},
        )

    models = client.get("/api/settings/models/list").json()

    assert len(models) == 1
    assert models[0]["description"] == "two"


def test_invalid_model_is_422(client: TestClient) -> None:
    response = client.put(
        "/api/settings/models/entry",
        json={"description": "nameless"},
    )

    assert response.status_code == 422


def test_delete_model(client: TestClient) -> None:
    client.put(
        "/api/settings/models/entry",
        json={"name": "m", "description": ""},
    )

    assert (
        client.delete(
            "/api/settings/models/entry/m"
        ).status_code
        == 200
    )
    assert client.get("/api/settings/models/list").json() == []


def test_delete_unknown_model_is_422(
    client: TestClient,
) -> None:
    assert (
        client.delete(
            "/api/settings/models/entry/ghost"
        ).status_code
        == 422
    )


# ======================================================================
# Skills library
# ======================================================================


def test_skills_library_starts_empty(
    client: TestClient,
) -> None:
    response = client.get("/api/skills/library")

    assert response.status_code == 200
    assert response.json() == []


def test_create_and_read_skill(client: TestClient) -> None:
    created = client.post(
        "/api/skills/library",
        json={
            "name": "demo",
            "description": "A demo skill.",
            "instructions": "# Demo\n\nStep one.",
        },
    )

    assert created.status_code == 200
    assert created.json()["name"] == "demo"

    fetched = client.get("/api/skills/library/demo")

    assert fetched.status_code == 200
    assert "Step one." in fetched.json()["instructions"]


def test_skill_appears_in_list(client: TestClient) -> None:
    client.post(
        "/api/skills/library",
        json={
            "name": "demo",
            "description": "d",
            "instructions": "i",
        },
    )

    [skill] = client.get("/api/skills/library").json()

    assert skill["name"] == "demo"
    assert skill["instructions"] == ""


def test_duplicate_skill_is_422(client: TestClient) -> None:
    payload = {
        "name": "demo",
        "description": "d",
        "instructions": "i",
    }

    client.post("/api/skills/library", json=payload)

    assert (
        client.post(
            "/api/skills/library", json=payload
        ).status_code
        == 422
    )


def test_update_skill(client: TestClient) -> None:
    client.post(
        "/api/skills/library",
        json={
            "name": "demo",
            "description": "old",
            "instructions": "old body",
        },
    )

    response = client.put(
        "/api/skills/library/demo",
        json={
            "name": "demo",
            "description": "new",
            "instructions": "new body",
        },
    )

    assert response.status_code == 200
    assert response.json()["description"] == "new"


def test_delete_skill(client: TestClient) -> None:
    client.post(
        "/api/skills/library",
        json={
            "name": "demo",
            "description": "d",
            "instructions": "i",
        },
    )

    assert (
        client.delete(
            "/api/skills/library/demo"
        ).status_code
        == 200
    )
    assert (
        client.delete(
            "/api/skills/library/demo"
        ).status_code
        == 404
    )


def test_missing_skill_is_404(client: TestClient) -> None:
    assert (
        client.get("/api/skills/library/ghost").status_code == 404
    )


def test_invalid_skill_name_is_422(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/skills/library",
        json={
            "name": "../escape",
            "description": "d",
            "instructions": "i",
        },
    )

    assert response.status_code == 422


# ======================================================================
# Session naming
# ======================================================================


def test_session_has_derived_title(
    client: TestClient,
) -> None:
    session_id = client.post("/api/sessions").json()[
        "session_id"
    ]

    info = client.get(f"/api/sessions/{session_id}").json()

    assert info["title"]
    assert info["title_source"] == "auto"


def test_rename_session(client: TestClient) -> None:
    session_id = client.post("/api/sessions").json()[
        "session_id"
    ]

    response = client.patch(
        f"/api/sessions/{session_id}/title",
        json={"title": "My important chat"},
    )

    assert response.status_code == 200
    assert response.json()["title"] == "My important chat"

    info = client.get(f"/api/sessions/{session_id}").json()

    assert info["title"] == "My important chat"
    assert info["title_source"] == "manual"


def test_rename_rejects_empty(client: TestClient) -> None:
    session_id = client.post("/api/sessions").json()[
        "session_id"
    ]

    assert (
        client.patch(
            f"/api/sessions/{session_id}/title",
            json={"title": ""},
        ).status_code
        == 422
    )


def test_rename_unknown_session(client: TestClient) -> None:
    assert (
        client.patch(
            "/api/sessions/missing/title",
            json={"title": "x"},
        ).status_code
        == 404
    )


def test_rename_survives_listing(client: TestClient) -> None:
    session_id = client.post("/api/sessions").json()[
        "session_id"
    ]

    client.patch(
        f"/api/sessions/{session_id}/title",
        json={"title": "Renamed"},
    )

    sessions = client.get("/api/sessions").json()

    [info] = sessions

    assert info["title"] == "Renamed"


def test_delete_session_removes_it(client: TestClient) -> None:
    session_id = client.post("/api/sessions").json()[
        "session_id"
    ]

    assert (
        client.delete(
            f"/api/sessions/{session_id}"
        ).status_code
        == 200
    )
    assert (
        client.get(
            f"/api/sessions/{session_id}"
        ).status_code
        == 404
    )
    assert client.get("/api/sessions").json() == []
