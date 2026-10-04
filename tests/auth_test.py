from __future__ import annotations


from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from agent_workflow.web.app import create_app
from agent_workflow.web.auth import (
    TOKEN_ENV,
    AuthSettings,
)


@pytest.fixture
def secure_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    monkeypatch.setenv(TOKEN_ENV, "secret-token-value")

    app = create_app()

    with TestClient(app) as client:
        yield client


def test_auth_disabled_without_token() -> None:
    settings = AuthSettings.from_env()

    assert settings.enabled is False
    # Disabled means every candidate is accepted.
    assert settings.matches("") is True
    assert settings.matches("anything") is True


def test_auth_enabled_with_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "secret-token-value")

    settings = AuthSettings.from_env()

    assert settings.enabled is True
    assert settings.matches("secret-token-value") is True
    assert settings.matches("wrong") is False
    assert settings.matches("") is False


def test_blank_token_is_treated_as_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "   ")

    assert AuthSettings.from_env().enabled is False


def test_rest_requires_token(secure_client: TestClient) -> None:
    response = secure_client.get("/api/sessions")

    assert response.status_code == 401


def test_rest_accepts_header_token(
    secure_client: TestClient,
) -> None:
    response = secure_client.get(
        "/api/sessions",
        headers={"x-agentoflow-token": "secret-token-value"},
    )

    assert response.status_code == 200


def test_rest_accepts_bearer_token(
    secure_client: TestClient,
) -> None:
    response = secure_client.get(
        "/api/sessions",
        headers={"authorization": "Bearer secret-token-value"},
    )

    assert response.status_code == 200


def test_rest_rejects_wrong_token(
    secure_client: TestClient,
) -> None:
    response = secure_client.get(
        "/api/sessions",
        headers={"x-agentoflow-token": "nope"},
    )

    assert response.status_code == 401


def test_websocket_rejects_missing_token(
    secure_client: TestClient,
) -> None:
    from starlette.websockets import WebSocketDisconnect

    session_id = secure_client.post(
        "/api/sessions",
        headers={"x-agentoflow-token": "secret-token-value"},
    ).json()["session_id"]

    with pytest.raises(WebSocketDisconnect) as info:
        with secure_client.websocket_connect(
            f"/ws/sessions/{session_id}"
        ):
            pass

    assert info.value.code == 4001


def test_websocket_accepts_query_token(
    secure_client: TestClient,
) -> None:
    import json

    session_id = secure_client.post(
        "/api/sessions",
        headers={"x-agentoflow-token": "secret-token-value"},
    ).json()["session_id"]

    with secure_client.websocket_connect(
        f"/ws/sessions/{session_id}?token=secret-token-value"
    ) as websocket:
        message = json.loads(websocket.receive_text())

    assert message["type"] == "session.snapshot"


def test_token_is_not_hardcoded_in_source() -> None:
    """Credentials must never be committed."""

    import agent_workflow.web.auth as auth_module

    source = (
        auth_module.__file__ and open(auth_module.__file__).read()
    )

    assert "secret-token-value" not in source


def test_default_bind_is_localhost() -> None:
    """Remote exposure must be opt-in, not the default."""

    import inspect

    import agent_workflow.web.app as app_module

    source = inspect.getsource(app_module.main)

    assert '"127.0.0.1"' in source
    assert '"8000"' in source
    assert 'os.getenv("AGENTOFLOW_HOST"' in source
