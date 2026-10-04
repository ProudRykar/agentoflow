from __future__ import annotations

from pathlib import Path

import pytest

from agent_workflow.core.application.settings_service import (
    CONFIG_FILE,
    EDITABLE_FILES,
    MODELS_FILE,
    REDACTED,
    SettingsError,
    SettingsService,
    SettingsValidationError,
    is_secret_key,
)
from agent_workflow.core.infrastructure.paths import (
    AgentWorkflowPaths,
)


@pytest.fixture
def paths(tmp_path: Path) -> AgentWorkflowPaths:
    resolved = AgentWorkflowPaths(home=tmp_path)

    resolved.ensure()

    return resolved


@pytest.fixture
def service(paths: AgentWorkflowPaths) -> SettingsService:
    return SettingsService(paths=paths)


# ======================================================================
# Reading
# ======================================================================


def test_read_default_config(service: SettingsService) -> None:
    document = service.read(CONFIG_FILE)

    assert document.exists is True
    assert "agent" in document.data
    assert document.data["agent"]["max_iterations"] == 10


def test_read_missing_file(service: SettingsService) -> None:
    document = service.read(MODELS_FILE)

    assert document.exists is False
    assert document.data == {}


def test_only_known_files_are_editable(
    service: SettingsService,
) -> None:
    assert set(EDITABLE_FILES) == {CONFIG_FILE, MODELS_FILE}

    with pytest.raises(SettingsError, match="may be edited"):
        service.read("secrets.toml")

    with pytest.raises(SettingsError, match="may be edited"):
        service.read("../escape.toml")


def test_traversal_is_rejected(
    service: SettingsService,
    paths: AgentWorkflowPaths,
) -> None:
    with pytest.raises(SettingsError):
        service.read("../../etc/passwd")


# ======================================================================
# Redaction
# ======================================================================


def test_secret_key_detection() -> None:
    assert is_secret_key("token") is True
    assert is_secret_key("api_key") is True
    assert is_secret_key("Authorization") is True
    assert is_secret_key("github-token") is True
    assert is_secret_key("clientSecret") is True
    assert is_secret_key("model") is False
    assert is_secret_key("max_iterations") is False


def test_secrets_are_redacted_on_read(
    service: SettingsService,
) -> None:
    service.write(
        CONFIG_FILE,
        {
            "agent": {"max_iterations": 3},
            "mcp": {
                "enabled": True,
                "servers": [
                    {
                        "name": "s",
                        "command": "x",
                        "headers": {
                            "Authorization": "Bearer hunter2",
                        },
                    }
                ],
            },
        },
    )

    redacted = service.read(CONFIG_FILE).data

    headers = redacted["mcp"]["servers"][0]["headers"]

    assert headers["Authorization"] == REDACTED
    assert "hunter2" not in str(redacted)

    revealed = service.read(
        CONFIG_FILE,
        reveal_secrets=True,
    ).data

    assert (
        revealed["mcp"]["servers"][0]["headers"]["Authorization"]
        == "Bearer hunter2"
    )


def test_raw_text_masks_secrets(service: SettingsService) -> None:
    service.write(
        CONFIG_FILE,
        {
            "mcp": {
                "servers": [
                    {
                        "name": "s",
                        "command": "x",
                        "headers": {"authorization": "Bearer abc"},
                    }
                ]
            }
        },
    )

    text = service.raw_text(CONFIG_FILE)

    assert "abc" not in text
    assert REDACTED in text

    assert (
        "abc"
        in service.raw_text(
            CONFIG_FILE,
            reveal_secrets=True,
        )
    )


def test_redacted_write_preserves_stored_secret(
    service: SettingsService,
) -> None:
    service.write(
        CONFIG_FILE,
        {
            "mcp": {
                "servers": [
                    {
                        "name": "s",
                        "command": "x",
                        "headers": {"authorization": "Bearer keepme"},
                    }
                ]
            }
        },
    )

    # Simulate the UI round-trip: it sends back the placeholder.
    current = service.read(CONFIG_FILE).data

    service.write(CONFIG_FILE, current)

    stored = service.read(
        CONFIG_FILE,
        reveal_secrets=True,
    ).data

    assert (
        stored["mcp"]["servers"][0]["headers"]["authorization"]
        == "Bearer keepme"
    )


# ======================================================================
# Writing
# ======================================================================


def test_write_persists(service: SettingsService) -> None:
    service.write(
        CONFIG_FILE,
        {"agent": {"max_iterations": 42}},
    )

    assert service.read(CONFIG_FILE).data["agent"][
        "max_iterations"
    ] == 42


def test_write_replaces_previous_content(
    service: SettingsService,
) -> None:
    service.write(
        CONFIG_FILE,
        {"agent": {"max_iterations": 42}},
    )

    document = service.write(
        CONFIG_FILE,
        {"agent": {"max_iterations": 7}},
    )

    assert document.data["agent"]["max_iterations"] == 7

    raw = service.path_for(CONFIG_FILE).read_text()

    assert "42" not in raw


def test_write_creates_backup(
    service: SettingsService,
    paths: AgentWorkflowPaths,
) -> None:
    service.write(
        CONFIG_FILE,
        {"agent": {"max_iterations": 1}},
    )
    service.write(
        CONFIG_FILE,
        {"agent": {"max_iterations": 2}},
    )

    backups = list(paths.home.glob("config.toml.bak.*"))

    assert len(backups) >= 1
    assert "max_iterations = 1" in backups[0].read_text()


def test_write_leaves_no_temp_files(
    service: SettingsService,
    paths: AgentWorkflowPaths,
) -> None:
    service.write(
        CONFIG_FILE,
        {"agent": {"max_iterations": 5}},
    )

    leftovers = [
        item.name
        for item in paths.home.iterdir()
        if item.name.endswith(".tmp")
    ]

    assert leftovers == []


def test_write_rejects_invalid_config(
    service: SettingsService,
) -> None:
    with pytest.raises(SettingsValidationError):
        service.write(CONFIG_FILE, {"agent": "not-a-table"})

    with pytest.raises(SettingsValidationError):
        service.write(
            CONFIG_FILE,
            {"llm": {"timeout": "soon"}},
        )


def test_write_rejects_bad_model_catalog(
    service: SettingsService,
) -> None:
    with pytest.raises(SettingsValidationError):
        service.write(
            MODELS_FILE,
            {"models": [{"description": "no name"}]},
        )


def test_write_requires_mapping(service: SettingsService) -> None:
    with pytest.raises(SettingsValidationError):
        service.write(CONFIG_FILE, [1, 2])  # type: ignore[arg-type]


# ======================================================================
# Text editing
# ======================================================================


def test_write_text(service: SettingsService) -> None:
    document = service.write_text(
        CONFIG_FILE,
        "[agent]\nmax_iterations = 11\n",
    )

    assert document.data["agent"]["max_iterations"] == 11


def test_write_text_rejects_bad_toml(
    service: SettingsService,
) -> None:
    with pytest.raises(SettingsValidationError, match="Invalid TOML"):
        service.write_text(CONFIG_FILE, "this is not = = toml")


def test_write_text_rejects_unloadable(
    service: SettingsService,
) -> None:
    with pytest.raises(SettingsValidationError):
        service.write_text(
            CONFIG_FILE,
            "[agent]\nmax_iterations = \"many\"\n",
        )


# ======================================================================
# Models
# ======================================================================


def test_add_model(service: SettingsService) -> None:
    service.add_model(
        {
            "name": "qwen3:8b",
            "description": "balanced",
            "capabilities": {"reasoning": 3},
            "requirements": {"context_size": 32768},
        }
    )

    [model] = service.models()

    assert model["name"] == "qwen3:8b"
    assert model["capabilities"] == {"reasoning": 3}


def test_add_model_replaces_same_name(
    service: SettingsService,
) -> None:
    service.add_model({"name": "m", "description": "one"})
    service.add_model({"name": "m", "description": "two"})

    models = service.models()

    assert len(models) == 1
    assert models[0]["description"] == "two"


def test_add_model_requires_name(service: SettingsService) -> None:
    with pytest.raises(SettingsValidationError):
        service.add_model({"description": "nameless"})


def test_remove_model(service: SettingsService) -> None:
    service.add_model({"name": "a", "description": ""})
    service.add_model({"name": "b", "description": ""})

    service.remove_model("a")

    assert [m["name"] for m in service.models()] == ["b"]


def test_remove_unknown_model(service: SettingsService) -> None:
    with pytest.raises(SettingsValidationError, match="not found"):
        service.remove_model("ghost")


def test_models_when_catalog_missing(
    service: SettingsService,
) -> None:
    assert service.models() == []


# ======================================================================
# Round trip with the harness loaders
# ======================================================================


def test_written_config_loads(
    service: SettingsService,
    paths: AgentWorkflowPaths,
) -> None:
    from agent_workflow.core.infrastructure.config import ConfigLoader

    service.write(
        CONFIG_FILE,
        {
            "agent": {"max_iterations": 3},
            "llm": {"model": "custom:1b", "timeout": 12.5},
            "mcp": {
                "enabled": True,
                "servers": [
                    {
                        "name": "s",
                        "command": "npx",
                        "args": ["-y", "srv"],
                    }
                ],
            },
        },
    )

    config = ConfigLoader().load(paths.config)

    assert config.agent.max_iterations == 3
    assert config.llm.model == "custom:1b"
    assert config.mcp.enabled is True
    assert config.mcp.servers[0].name == "s"


def test_written_catalog_loads(
    service: SettingsService,
    paths: AgentWorkflowPaths,
) -> None:
    from agent_workflow.core.infrastructure.model_catalog import (
        ModelCatalogLoader,
    )

    service.add_model(
        {
            "name": "qwen3:8b",
            "description": "balanced",
            "capabilities": {"reasoning": 3, "coding": 4},
            "requirements": {"context_size": 32768, "thinking": True},
        }
    )

    catalog = ModelCatalogLoader().load(paths.models)

    profile = catalog.get("qwen3:8b")

    assert profile.requirements.context_size == 32768
    assert profile.requirements.thinking is True


def test_skill_style_multiline_survives(
    service: SettingsService,
) -> None:
    """Instructions with newlines must round-trip intact."""

    body = "# Title\n\nStep one.\nStep two.\n"

    service.write(
        MODELS_FILE,
        {"models": [{"name": "m", "description": body}]},
    )

    assert service.models()[0]["description"] == body

