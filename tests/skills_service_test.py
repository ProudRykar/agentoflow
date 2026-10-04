from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_workflow.core.application.skills_service import (
    SKILL_FILENAME,
    SkillError,
    SkillsService,
    SkillValidationError,
    validate_name,
)


@pytest.fixture
def service(tmp_path: Path) -> SkillsService:
    return SkillsService(
        skills_directory=tmp_path / "skills"
    )


def _frontmatter(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    body = raw.split("---")[1]

    return yaml.safe_load(body)


# ======================================================================
# Names
# ======================================================================


def test_validate_name_normalises() -> None:
    assert validate_name("Python") == "python"
    assert validate_name("  my-skill  ") == "my-skill"
    assert validate_name("a_b1") == "a_b1"


@pytest.mark.parametrize(
    "name",
    ["", "   ", "../escape", "a/b", "a b", "-lead", "x" * 70],
)
def test_validate_name_rejects(name: str) -> None:
    with pytest.raises(SkillValidationError):
        validate_name(name)


# ======================================================================
# Create and read
# ======================================================================


def test_create_writes_skill_file(
    service: SkillsService,
    tmp_path: Path,
) -> None:
    service.create(
        "python-testing",
        description="How to test Python.",
        instructions="# Testing\n\n1. Write\n2. Run",
        version="2.0.0",
        metadata={"tags": ["python"]},
    )

    path = (
        tmp_path / "skills" / "python-testing" / SKILL_FILENAME
    )

    assert path.is_file()

    frontmatter = _frontmatter(path)

    assert frontmatter["name"] == "python-testing"
    assert frontmatter["description"] == "How to test Python."
    assert frontmatter["version"] == "2.0.0"
    assert frontmatter["tags"] == ["python"]


def test_created_skill_loads_by_the_agent(
    service: SkillsService,
) -> None:
    """The written file must be readable by the real loader."""

    from agent_workflow.core.entities.models.skills.loader import (
        SkillLoader,
    )
    from agent_workflow.core.entities.models.skills.registry import (
        SkillRegistry,
    )

    service.create(
        "demo",
        description="A demo skill.",
        instructions="Do the thing.",
    )

    loader = SkillLoader(SkillRegistry())

    [skill] = loader.load_directory(service.directory)

    assert skill.name == "demo"
    assert skill.instructions.strip() == "Do the thing."


def test_round_trip(service: SkillsService) -> None:
    service.create(
        "demo",
        description="Desc",
        instructions="Line one.\n\nLine two.",
    )

    document = service.get("demo")

    assert document.name == "demo"
    assert "Line one." in document.instructions
    assert "Line two." in document.instructions


def test_multiline_instructions_survive(
    service: SkillsService,
) -> None:
    body = (
        "# Title\n\n"
        "1. step one\n"
        "2. step two\n\n"
        "Notes with 'quotes' and \"doubles\"."
    )

    service.create(
        "demo",
        description="d",
        instructions=body,
    )

    assert service.get("demo").instructions == body


def test_metadata_cannot_shadow_core_fields(
    service: SkillsService,
) -> None:
    document = service.create(
        "demo",
        description="Real description",
        instructions="body",
        metadata={"name": "hijacked", "version": "9.9.9"},
    )

    assert document.name == "demo"
    assert document.description == "Real description"
    assert document.version == "1.0.0"


# ======================================================================
# Validation
# ======================================================================


def test_description_required(service: SkillsService) -> None:
    with pytest.raises(SkillValidationError, match="description"):
        service.create(
            "demo", description="  ", instructions="body"
        )


def test_instructions_required(service: SkillsService) -> None:
    with pytest.raises(SkillValidationError, match="instructions"):
        service.create(
            "demo", description="d", instructions="   "
        )


def test_long_description_rejected(
    service: SkillsService,
) -> None:
    with pytest.raises(SkillValidationError, match="too long"):
        service.create(
            "demo",
            description="x" * 600,
            instructions="body",
        )


def test_duplicate_rejected(service: SkillsService) -> None:
    service.create(
        "demo", description="d", instructions="i"
    )

    with pytest.raises(SkillValidationError, match="already exists"):
        service.create(
            "demo", description="d2", instructions="i2"
        )


def test_traversal_name_rejected(service: SkillsService) -> None:
    with pytest.raises(SkillValidationError):
        service.create(
            "../escape",
            description="d",
            instructions="i",
        )


def test_get_missing_skill(service: SkillsService) -> None:
    with pytest.raises(SkillError):
        service.get("ghost")


# ======================================================================
# Update, upsert, delete
# ======================================================================


def test_update_existing(service: SkillsService) -> None:
    service.create(
        "demo", description="old", instructions="old body"
    )

    service.update(
        "demo", description="new", instructions="new body"
    )

    document = service.get("demo")

    assert document.description == "new"
    assert document.instructions == "new body"


def test_update_missing_rejected(service: SkillsService) -> None:
    with pytest.raises(SkillValidationError, match="does not exist"):
        service.update(
            "ghost", description="d", instructions="i"
        )


def test_upsert_creates_then_updates(
    service: SkillsService,
) -> None:
    service.upsert(
        "demo", description="a", instructions="a"
    )
    service.upsert(
        "demo", description="b", instructions="b"
    )

    assert [s.name for s in service.list()] == ["demo"]
    assert service.get("demo").description == "b"


def test_delete_removes_file_and_dir(
    service: SkillsService,
    tmp_path: Path,
) -> None:
    service.create(
        "demo", description="d", instructions="i"
    )

    assert service.delete("demo") is True

    directory = tmp_path / "skills" / "demo"

    assert not directory.exists()


def test_delete_missing_returns_false(
    service: SkillsService,
) -> None:
    assert service.delete("ghost") is False


def test_delete_keeps_shared_directory(
    service: SkillsService,
    tmp_path: Path,
) -> None:
    service.create(
        "demo", description="d", instructions="i"
    )

    extra = tmp_path / "skills" / "demo" / "notes.md"
    extra.write_text("keep", encoding="utf-8")

    service.delete("demo")

    assert (tmp_path / "skills" / "demo").is_dir()
    assert extra.is_file()


# ======================================================================
# Listing
# ======================================================================


def test_list_sorted(service: SkillsService) -> None:
    for name in ("zeta", "alpha", "mid"):
        service.create(
            name, description="d", instructions="i"
        )

    assert [s.name for s in service.list()] == [
        "alpha",
        "mid",
        "zeta",
    ]


def test_list_empty_directory(service: SkillsService) -> None:
    assert service.list() == []


def test_list_ignores_broken_file(service: SkillsService) -> None:
    service.create(
        "good", description="d", instructions="i"
    )

    broken = service.directory / "bad"
    broken.mkdir(parents=True, exist_ok=True)
    (broken / SKILL_FILENAME).write_text(
        "no frontmatter", encoding="utf-8"
    )

    assert [s.name for s in service.list()] == ["good"]


def test_exists(service: SkillsService) -> None:
    assert service.exists("ghost") is False

    service.create("demo", description="d", instructions="i")

    assert service.exists("demo") is True
    assert service.exists("../x") is False


def test_no_temp_files_left(
    service: SkillsService,
    tmp_path: Path,
) -> None:
    service.create(
        "demo", description="d", instructions="i"
    )

    leftovers = [
        item.name
        for item in (tmp_path / "skills" / "demo").iterdir()
        if item.name.endswith(".tmp")
    ]

    assert leftovers == []
