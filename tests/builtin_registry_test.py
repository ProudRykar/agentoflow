from core.entities.models.builtin.registry import (
    create_builtin_registry,
)


def test_builtin_registry_contains_read_file() -> None:
    registry = create_builtin_registry()

    assert registry.has("read_file")

    tool = registry.get("read_file")

    assert tool.name == "read_file"
    assert tool.description == "Read a UTF-8 text file"