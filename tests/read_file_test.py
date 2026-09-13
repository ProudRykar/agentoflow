from pathlib import Path

import pytest

from core.entities.models.builtin.read_file import (
    ReadFileInput,
    read_file,
)
from core.entities.models.path_policy import PathPolicyError
from core.entities.models.tool import ToolContext


def create_context(
    working_directory: Path,
    allowed_path: tuple[Path, ...],
) -> ToolContext:
    return ToolContext(
        working_directory=working_directory,
        environment={},
        allowed_path=allowed_path,
        permissions=frozenset({"filesystem.read"}),
    )


@pytest.mark.asyncio
async def test_read_file(tmp_path: Path) -> None:
    file_path = tmp_path / "hello.txt"
    file_path.write_text(
        "Hello, agent!",
        encoding="utf-8",
    )

    context = create_context(
        working_directory=tmp_path,
        allowed_path=(tmp_path,),
    )

    result = await read_file(
        ReadFileInput(path="hello.txt"),
        context,
    )

    assert result == "Hello, agent!"


@pytest.mark.asyncio
async def test_read_nested_file(tmp_path: Path) -> None:
    nested = tmp_path / "documents"
    nested.mkdir()

    file_path = nested / "test.txt"
    file_path.write_text(
        "nested file",
        encoding="utf-8",
    )

    context = create_context(
        working_directory=tmp_path,
        allowed_path=(tmp_path,),
    )

    result = await read_file(
        ReadFileInput(path="documents/test.txt"),
        context,
    )

    assert result == "nested file"


@pytest.mark.asyncio
async def test_parent_traversal_is_denied(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "work"
    allowed.mkdir()

    secret = tmp_path / "secret.txt"
    secret.write_text(
        "secret",
        encoding="utf-8",
    )

    context = create_context(
        working_directory=allowed,
        allowed_path=(allowed,),
    )

    with pytest.raises(PathPolicyError):
        await read_file(
            ReadFileInput(path="../secret.txt"),
            context,
        )


@pytest.mark.asyncio
async def test_absolute_path_outside_allowed_is_denied(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "work"
    allowed.mkdir()

    secret = tmp_path / "secret.txt"
    secret.write_text(
        "secret",
        encoding="utf-8",
    )

    context = create_context(
        working_directory=allowed,
        allowed_path=(allowed,),
    )

    with pytest.raises(PathPolicyError):
        await read_file(
            ReadFileInput(path=str(secret)),
            context,
        )


@pytest.mark.asyncio
async def test_symlink_escape_is_denied(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "work"
    allowed.mkdir()

    secret = tmp_path / "secret.txt"
    secret.write_text(
        "secret",
        encoding="utf-8",
    )

    symlink = allowed / "secret-link"
    symlink.symlink_to(secret)

    context = create_context(
        working_directory=allowed,
        allowed_path=(allowed,),
    )

    with pytest.raises(PathPolicyError):
        await read_file(
            ReadFileInput(path="secret-link"),
            context,
        )