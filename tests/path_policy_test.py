from pathlib import Path

import pytest

from core.entities.models.path_policy import (
    PathPolicy,
    PathPolicyError,
)


def test_allowed_file(tmp_path: Path) -> None:
    policy = PathPolicy((tmp_path,))

    file_path = tmp_path / "file.txt"

    assert policy.resolve(file_path) == file_path.resolve()


def test_allowed_nested_file(tmp_path: Path) -> None:
    policy = PathPolicy((tmp_path,))

    nested = tmp_path / "sub" / "file.txt"

    assert policy.resolve(nested) == nested.resolve()


def test_parent_traversal_is_denied(tmp_path: Path) -> None:
    allowed = tmp_path / "work"
    allowed.mkdir()

    policy = PathPolicy((allowed,))

    path = allowed / ".." / "secret.txt"

    with pytest.raises(PathPolicyError):
        policy.resolve(path)


def test_absolute_path_outside_allowed_is_denied(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "work"
    allowed.mkdir()

    outside = tmp_path / "secret.txt"

    policy = PathPolicy((allowed,))

    with pytest.raises(PathPolicyError):
        policy.resolve(outside)


def test_multiple_allowed_paths(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first.mkdir()
    second.mkdir()

    policy = PathPolicy((first, second))

    assert policy.resolve(first / "file.txt") == (
        first / "file.txt"
    ).resolve()

    assert policy.resolve(second / "file.txt") == (
        second / "file.txt"
    ).resolve()