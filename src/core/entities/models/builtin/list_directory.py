from __future__ import annotations

from dataclasses import dataclass

from core.entities.models.path_policy import PathPolicy
from core.entities.models.tool import ToolContext


@dataclass(slots=True, frozen=True)
class ListDirectoryInput:
    path: str = "."


async def list_directory(
    arguments: ListDirectoryInput,
    context: ToolContext,
) -> str:
    policy = PathPolicy(context.allowed_path)

    path = policy.resolve(
        context.working_directory / arguments.path,
    )

    if not path.is_dir():
        raise NotADirectoryError(
            f"Path '{arguments.path}' is not a directory"
        )

    entries = sorted(
        path.iterdir(),
        key=lambda entry: (not entry.is_dir(), entry.name.lower()),
    )

    if not entries:
        return "(empty)"

    return "\n".join(
        f"{entry.name}/" if entry.is_dir() else entry.name
        for entry in entries
    )