from __future__ import annotations

from dataclasses import dataclass

from core.entities.models.path_policy import PathPolicy
from core.entities.models.tool import ToolContext


@dataclass(slots=True, frozen=True)
class WriteFileInput:
    path: str
    content: str


async def write_file(
    arguments: WriteFileInput,
    context: ToolContext,
) -> str:
    policy = PathPolicy(context.allowed_path)

    path = policy.resolve(
        context.working_directory / arguments.path,
    )

    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"Parent directory '{path.parent}' does not exist"
        )

    path.write_text(
        arguments.content,
        encoding="utf-8",
    )

    return f"File '{arguments.path}' written successfully"