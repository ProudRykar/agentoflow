from __future__ import annotations

from dataclasses import dataclass

from agent_workflow.core.entities.models.path_policy import PathPolicy
from agent_workflow.core.entities.models.tool import ToolContext


@dataclass(slots=True, frozen=True)
class EditFileInput:
    path: str
    old_text: str
    new_text: str


async def edit_file(
    arguments: EditFileInput,
    context: ToolContext,
) -> str:
    policy = PathPolicy(context.allowed_path)

    path = policy.resolve(
        context.working_directory / arguments.path,
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"File '{arguments.path}' does not exist"
        )

    content = path.read_text(encoding="utf-8")

    occurrences = content.count(arguments.old_text)

    if occurrences == 0:
        raise ValueError(
            f"Text to replace was not found in '{arguments.path}'"
        )

    if occurrences > 1:
        raise ValueError(
            f"Text to replace occurs {occurrences} times "
            f"in '{arguments.path}'; expected exactly one occurrence"
        )

    updated_content = content.replace(
        arguments.old_text,
        arguments.new_text,
        1,
    )

    path.write_text(
        updated_content,
        encoding="utf-8",
    )

    return f"File '{arguments.path}' edited successfully"
