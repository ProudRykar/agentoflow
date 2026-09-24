from dataclasses import dataclass

from agent_workflow.core.entities.models.path_policy import PathPolicy
from agent_workflow.core.entities.models.tool import ToolContext


@dataclass(slots=True, frozen=True)
class ReadFileInput:
    path: str


async def read_file(
    arguments: ReadFileInput,
    context: ToolContext,
) -> str:
    policy = PathPolicy(context.allowed_path)

    path = policy.resolve(
        context.working_directory / arguments.path
    )

    return path.read_text(encoding="utf-8")