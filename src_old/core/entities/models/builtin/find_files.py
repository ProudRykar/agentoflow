from __future__ import annotations

from dataclasses import dataclass

from core.entities.models.path_policy import PathPolicy
from core.entities.models.tool import ToolContext


SKIP_DIRECTORIES = frozenset({
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
})


@dataclass(slots=True, frozen=True)
class FindFilesInput:
    pattern: str
    path: str = "."
    max_results: int = 100


async def find_files(
    arguments: FindFilesInput,
    context: ToolContext,
) -> str:
    policy = PathPolicy(context.allowed_path)

    root = policy.resolve(
        context.working_directory / arguments.path,
    )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Path '{arguments.path}' is not a directory"
        )

    if not arguments.pattern:
        raise ValueError("pattern must not be empty")

    if arguments.max_results <= 0:
        raise ValueError("max_results must be greater than 0")

    results: list[str] = []

    for path in root.rglob(arguments.pattern):
        if not path.is_file():
            continue

        if any(
            directory in SKIP_DIRECTORIES
            for directory in path.parts
        ):
            continue

        try:
            policy.resolve(path)
        except OSError:
            continue

        results.append(
            str(path.relative_to(context.working_directory))
        )

        if len(results) >= arguments.max_results:
            return "\n".join(results)

    if not results:
        return "(no matches)"

    return "\n".join(results)
