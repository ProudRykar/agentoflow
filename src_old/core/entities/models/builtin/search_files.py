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

MAX_FILE_SIZE = 1_000_000


@dataclass(slots=True, frozen=True)
class SearchFilesInput:
    query: str
    path: str = "."
    max_results: int = 50


async def search_files(
    arguments: SearchFilesInput,
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

    if not arguments.query:
        raise ValueError("query must not be empty")

    if arguments.max_results <= 0:
        raise ValueError("max_results must be greater than 0")

    results: list[str] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if any(
            directory in SKIP_DIRECTORIES
            for directory in path.parts
        ):
            continue

        try:
            policy.resolve(path)

            if path.stat().st_size > MAX_FILE_SIZE:
                continue

            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError, OSError):
            continue

        relative_path = path.relative_to(root)

        for line_number, line in enumerate(
            content.splitlines(),
            start=1,
        ):
            if arguments.query not in line:
                continue

            results.append(
                f"{relative_path}:{line_number}: {line}"
            )

            if len(results) >= arguments.max_results:
                return "\n".join(results)

    if not results:
        return "(no matches)"

    return "\n".join(results)
