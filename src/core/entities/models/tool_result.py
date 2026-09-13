from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ToolError:
    message: str
    code: str
    retryable: bool


@dataclass(slots=True, frozen=True)
class ToolResult:
    output: str | None = None
    error: ToolError | None = None