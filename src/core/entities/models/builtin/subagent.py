from __future__ import annotations

from dataclasses import dataclass

from core.entities.models.model_catalog import ModelCapability
from core.entities.models.subagent import (
    SubagentStatus,
    SubagentTask,
    TaskComplexity,
    TaskProfile,
)
from core.entities.models.subagent_manager import SubagentManager
from core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolError,
    ToolPolicy,
    ToolResult,
)


@dataclass(slots=True, frozen=True)
class SubagentRunInput:
    role: str
    objective: str

    instructions: str = ""
    context: str = ""

    capability: ModelCapability = (
        ModelCapability.GENERAL
    )

    complexity: TaskComplexity = (
        TaskComplexity.MEDIUM
    )

    min_context_tokens: int | None = None

    model: str = "auto"

    tools: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()

    max_iterations: int | None = None
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    timeout_seconds: float | None = None


def create_subagent_tool(
    manager: SubagentManager,
) -> Tool[SubagentRunInput, ToolResult]:
    async def handler(
        input_data: SubagentRunInput,
        context: ToolContext,
    ) -> ToolResult:
        task = SubagentTask(
            role=input_data.role,
            objective=input_data.objective,
            instructions=input_data.instructions,
            context=input_data.context,
            model=input_data.model,
            profile=TaskProfile(
                capability=input_data.capability,
                complexity=input_data.complexity,
                min_context_tokens=(
                    input_data.min_context_tokens
                ),
            ),
            tools=input_data.tools,
            permissions=frozenset(
                input_data.permissions,
            ),
            max_iterations=input_data.max_iterations,
            max_tool_calls=input_data.max_tool_calls,
            max_tokens=input_data.max_tokens,
            timeout_seconds=input_data.timeout_seconds,
        )

        result = await manager.run(
            task,
            parent_run_id=context.run_id,
            parent_context=context,
            on_event=context.event_callback,
        )

        if result.status is not SubagentStatus.COMPLETED:
            return ToolResult(
                error=ToolError(
                    message=(
                        result.error
                        or "Subagent failed"
                    ),
                    code=(
                        f"subagent_{result.status.value}"
                    ),
                    retryable=(
                        result.status
                        is SubagentStatus.TIMEOUT
                    ),
                ),
            )

        return ToolResult(
            output=result.output or "",
        )

    return Tool(
        name="subagent.run",
        description=(
            "Delegate a task to a separate specialized agent. "
            "Use this when a task benefits from independent "
            "reasoning, coding, research, analysis, or focused "
            "work. Prefer model='auto' so the system selects "
            "the appropriate model from the configured model "
            "catalog. Specify capability and complexity to "
            "describe the task requirements. The subagent runs "
            "with isolated context and a limited tool set."
        ),
        input_type=SubagentRunInput,
        handler=handler,
        policy=ToolPolicy(
            permissions=frozenset(),
            timeout=900.0,
            max_output_size=100_000,
        ),
    )