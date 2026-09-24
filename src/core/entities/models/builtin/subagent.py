from __future__ import annotations

from dataclasses import dataclass

from core.entities.models.model_catalog import ModelCapability
from core.entities.models.subagent import (
    SubagentPower,
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
class RolePreset:
    """Defaults for a well-known subagent role.

    Presets only fill fields left at their defaults;
    explicit call arguments always win.
    """

    capability: ModelCapability
    power: SubagentPower
    tools: tuple[str, ...]


ROLE_PRESETS: dict[str, RolePreset] = {
    "researcher": RolePreset(
        capability=ModelCapability.RESEARCH,
        power=SubagentPower.LOW,
        tools=(
            "web_fetch",
            "web_crawl",
        ),
    ),
    "coder": RolePreset(
        capability=ModelCapability.CODING,
        power=SubagentPower.MEDIUM,
        tools=(
            "read_file",
            "write_file",
            "edit_file",
            "execute_shell",
        ),
    ),
    "reviewer": RolePreset(
        capability=ModelCapability.GENERAL,
        power=SubagentPower.MEDIUM,
        tools=(
            "read_file",
            "search_files",
        ),
    ),
}


def apply_role_preset(
    input_data: SubagentRunInput,
) -> SubagentRunInput:
    preset = ROLE_PRESETS.get(input_data.role)

    if preset is None:
        return input_data

    capability = input_data.capability
    power = input_data.power
    tools = input_data.tools

    if capability is ModelCapability.GENERAL:
        capability = preset.capability

    if power is SubagentPower.AUTO:
        power = preset.power

    if not tools:
        tools = preset.tools

    if (
        capability is input_data.capability
        and power is input_data.power
        and tools == input_data.tools
    ):
        return input_data

    return SubagentRunInput(
        role=input_data.role,
        objective=input_data.objective,
        instructions=input_data.instructions,
        context=input_data.context,
        capability=capability,
        complexity=input_data.complexity,
        power=power,
        min_context_tokens=input_data.min_context_tokens,
        model=input_data.model,
        tools=tools,
        permissions=input_data.permissions,
        max_iterations=input_data.max_iterations,
        max_tool_calls=input_data.max_tool_calls,
        max_tokens=input_data.max_tokens,
        timeout_seconds=input_data.timeout_seconds,
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

    power: SubagentPower = SubagentPower.AUTO

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
        input_data = apply_role_preset(input_data)

        task = SubagentTask(
            role=input_data.role,
            objective=input_data.objective,
            instructions=input_data.instructions,
            context=input_data.context,
            model=input_data.model,
            profile=TaskProfile(
                capability=input_data.capability,
                complexity=input_data.complexity,
                power=input_data.power,
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
            "describe the task requirements. Choose compute "
            "power explicitly: power='low' for simple lookups "
            "and short extracts (cheapest, no VRAM swap), "
            "power='medium' for ordinary subtasks, power='high' "
            "for the best available model on hard work (usually "
            "the strong main model with no swap; a heavier pick "
            "forces a VRAM model swap, use deliberately). "
            "power='auto' "
            "lets the router decide. Known roles carry presets, "
            "so a bare role is enough: role='researcher' gets "
            "research capability, low power, and web tools; "
            "role='coder' gets coding, medium power, and file "
            "tools; role='reviewer' gets read tools. Explicit "
            "fields always override presets. Pass arguments "
            "flat, never nested under a 'properties' object. "
            "Group similar subtasks so "
            "they reuse one loaded model. The subagent runs "
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