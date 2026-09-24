from __future__ import annotations

import asyncio
from typing import Any

from agent_workflow.core.entities.models.arguments import (
    ArgumentDecoder,
    ArgumentDecoderError,
)
from agent_workflow.core.entities.models.builtin.execute_shell import (
    ShellCommandError,
)
from agent_workflow.core.entities.models.tool import Tool, ToolContext
from agent_workflow.core.entities.models.tool_registry import ToolRegistry
from agent_workflow.core.entities.models.tool_result import ToolError, ToolResult


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        argument_decoder: ArgumentDecoder | None = None,
    ) -> None:
        self._registry = registry
        self._argument_decoder = (
            argument_decoder
            or ArgumentDecoder()
        )

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        try:
            tool = self._registry.get(
                tool_name,
            )
        except Exception as exc:
            return ToolResult(
                error=ToolError(
                    message=str(exc),
                    code="tool_not_found",
                    retryable=False,
                ),
            )

        try:
            self._check_permissions(
                tool,
                context,
            )
        except PermissionError as exc:
            return ToolResult(
                error=ToolError(
                    message=str(exc),
                    code="permission_denied",
                    retryable=False,
                ),
            )

        try:
            decoded_arguments = (
                self._argument_decoder.decode(
                    arguments,
                    tool.input_type,
                )
            )
        except ArgumentDecoderError as exc:
            return ToolResult(
                error=ToolError(
                    message=str(exc),
                    code=exc.code,
                    retryable=False,
                ),
            )

        try:
            output = await asyncio.wait_for(
                tool.handler(
                    decoded_arguments,
                    context,
                ),
                timeout=tool.policy.timeout,
            )

        except asyncio.TimeoutError:
            return ToolResult(
                error=ToolError(
                    message=(
                        f"Tool '{tool_name}' timed out after "
                        f"{tool.policy.timeout} seconds"
                    ),
                    code="timeout",
                    retryable=True,
                ),
            )

        except ShellCommandError as exc:
            return ToolResult(
                error=ToolError(
                    message=str(exc),
                    code="command_failed",
                    retryable=True,
                ),
            )

        except Exception as exc:
            return ToolResult(
                error=ToolError(
                    message=str(exc),
                    code="execution_error",
                    retryable=True,
                ),
            )

        output_text = str(output)

        if len(output_text) > tool.policy.max_output_size:
            return ToolResult(
                error=ToolError(
                    message=(
                        f"Tool '{tool_name}' output exceeds "
                        f"the maximum size of "
                        f"{tool.policy.max_output_size} characters"
                    ),
                    code="output_too_large",
                    retryable=False,
                ),
            )

        return ToolResult(
            output=output_text,
        )

    @staticmethod
    def _check_permissions(
        tool: Tool[Any, Any],
        context: ToolContext,
    ) -> None:
        available_permissions = (
            context.permissions
            | context.approved_permissions
        )

        missing = (
            tool.policy.permissions
            - available_permissions
        )

        if missing:
            raise PermissionError(
                "Missing permissions: "
                + ", ".join(sorted(missing))
            )