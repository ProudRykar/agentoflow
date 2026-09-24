from __future__ import annotations

import shlex
from typing import Any

from agent_workflow.core.entities.models.agent_hook import AgentHook
from agent_workflow.core.entities.models.approval import (
    ApprovalDeniedError,
    ApprovalHandler,
    ApprovalRequest,
)
from agent_workflow.core.entities.models.llm import LLMResponse, LLMToolCall
from agent_workflow.core.entities.models.shell_policy import ShellPolicy
from agent_workflow.core.entities.models.tool import ToolContext
from agent_workflow.core.entities.models.tool_definition import ToolDefinition
from agent_workflow.core.entities.models.tool_result import ToolResult


class ApprovalHook(AgentHook):
    def __init__(
        self,
        handler: ApprovalHandler,
        shell_policy: ShellPolicy,
    ) -> None:
        self._handler = handler
        self._shell_policy = shell_policy

    async def before_llm(
        self,
        iteration: int,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
    ) -> None:
        return

    async def after_llm(
        self,
        iteration: int,
        response: LLMResponse,
    ) -> None:
        return

    async def before_tool(
        self,
        iteration: int,
        tool_call: LLMToolCall,
        context: ToolContext,
    ) -> None:
        permission = self._required_permission(
            tool_call,
        )

        if permission is None:
            return

        request = ApprovalRequest(
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            permission=permission,
            reason=self._reason(
                tool_call,
                permission,
            ),
        )

        approved = await self._handler(request)

        if not approved:
            raise ApprovalDeniedError(
                f"User denied permission '{permission}'",
            )

        context.approved_permissions = (
            context.approved_permissions
            | frozenset({permission})
        )

    async def after_tool(
        self,
        iteration: int,
        tool_call: LLMToolCall,
        result: ToolResult,
    ) -> None:
        return

    def _required_permission(
        self,
        tool_call: LLMToolCall,
    ) -> str | None:
        if tool_call.name == "execute_shell":
            command = tool_call.arguments.get("command")

            if not isinstance(command, str):
                return None

            try:
                argv = shlex.split(command)
            except ValueError:
                return None

            if not argv:
                return None

            classification = self._shell_policy.classify(argv[0])
            return classification.permission

        if tool_call.name == "web_fetch":
            return "web.network"

        return None

    @staticmethod
    def _reason(
        tool_call: LLMToolCall,
        permission: str,
    ) -> str:
        if tool_call.name == "execute_shell":
            command = tool_call.arguments.get("command")

            if isinstance(command, str):
                return (
                    f"Shell command '{command}' requires "
                    f"permission '{permission}'."
                )

        if tool_call.name == "web_fetch":
            url = tool_call.arguments.get("url")

            if isinstance(url, str):
                return (
                    f"Fetching URL '{url}' requires "
                    f"permission '{permission}'."
                )

        return (
            f"Tool '{tool_call.name}' requires "
            f"permission '{permission}'."
        )