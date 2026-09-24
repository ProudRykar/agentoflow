from __future__ import annotations

from typing import Any

from agent_workflow.cli.ui.state import (
    ApprovalView,
    MessageRole,
    UIState,
)


class UIController:
    def __init__(
        self,
        state: UIState,
    ) -> None:
        self.state = state

    def user_message(
        self,
        content: str,
    ) -> None:
        self.state.add_message(
            MessageRole.USER,
            content,
        )

    def assistant_message(
        self,
        content: str,
    ) -> None:
        self.state.add_message(
            MessageRole.ASSISTANT,
            content,
        )

    def agent_started(
        self,
        max_iterations: int,
    ) -> None:
        self.state.thinking = True
        self.state.max_iterations = max_iterations
        self.state.error = None

    def agent_finished(self) -> None:
        self.state.thinking = False
        self.state.clear_approval()

    def llm_requested(
        self,
        iteration: int,
    ) -> None:
        self.state.thinking = True
        self.state.iteration = iteration

    def llm_responded(self) -> None:
        pass

    def tool_started(
        self,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> None:
        self.state.add_tool(
            call_id=call_id,
            name=name,
            arguments=arguments,
        )

    def tool_finished(
        self,
        call_id: str,
        output: str | None,
        error: str | None,
    ) -> None:
        self.state.finish_tool(
            call_id=call_id,
            output=output,
            error=error,
        )

    def approval_requested(
        self,
        tool_name: str,
        permission: str,
        reason: str,
        arguments: dict[str, Any],
    ) -> None:
        self.state.active_approval = ApprovalView(
            tool_name=tool_name,
            permission=permission,
            reason=reason,
            arguments=arguments,
        )

    def approval_finished(self) -> None:
        self.state.clear_approval()

    def set_error(
        self,
        message: str,
    ) -> None:
        self.state.error = message
        self.state.thinking = False