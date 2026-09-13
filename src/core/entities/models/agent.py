from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from core.entities.models.agent_hook import AgentHook
from core.entities.models.agent_trace import (
    AgentEvent,
    AgentFinished,
    AgentStarted,
    LLMRequested,
    LLMResponded,
    ToolFinished,
    ToolStarted,
)
from core.entities.models.context_manager import ContextManager
from core.entities.models.guardrail_error import GuardrailDeniedError
from core.entities.models.llm import LLMResponse, LLMToolCall
from core.entities.models.llm_client import LLMClient
from core.entities.models.tool import ToolContext
from core.entities.models.tool_definition import ToolDefinition
from core.entities.models.tool_definition_builder import (
    ToolDefinitionBuilder,
)
from core.entities.models.tool_executor import ToolExecutor
from core.entities.models.tool_registry import ToolRegistry
from core.entities.models.tool_result import ToolError, ToolResult


EventCallback = Callable[[AgentEvent], Awaitable[None]]


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        executor: ToolExecutor,
        definition_builder: ToolDefinitionBuilder | None = None,
        context_manager: ContextManager | None = None,
        max_iterations: int = 10,
        hooks: Sequence[AgentHook] = (),
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._executor = executor
        self._definition_builder = (
            definition_builder or ToolDefinitionBuilder()
        )
        self._max_iterations = max_iterations
        self._hooks = tuple(hooks)
        self._context_manager = context_manager or ContextManager()
        

    async def _emit(
        self,
        event: AgentEvent,
        on_event: EventCallback | None,
    ) -> None:
        if on_event is not None:
            await on_event(event)

    async def _before_llm(
        self,
        iteration: int,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
    ) -> None:
        for hook in self._hooks:
            await hook.before_llm(
                iteration=iteration,
                messages=messages,
                tools=tools,
            )

    async def _after_llm(
        self,
        iteration: int,
        response: LLMResponse,
    ) -> None:
        for hook in self._hooks:
            await hook.after_llm(
                iteration=iteration,
                response=response,
            )

    async def _before_tool(
        self,
        iteration: int,
        tool_call: LLMToolCall,
        context: ToolContext,
    ) -> None:
        for hook in self._hooks:
            await hook.before_tool(
                iteration=iteration,
                tool_call=tool_call,
                context=context,
            )

    async def _after_tool(
        self,
        iteration: int,
        tool_call: LLMToolCall,
        result: ToolResult,
    ) -> None:
        for hook in self._hooks:
            await hook.after_tool(
                iteration=iteration,
                tool_call=tool_call,
                result=result,
            )

    async def run(
        self,
        prompt: str,
        context: ToolContext,
        on_event: EventCallback | None = None,
    ) -> str:
        await self._emit(
            AgentStarted(prompt=prompt),
            on_event,
        )

        self._context_manager.create(prompt)

        return await self._run_loop(
            context=context,
            on_event=on_event,
        )

    async def continue_run(
        self,
        prompt: str,
        context: ToolContext,
        on_event: EventCallback | None = None,
    ) -> str:
        await self._emit(
            AgentStarted(prompt=prompt),
            on_event,
        )

        self._context_manager.add_user_message(prompt)

        return await self._run_loop(
            context=context,
            on_event=on_event,
        )

    def clear_context(self) -> None:
        self._context_manager.clear()

    async def _run_loop(
        self,
        context: ToolContext,
        on_event: EventCallback | None,
    ) -> str:
        tools = tuple(
            self._definition_builder.build(tool)
            for tool in self._registry.all()
        )

        for iteration in range(1, self._max_iterations + 1):
            messages = self._context_manager.messages()

            await self._emit(
                LLMRequested(
                    iteration=iteration,
                    message_count=len(messages),
                    tool_count=len(tools),
                ),
                on_event,
            )

            await self._before_llm(
                iteration=iteration,
                messages=messages,
                tools=tools,
            )

            response = await self._llm.chat(
                messages=messages,
                tools=tools,
            )

            await self._emit(
                LLMResponded(
                    iteration=iteration,
                    content=response.content,
                    tool_call_count=len(response.tool_calls),
                ),
                on_event,
            )

            await self._after_llm(
                iteration=iteration,
                response=response,
            )

            if not response.tool_calls:
                result = response.content or ""

                await self._emit(
                    AgentFinished(result=result),
                    on_event,
                )

                return result

            self._context_manager.add_assistant_message(
                self._assistant_message(response)
            )

            for tool_call in response.tool_calls:
                await self._emit(
                    ToolStarted(
                        iteration=iteration,
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                    ),
                    on_event,
                )

                try:
                    await self._before_tool(
                        iteration=iteration,
                        tool_call=tool_call,
                        context=context,
                    )
                except GuardrailDeniedError as exc:
                    result = ToolResult(
                        error=ToolError(
                            message=str(exc),
                            code="guardrail_denied",
                            retryable=False,
                        )
                    )
                else:
                    result = await self._executor.execute(
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                        context=context,
                    )

                await self._after_tool(
                    iteration=iteration,
                    tool_call=tool_call,
                    result=result,
                )

                await self._emit(
                    ToolFinished(
                        iteration=iteration,
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        output=result.output,
                        error_code=(
                            result.error.code
                            if result.error is not None
                            else None
                        ),
                    ),
                    on_event,
                )

                self._context_manager.add_tool_result(
                    self._tool_message(
                        tool_call_id=tool_call.id,
                        result=result,
                    )
                )

        raise RuntimeError(
            "Agent exceeded maximum iterations: "
            f"{self._max_iterations}"
        )

    @staticmethod
    def _assistant_message(
        response: LLMResponse,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": response.content or "",
        }

        if response.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                }
                for tool_call in response.tool_calls
            ]

        return message

    @staticmethod
    def _tool_message(
        tool_call_id: str,
        result: ToolResult,
    ) -> dict[str, Any]:
        if result.error is not None:
            content = (
                f"Tool error [{result.error.code}]: "
                f"{result.error.message}"
            )
        else:
            content = result.output or ""

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }