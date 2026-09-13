from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from core.entities.models.agent_hook import AgentHook
from core.entities.models.agent_trace import (
    AgentEvent,
    AgentFinished,
    AgentStarted,
    LLMContentChunk,
    LLMRequested,
    LLMResponded,
    LLMThinkingChunk,
    ToolFinished,
    ToolStarted,
)
from core.entities.models.approval import ApprovalDeniedError
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


EventCallback = Callable[
    [AgentEvent],
    Awaitable[None],
]


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
            definition_builder
            or ToolDefinitionBuilder()
        )
        self._max_iterations = max_iterations
        self._hooks = tuple(hooks)
        self._context_manager = (
            context_manager
            or ContextManager()
        )

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
            AgentStarted(
                prompt=prompt,
            ),
            on_event,
        )

        self._context_manager.create(
            prompt,
        )

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
            AgentStarted(
                prompt=prompt,
            ),
            on_event,
        )

        self._context_manager.add_user_message(
            prompt,
        )

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

        for iteration in range(
            1,
            self._max_iterations + 1,
        ):
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

            response = await self._stream_llm(
                iteration=iteration,
                messages=messages,
                tools=tools,
                on_event=on_event,
            )

            await self._after_llm(
                iteration=iteration,
                response=response,
            )

            if not response.tool_calls:
                result = response.content or ""

                await self._emit(
                    AgentFinished(
                        result=result,
                    ),
                    on_event,
                )

                return result

            self._context_manager.add_assistant_message(
                self._assistant_message(
                    response,
                ),
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
                        ),
                    )
                except ApprovalDeniedError as exc:
                    result = ToolResult(
                        error=ToolError(
                            message=str(exc),
                            code="approval_denied",
                            retryable=False,
                        ),
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
                            if result.error
                            else None
                        ),
                        error_message=(
                            result.error.message
                            if result.error
                            else None
                        ),
                    ),
                    on_event,
                )

                self._context_manager.add_tool_result(
                    self._tool_message(
                        tool_call_id=tool_call.id,
                        result=result,
                    ),
                )

        raise RuntimeError(
            "Agent exceeded maximum iterations: "
            f"{self._max_iterations}"
        )

    async def _stream_llm(
        self,
        iteration: int,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        on_event: EventCallback | None,
    ) -> LLMResponse:
        thinking_parts: list[str] = []
        content_parts: list[str] = []

        tool_calls: dict[str, LLMToolCall] = {}
        raw_chunks: list[dict[str, Any]] = []

        async for chunk in self._llm.chat_stream(
            messages=messages,
            tools=tools,
        ):
            raw_chunks.append(chunk)

            message = chunk.get(
                "message",
                {},
            )

            thinking = message.get(
                "thinking",
            )

            if thinking:
                thinking = str(thinking)
                thinking_parts.append(thinking)

                await self._emit(
                    LLMThinkingChunk(
                        iteration=iteration,
                        content=thinking,
                    ),
                    on_event,
                )

            content = message.get(
                "content",
            )

            if content:
                content = str(content)
                content_parts.append(content)

                await self._emit(
                    LLMContentChunk(
                        iteration=iteration,
                        content=content,
                    ),
                    on_event,
                )

            for raw_tool_call in message.get(
                "tool_calls",
                [],
            ):
                parsed = self._llm.parse_stream_tool_call(
                    raw_tool_call,
                )

                existing = tool_calls.get(
                    parsed.id,
                )

                if existing is None:
                    tool_calls[parsed.id] = parsed
                    continue

                tool_calls[parsed.id] = LLMToolCall(
                    id=existing.id,
                    name=(
                        parsed.name
                        or existing.name
                    ),
                    arguments={
                        **existing.arguments,
                        **parsed.arguments,
                    },
                )

            if chunk.get(
                "done",
                False,
            ):
                break

        response = LLMResponse(
            content="".join(content_parts) or None,
            thinking="".join(thinking_parts) or None,
            tool_calls=tuple(
                tool_calls.values(),
            ),
            raw=(
                raw_chunks[-1]
                if raw_chunks
                else {}
            ),
        )

        await self._emit(
            LLMResponded(
                iteration=iteration,
                content=response.content,
                thinking=response.thinking,
                tool_call_count=len(
                    response.tool_calls,
                ),
            ),
            on_event,
        )

        return response

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
                for tool in response.tool_calls
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
