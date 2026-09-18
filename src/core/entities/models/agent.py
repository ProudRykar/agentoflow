from __future__ import annotations

import inspect

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import Any

from core.entities.models.agent_hook import AgentHook
from core.entities.models.agent_orchestrator import (
    AgentOrchestrator,
    PhaseTransition,
)
from core.entities.models.agent_trace import (
    AgentEvent,
    AgentFinished,
    AgentPhaseChanged,
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
from core.entities.models.tool import (
    ToolContext,
    ToolError,
    ToolResult,
)
from core.entities.models.tool_definition import ToolDefinition
from core.entities.models.tool_definition_builder import (
    ToolDefinitionBuilder,
)
from core.entities.models.tool_executor import ToolExecutor
from core.entities.models.tool_registry import ToolRegistry
from core.entities.models.task_contract import TaskContract
from core.entities.models.research_contract import (
    ResearchResult,
)
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
        max_tool_calls: int | None = None,
        allowed_tools: frozenset[str] | None = None,
        hooks: Sequence[AgentHook] = (),
        *,
        run_id: str = "",
        parent_run_id: str | None = None,
        agent_id: str = "main",
        role: str = "main",
        model: str = "",
        orchestrator: AgentOrchestrator | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._executor = executor

        self._definition_builder = (
            definition_builder
            or ToolDefinitionBuilder()
        )

        self._max_iterations = max_iterations
        self._max_tool_calls = max_tool_calls
        self._allowed_tools = allowed_tools
        self._hooks = tuple(hooks)

        self._context_manager = (
            context_manager
            or ContextManager()
        )

        self._run_id = run_id
        self._parent_run_id = parent_run_id
        self._agent_id = agent_id
        self._role = role
        self._model = model

        self._orchestrator = (
            orchestrator
            or AgentOrchestrator()
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def parent_run_id(self) -> str | None:
        return self._parent_run_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def role(self) -> str:
        return self._role

    @property
    def model(self) -> str:
        return self._model

    @property
    def orchestrator(self) -> AgentOrchestrator:
        return self._orchestrator

    @property
    def state(self):
        return self._orchestrator.state

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    async def _emit(
        self,
        event: AgentEvent,
        on_event: EventCallback | None,
    ) -> None:
        if on_event is not None:
            await on_event(event)

    async def _emit_phase_transition(
        self,
        transition: PhaseTransition | None,
        on_event: EventCallback | None,
    ) -> None:
        if transition is None:
            return

        await self._emit(
            AgentPhaseChanged(
                previous_phase=transition.previous_phase,
                phase=transition.phase,
                reason=transition.reason,
                iteration=self.state.iteration,
                run_id=self._run_id,
                parent_run_id=self._parent_run_id,
            ),
            on_event,
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Public execution
    # ------------------------------------------------------------------

    def _consume_runtime_tool_result(
        self,
        result: ToolResult,
    ) -> None:
        if result.error is not None:
            return

        research_result = (
            ResearchResult.from_json(
                result.output,
            )
        )

        if research_result is None:
            return

        self._orchestrator.record_research_result(
            research_result,
        )

    async def run(
        self,
        prompt: str,
        context: ToolContext,
        on_event: EventCallback | None = None,
        task_contract: TaskContract | None = None,
    ) -> str:
        if self._orchestrator is not None:
            self._orchestrator.set_task_contract(
                task_contract or TaskContract(),
            )

            transition = self._orchestrator.on_agent_started()

            await self._emit_phase_transition(
                transition,
                on_event,
            )

        await self._emit(
            AgentStarted(
                prompt=prompt,
                run_id=self._run_id,
                parent_run_id=self._parent_run_id,
                agent_id=self._agent_id,
                role=self._role,
                model=self._model,
            ),
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
        task_contract: TaskContract | None = None,
    ) -> str:
        context = replace(
            context,
            event_callback=on_event,
            run_id=self._run_id,
            parent_run_id=self._parent_run_id,
        )

        await self._emit(
            AgentStarted(
                prompt=prompt,
                run_id=self._run_id,
                parent_run_id=self._parent_run_id,
                agent_id=self._agent_id,
                role=self._role,
                model=self._model,
            ),
            on_event,
        )

        if (
            self._orchestrator is not None
            and task_contract is not None
        ):
            self._orchestrator.set_task_contract(
                task_contract,
            )

        await self._emit_phase_transition(
            self._orchestrator.on_agent_started(),
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

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _build_tools(self) -> tuple[ToolDefinition, ...]:
        definitions: list[ToolDefinition] = []

        for tool in self._registry.all():
            if (
                self._allowed_tools is not None
                and tool.name not in self._allowed_tools
            ):
                continue

            definitions.append(
                self._definition_builder.build(
                    tool,
                ),
            )

        return tuple(definitions)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run_loop(
        self,
        context: ToolContext,
        on_event: EventCallback | None,
    ) -> str:
        tools = self._build_tools()
        tool_calls_used = 0
        completion_gate_instruction: str | None = None

        for iteration in range(
            1,
            self._max_iterations + 1,
        ):
            messages = self._context_manager.messages()

            if completion_gate_instruction is not None:
                messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": completion_gate_instruction,
                    },
                ]

            await self._emit_phase_transition(
                self._orchestrator.on_llm_requested(
                    iteration,
                ),
                on_event,
            )

            await self._emit(
                LLMRequested(
                    iteration=iteration,
                    message_count=len(messages),
                    tool_count=len(tools),
                    run_id=self._run_id,
                    parent_run_id=self._parent_run_id,
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

            # ----------------------------------------------------------
            # Final response
            # ----------------------------------------------------------

            if not response.tool_calls:
                result = response.content or ""

                if self._orchestrator is None:
                    await self._emit(
                        AgentFinished(
                            result=result,
                            run_id=self._run_id,
                            parent_run_id=self._parent_run_id,
                            agent_id=self._agent_id,
                            role=self._role,
                            model=self._model,
                        ),
                        on_event,
                    )

                    return result

                synthesis = self._orchestrator.begin_synthesis()

                await self._emit_phase_transition(
                    synthesis,
                    on_event,
                )

                completion = self._orchestrator.complete()

                await self._emit_phase_transition(
                    completion,
                    on_event,
                )

                if self._orchestrator.state.finished:
                    await self._emit(
                        AgentFinished(
                            result=result,
                            run_id=self._run_id,
                            parent_run_id=self._parent_run_id,
                            agent_id=self._agent_id,
                            role=self._role,
                            model=self._model,
                        ),
                        on_event,
                    )

                    return result

                completion_check = self._orchestrator.check_completion()

                completion_gate_instruction = (
                    "The runtime completion gate rejected task completion. "
                    "Do not claim that the task is complete. "
                    "Continue working until all required runtime stages are "
                    "actually satisfied. "
                    f"Missing stages: {', '.join(completion_check.missing)}. "
                    f"Required action: continue from phase "
                    f"{completion_check.next_phase.value if completion_check.next_phase else 'unknown'}."
                )

                continue

            # ----------------------------------------------------------
            # Assistant tool-call message
            # ----------------------------------------------------------

            self._context_manager.add_assistant_message(
                self._assistant_message(
                    response,
                ),
            )

            # ----------------------------------------------------------
            # Tool execution
            # ----------------------------------------------------------
            completion_gate_instruction = None
            
            for tool_call in response.tool_calls:
                if (
                    self._max_tool_calls is not None
                    and tool_calls_used >= self._max_tool_calls
                ):
                    error_message = (
                        "Agent exceeded maximum tool calls: "
                        f"{self._max_tool_calls}"
                    )

                    await self._emit_phase_transition(
                        self._orchestrator.block(
                            error_message,
                        ),
                        on_event,
                    )

                    raise RuntimeError(
                        error_message,
                    )

                tool_calls_used += 1

                await self._emit_phase_transition(
                    self._orchestrator.on_tool_started(
                        tool_call.name,
                    ),
                    on_event,
                )

                await self._emit(
                    ToolStarted(
                        iteration=iteration,
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                        run_id=self._run_id,
                        parent_run_id=self._parent_run_id,
                    ),
                    on_event,
                )

                # ------------------------------------------------------
                # Before tool hooks
                # ------------------------------------------------------

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

                # ------------------------------------------------------
                # After tool hooks
                # ------------------------------------------------------

                await self._after_tool(
                    iteration=iteration,
                    tool_call=tool_call,
                    result=result,
                )

                # ------------------------------------------------------
                # Tool event
                # ------------------------------------------------------

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
                        run_id=self._run_id,
                        parent_run_id=self._parent_run_id,
                    ),
                    on_event,
                )

                self._consume_runtime_tool_result(
                    result,
                )

                await self._emit_phase_transition(
                    self._orchestrator.on_tool_finished(
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

                # ------------------------------------------------------
                # Tool result goes back to the LLM
                # ------------------------------------------------------

                self._context_manager.add_tool_result(
                    self._tool_message(
                        tool_call_id=tool_call.id,
                        result=result,
                    ),
                )

        # --------------------------------------------------------------
        # Maximum iterations
        # --------------------------------------------------------------

        error_message = (
            "Agent exceeded maximum iterations: "
            f"{self._max_iterations}"
        )

        await self._emit_phase_transition(
            self._orchestrator.block(
                error_message,
            ),
            on_event,
        )

        raise RuntimeError(
            error_message,
        )

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    async def _stream_llm(
        self,
        iteration: int,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        on_event: EventCallback | None,
    ) -> LLMResponse:
        """
        Use native streaming when the concrete LLM provides it.

        Test/dummy LLMs may implement only chat(). In that case we
        transparently use the non-streaming fallback.
        """

        chat_stream_method = getattr(
            type(self._llm),
            "chat_stream",
            None,
        )

        if chat_stream_method is LLMClient.chat_stream:
            return await self._chat_llm(
                iteration=iteration,
                messages=messages,
                tools=tools,
                on_event=on_event,
            )

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
                thinking_text = str(thinking)
                thinking_parts.append(
                    thinking_text,
                )

                await self._emit(
                    LLMThinkingChunk(
                        iteration=iteration,
                        content=thinking_text,
                        run_id=self._run_id,
                        parent_run_id=self._parent_run_id,
                    ),
                    on_event,
                )

            content = message.get(
                "content",
            )

            if content:
                content_text = str(content)
                content_parts.append(
                    content_text,
                )

                await self._emit(
                    LLMContentChunk(
                        iteration=iteration,
                        content=content_text,
                        run_id=self._run_id,
                        parent_run_id=self._parent_run_id,
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
            content=(
                "".join(content_parts)
                or None
            ),
            thinking=(
                "".join(thinking_parts)
                or None
            ),
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
                run_id=self._run_id,
                parent_run_id=self._parent_run_id,
            ),
            on_event,
        )

        return response

    async def _chat_llm(
        self,
        iteration: int,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        on_event: EventCallback | None,
    ) -> LLMResponse:
        chat = getattr(
            self._llm,
            "chat",
            None,
        )

        if chat is None:
            raise TypeError(
                f"{type(self._llm).__name__} implements neither "
                "chat_stream() nor chat()"
            )

        response = chat(
            messages=messages,
            tools=tools,
        )

        if inspect.isawaitable(response):
            response = await response

        if not isinstance(
            response,
            LLMResponse,
        ):
            raise TypeError(
                f"{type(self._llm).__name__}.chat() returned "
                f"{type(response).__name__}, expected LLMResponse"
            )

        if response.thinking:
            await self._emit(
                LLMThinkingChunk(
                    iteration=iteration,
                    content=response.thinking,
                    run_id=self._run_id,
                    parent_run_id=self._parent_run_id,
                ),
                on_event,
            )

        if response.content:
            await self._emit(
                LLMContentChunk(
                    iteration=iteration,
                    content=response.content,
                    run_id=self._run_id,
                    parent_run_id=self._parent_run_id,
                ),
                on_event,
            )

        await self._emit(
            LLMResponded(
                iteration=iteration,
                content=response.content,
                thinking=response.thinking,
                tool_call_count=len(
                    response.tool_calls,
                ),
                run_id=self._run_id,
                parent_run_id=self._parent_run_id,
            ),
            on_event,
        )

        return response

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

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