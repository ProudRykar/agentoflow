from __future__ import annotations

import inspect
import time

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import Any

from core.context.checkpoint import build_checkpoint
from core.context.context_assembler import ContextAssembler
from core.context.context_controller import ContextController
from core.context.context_item import ContextItem
from core.context.evidence import EvidenceReceipt
from core.context.history import HistoryKind
from core.context.memory import retrieve_snapshot
from core.context.synthesis import build_synthesis_guide
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
from core.entities.models.llm import (
    LLMResponse,
    LLMToolCall,
)
from core.entities.models.llm_client import LLMClient
from core.entities.models.memory_manager import MemoryManager
from core.entities.models.research_contract import ResearchResult
from core.entities.models.task_contract import TaskContract
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


EventCallback = Callable[
    [AgentEvent],
    Awaitable[None],
]


class Agent:
    """
    Runtime executor for the agent harness.

    Lifecycle:

        run()
            = new task + new anchor + new execution,
              conversation is cleared

        continue_run()
            = new task + new anchor + new execution,
              conversation is preserved

        resume_run()
            = same task + same anchor + same execution

    A prepared task (Planner -> TaskAnchor + TaskPlan ->
    orchestrator.prepare_task()) is reused, never recreated.
    Without a prepared task, a fallback anchor is built from
    the prompt.

    Task preparation:

        UI / caller
            ↓
        Planner
            ↓
        TaskPlan + TaskContract
            ↓
        orchestrator.prepare_task()
            ↓
        Agent.run() / continue_run()
            ↓
        orchestrator.on_agent_started(prompt)

    Agent is responsible for execution.

    It must not re-plan or reset a task that has already been
    prepared by the caller.
    """

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
        assembler: ContextAssembler | None = None,
        controller: ContextController | None = None,
        memory: MemoryManager | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._executor = executor

        self._definition_builder = (
            definition_builder
            or ToolDefinitionBuilder()
        )

        self._context_manager = (
            context_manager
            or ContextManager()
        )

        self._max_iterations = max_iterations
        self._max_tool_calls = max_tool_calls
        self._allowed_tools = allowed_tools
        self._hooks = tuple(hooks)

        self._run_id = run_id
        self._parent_run_id = parent_run_id
        self._agent_id = agent_id
        self._role = role
        self._model = model

        self._orchestrator = (
            orchestrator
            or AgentOrchestrator()
        )

        self._assembler = (
            assembler
            or ContextAssembler()
        )

        self._controller = (
            controller
            or ContextController(
                budget=self._assembler.budget,
                counter=self._assembler.counter,
            )
        )

        self._memory = memory

    # ==================================================================
    # Properties
    # ==================================================================

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
    def assembler(self) -> ContextAssembler:
        return self._assembler

    @property
    def controller(self) -> ContextController:
        return self._controller

    @property
    def memory(self) -> MemoryManager | None:
        return self._memory

    async def _memory_snapshot(self) -> tuple[ContextItem, ...]:
        if self._memory is None:
            return ()

        anchor = self._orchestrator.task_anchor

        if anchor is None:
            return ()

        entries = await self._memory.all()

        return retrieve_snapshot(
            entries,
            f"{anchor.original_prompt} {anchor.objective}",
        ).items

    def _record_history(
        self,
        kind: HistoryKind,
        content: str,
        reference: str | None = None,
    ) -> None:
        anchor = self._orchestrator.task_anchor

        if anchor is None:
            return

        self._controller.record(
            task_id=anchor.task_id,
            run_id=self._run_id,
            kind=kind,
            content=content,
            reference=reference,
        )

    @property
    def task_anchor(self):
        return self._orchestrator.task_anchor

    @property
    def state(self):
        return self._orchestrator.state

    # ==================================================================
    # Task preparation
    # ==================================================================

    def _prepare_task_contract(
        self,
        task_contract: TaskContract | None,
    ) -> None:
        """
        Prepare the contract only when the caller has not already
        prepared a complete task.

        The normal UI flow is:

            Planner.plan()
                ↓
            orchestrator.prepare_task()
                ↓
            Agent.run()

        In that flow, calling set_task_contract() here would reset
        runtime progress and execution plan.

        Direct Agent usage is still supported: when no prepared task
        exists, the supplied contract is installed and the
        orchestrator can build the TaskPlan from the prompt.
        """

        if task_contract is None:
            return

        task_prepared = bool(
            getattr(
                self._orchestrator,
                "_task_plan_prepared",
                False,
            ),
        )

        if task_prepared:
            current_contract = getattr(
                self._orchestrator,
                "task_contract",
                None,
            )

            if (
                current_contract is not None
                and current_contract != task_contract
            ):
                raise ValueError(
                    "Agent received a TaskContract that differs "
                    "from the already prepared task contract",
                )

            return

        self._orchestrator.set_task_contract(
            task_contract,
        )

    # ==================================================================
    # Events
    # ==================================================================

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

        self._record_history(
            HistoryKind.PHASE_CHANGE,
            f"{transition.previous_phase.value} -> "
            f"{transition.phase.value}: {transition.reason}",
        )

    # ==================================================================
    # Hooks
    # ==================================================================

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

    # ==================================================================
    # Public execution
    # ==================================================================

    async def run(
        self,
        prompt: str,
        context: ToolContext,
        on_event: EventCallback | None = None,
        task_contract: TaskContract | None = None,
    ) -> str:
        """
        Start a new conversation run.

        ContextManager starts a new context.

        Task preparation must normally happen before this method.
        The method still accepts task_contract for direct usage.
        """

        self._prepare_task_contract(
            task_contract,
        )

        self._orchestrator.set_run_id(
            self._run_id,
        )

        # Reuse the prepared anchor when one exists;
        # otherwise build a fallback anchor from the prompt.
        self._orchestrator.ensure_task_anchor(
            prompt,
            task_plan=self._orchestrator.task_plan,
        )

        transition = self._orchestrator.on_agent_started(
            prompt=prompt,
            run_id=self._run_id,
        )

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

        self._context_manager.create(
            prompt,
        )

        # New conversation starts a new history window.
        self._controller.new_window()

        self._record_history(
            HistoryKind.USER_MESSAGE,
            prompt,
        )

        return await self._run_loop(
            context=replace(
                context,
                event_callback=on_event,
                run_id=self._run_id,
                parent_run_id=self._parent_run_id,
            ),
            on_event=on_event,
        )

    async def continue_run(
        self,
        prompt: str,
        context: ToolContext,
        on_event: EventCallback | None = None,
        task_contract: TaskContract | None = None,
    ) -> str:
        """
        Start a NEW execution run while preserving conversation context.

        Every call is a NEW task with a NEW anchor: the previous
        anchor stays in history, it never leaks into the new task.

        This is deliberately NOT on_agent_resumed().
        """

        self._prepare_task_contract(
            task_contract,
        )

        self._orchestrator.set_run_id(
            self._run_id,
        )

        self._orchestrator.ensure_task_anchor(
            prompt,
            task_plan=self._orchestrator.task_plan,
            force_new=True,
        )

        transition = self._orchestrator.on_agent_started(
            prompt=prompt,
            run_id=self._run_id,
        )

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

        self._context_manager.add_user_message(
            prompt,
        )

        # Same conversation window, new task.
        self._record_history(
            HistoryKind.USER_MESSAGE,
            prompt,
        )

        return await self._run_loop(
            context=replace(
                context,
                event_callback=on_event,
                run_id=self._run_id,
                parent_run_id=self._parent_run_id,
            ),
            on_event=on_event,
        )

    async def resume_run(
        self,
        context: ToolContext,
        on_event: EventCallback | None = None,
    ) -> str:
        """
        Resume an actually paused/blocked run.

        Same task, same anchor, same execution: nothing is
        recreated here.

        This is the ONLY public method that calls
        on_agent_resumed().
        """

        self._orchestrator.set_run_id(
            self._run_id,
        )

        if self._orchestrator.task_anchor is None:
            raise RuntimeError(
                "Cannot resume a run without a prepared task",
            )

        transition = self._orchestrator.on_agent_resumed()

        await self._emit_phase_transition(
            transition,
            on_event,
        )

        return await self._run_loop(
            context=replace(
                context,
                event_callback=on_event,
                run_id=self._run_id,
                parent_run_id=self._parent_run_id,
            ),
            on_event=on_event,
        )

    def clear_context(self) -> None:
        self._context_manager.clear()

    # ==================================================================
    # Tools
    # ==================================================================

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

    # ==================================================================
    # Runtime semantic results
    # ==================================================================

    def _record_research_evidence(
        self,
        tool_name: str,
        result: ToolResult,
    ) -> EvidenceReceipt | None:
        """
        Register a successful research tool result in the
        EvidenceStore and return a compact receipt for the
        conversation.

        The full page bodies stay in the store and never enter
        the dialogue. Non-research tool results return None.
        """

        del tool_name

        if result.error is not None:
            return None

        output = result.output

        if not output:
            return None

        try:
            research_result = ResearchResult.from_json(
                output,
            )
        except (TypeError, ValueError):
            return None

        if research_result is None:
            return None

        return self._orchestrator.store_research_evidence(
            research_result,
        )

    def _synthesis_revision(
        self,
        result: str,
        revised: bool,
    ) -> str | None:
        """
        Decide whether a candidate final response needs one
        more guided pass before release.

        Blocking and one-shot: research drafts are revised
        against the synthesis guide, drafts with code must
        pass verification. Returns the instruction for the
        extra pass, or None when the draft may release.
        """

        if revised:
            return None

        needs_guide = (
            self._orchestrator.task_contract.requires_research
        )
        needs_verify = "```" in result

        if not needs_guide and not needs_verify:
            return None

        parts: list[str] = []

        if needs_guide:
            guide = build_synthesis_guide(
                self._orchestrator.task_state,
            )

            parts.append(
                guide.render()
                + "\nRevise your previous draft to satisfy "
                "this guide. Keep every verified fact."
            )

        if needs_verify:
            parts.append(
                "VERIFICATION REQUIRED before release: "
                "re-check every API signature, code pattern, "
                "and command in your draft against the "
                "[EVIDENCE] excerpts above. Fix each mismatch "
                "or remove the snippet. Do not invent API "
                "shapes from prior knowledge."
            )

        return "\n\n".join(parts)

    async def _advance_after_semantic_progress(
        self,
        reason: str,
        on_event: EventCallback | None,
    ) -> None:
        transition = self._orchestrator.advance_plan(
            reason=reason,
        )

        await self._emit_phase_transition(
            transition,
            on_event,
        )

    # ==================================================================
    # Main loop
    # ==================================================================

    async def _run_loop(
        self,
        context: ToolContext,
        on_event: EventCallback | None,
    ) -> str:
        if self._orchestrator.task_anchor is None:
            raise RuntimeError(
                "Cannot run without a TaskAnchor",
            )

        tools = self._build_tools()

        tool_calls_used = 0

        completion_gate_instruction: str | None = None

        guide_revision_done = False

        for iteration in range(
            1,
            self._max_iterations + 1,
        ):
            # ----------------------------------------------------------
            # Phase: LLM requested
            # ----------------------------------------------------------

            await self._emit_phase_transition(
                self._orchestrator.on_llm_requested(
                    iteration,
                ),
                on_event,
            )

            # ----------------------------------------------------------
            # Assemble the LLM view: anchor + state + execution +
            # checkpoint + research + memory + recent conversation.
            # The model never sees raw runtime objects.
            # ----------------------------------------------------------

            history_selected: tuple[ContextItem, ...] = ()

            memory_snapshot = await self._memory_snapshot()

            request = self._assembler.build(
                anchor=self._orchestrator.task_anchor,
                task_state=self._orchestrator.task_state,
                execution=(
                    self._orchestrator.execution_context
                ),
                conversation_recent=tuple(
                    self._context_manager.dialogue()
                ),
                evidence_selected=(
                    self._orchestrator.evidence_store.all()
                ),
                memory_snapshot=memory_snapshot,
                research=self._orchestrator.research_context,
                completion_instruction=(
                    completion_gate_instruction
                ),
                checkpoint=build_checkpoint(
                    self._orchestrator,
                ),
                execution_plan=self._orchestrator.plan,
                history_selected=history_selected,
                tools=tools,
            )

            # ------------------------------------------------------
            # Budget gate: at most one rollover per iteration.
            # The rebuilt request carries a one-shot history
            # selection from the closed window.
            # ------------------------------------------------------

            if self._controller.is_over_budget(request):
                rollover = self._controller.rollover(
                    self._orchestrator,
                    self._context_manager,
                )

                history_selected = (
                    rollover.history_context
                )

                request = self._assembler.build(
                    anchor=self._orchestrator.task_anchor,
                    task_state=self._orchestrator.task_state,
                    execution=(
                        self._orchestrator.execution_context
                    ),
                    conversation_recent=tuple(
                        self._context_manager.dialogue()
                    ),
                    evidence_selected=(
                        self._orchestrator.evidence_store.all()
                    ),
                    memory_snapshot=memory_snapshot,
                    research=self._orchestrator.research_context,
                    completion_instruction=(
                        completion_gate_instruction
                    ),
                    checkpoint=rollover.checkpoint,
                    execution_plan=self._orchestrator.plan,
                    history_selected=history_selected,
                    tools=tools,
                )

            messages = request.to_message_list()

            await self._emit(
                LLMRequested(
                    iteration=iteration,
                    message_count=len(messages),
                    tool_count=len(tools),
                    run_id=self._run_id,
                    parent_run_id=self._parent_run_id,
                    estimated_tokens=(
                        self._controller.request_tokens(
                            request,
                        )
                    ),
                ),
                on_event,
            )

            # ----------------------------------------------------------
            # Before LLM hooks
            # ----------------------------------------------------------

            await self._before_llm(
                iteration=iteration,
                messages=messages,
                tools=tools,
            )

            # ----------------------------------------------------------
            # LLM
            # ----------------------------------------------------------

            response = await self._stream_llm(
                iteration=iteration,
                messages=messages,
                tools=tools,
                on_event=on_event,
            )

            # ----------------------------------------------------------
            # After LLM hooks
            # ----------------------------------------------------------

            await self._after_llm(
                iteration=iteration,
                response=response,
            )

            # ----------------------------------------------------------
            # Candidate final response
            # ----------------------------------------------------------

            if not response.tool_calls:
                result = response.content or ""

                if not result.strip():
                    completion_gate_instruction = (
                        "The previous response was empty. "
                        "Continue working on the current plan."
                    )
                    continue

                # ------------------------------------------------------
                # Guided revision (at most one extra pass).
                #
                # Research drafts are revised against the
                # synthesis guide; drafts with code must pass a
                # verification pass before release (blocking).
                # The draft is appended to the dialogue so the
                # next request can actually revise it.
                # ------------------------------------------------------

                revision = self._synthesis_revision(
                    result,
                    guide_revision_done,
                )

                if revision is not None:
                    guide_revision_done = True

                    self._context_manager.add_assistant_message(
                        {
                            "role": "assistant",
                            "content": result,
                        },
                    )

                    completion_gate_instruction = revision
                    continue

                # ------------------------------------------------------
                # Synthesis
                # ------------------------------------------------------

                synthesis_transition = (
                    self._orchestrator.begin_synthesis()
                )

                await self._emit_phase_transition(
                    synthesis_transition,
                    on_event,
                )

                # ------------------------------------------------------
                # Completion
                # ------------------------------------------------------

                completion_transition = (
                    self._orchestrator.complete()
                )

                await self._emit_phase_transition(
                    completion_transition,
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

                    self._record_history(
                        HistoryKind.ASSISTANT_MESSAGE,
                        result,
                    )

                    return result

                completion = (
                    self._orchestrator.check_completion()
                )

                missing = ", ".join(
                    completion.missing,
                )

                next_phase = (
                    completion.next_phase.value
                    if completion.next_phase is not None
                    else "unknown"
                )

                completion_gate_instruction = (
                    "Completion was rejected by the runtime "
                    "completion gate. "
                    "Do not claim that the task is complete. "
                    "Continue the current plan. "
                    f"Missing requirements: {missing}. "
                    f"Required next phase: {next_phase}."
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

            completion_gate_instruction = None

            # ----------------------------------------------------------
            # Execute tools
            # ----------------------------------------------------------

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

                # ------------------------------------------------------
                # Phase: tool started
                # ------------------------------------------------------

                await self._emit_phase_transition(
                    self._orchestrator.on_tool_started(
                        tool_call.name,
                    ),
                    on_event,
                )

                # ------------------------------------------------------
                # ToolStarted
                #
                # ToolStarted DOES contain arguments.
                # ------------------------------------------------------

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

                self._record_history(
                    HistoryKind.TOOL_CALL,
                    f"{tool_call.name} "
                    f"arguments={tool_call.arguments}",
                    reference=tool_call.id,
                )

                # ------------------------------------------------------
                # Before-tool hooks
                # ------------------------------------------------------

                tool_started_at = time.monotonic()

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
                    # --------------------------------------------------
                    # Actual tool execution
                    # --------------------------------------------------

                    result = await self._executor.execute(
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                        context=context,
                    )

                # ------------------------------------------------------
                # After-tool hooks
                #
                # AgentHook.after_tool() does NOT receive context.
                # ------------------------------------------------------

                await self._after_tool(
                    iteration=iteration,
                    tool_call=tool_call,
                    result=result,
                )

                # ------------------------------------------------------
                # ToolFinished
                #
                # ToolFinished does NOT contain arguments.
                # Arguments are already represented by ToolStarted.
                # ------------------------------------------------------

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
                        error_message=(
                            result.error.message
                            if result.error is not None
                            else None
                        ),
                        run_id=self._run_id,
                        parent_run_id=self._parent_run_id,
                        duration_seconds=(
                            time.monotonic() - tool_started_at
                        ),
                    ),
                    on_event,
                )

                # ------------------------------------------------------
                # Research evidence
                #
                # Research tool output is registered in the
                # EvidenceStore. Only a compact receipt enters
                # the conversation; page bodies never do.
                # ------------------------------------------------------

                receipt = self._record_research_evidence(
                    tool_call.name,
                    result,
                )

                # ------------------------------------------------------
                # Tool finished -> orchestrator
                # ------------------------------------------------------

                await self._emit_phase_transition(
                    self._orchestrator.on_tool_finished(
                        error_code=(
                            result.error.code
                            if result.error is not None
                            else None
                        ),
                        error_message=(
                            result.error.message
                            if result.error is not None
                            else None
                        ),
                    ),
                    on_event,
                )

                # ------------------------------------------------------
                # Semantic progress
                # ------------------------------------------------------

                if (
                    receipt is not None
                    and receipt.satisfied
                ):
                    await self._advance_after_semantic_progress(
                        reason=(
                            "research contract satisfied by "
                            "runtime research result"
                        ),
                        on_event=on_event,
                    )

                # ------------------------------------------------------
                # Tool result -> ConversationManager + History
                # ------------------------------------------------------

                if receipt is not None:
                    receipt_text = receipt.to_text(
                        tool_call.name,
                    )

                    self._context_manager.add_tool_result(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": receipt_text,
                        },
                    )

                    self._record_history(
                        HistoryKind.TOOL_RESULT,
                        receipt_text,
                        reference=tool_call.id,
                    )

                    self._record_history(
                        HistoryKind.EVIDENCE_REF,
                        "evidence: "
                        + ", ".join(receipt.evidence_ids),
                        reference=tool_call.id,
                    )
                else:
                    self._context_manager.add_tool_result(
                        self._tool_message(
                            tool_call_id=tool_call.id,
                            result=result,
                        ),
                    )

                    if result.error is not None:
                        history_content = (
                            f"Tool error [{result.error.code}]: "
                            f"{result.error.message}"
                        )
                    else:
                        history_content = result.output or ""

                    self._record_history(
                        HistoryKind.TOOL_RESULT,
                        history_content,
                        reference=tool_call.id,
                    )

        # ==================================================================
        # Maximum iterations
        # ==================================================================

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

    # ==================================================================
    # LLM
    # ==================================================================

    async def _stream_llm(
        self,
        iteration: int,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        on_event: EventCallback | None,
    ) -> LLMResponse:
        chat_stream_method = getattr(
            type(self._llm),
            "chat_stream",
            None,
        )

        # LLMClient.chat_stream() is the base implementation.
        # If the concrete client hasn't overridden it, use chat().
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
            raw_chunks.append(
                chunk,
            )

            message = chunk.get(
                "message",
                {},
            )

            # ----------------------------------------------------------
            # Thinking
            # ----------------------------------------------------------

            thinking = message.get(
                "thinking",
            )

            if thinking:
                text = str(thinking)

                thinking_parts.append(
                    text,
                )

                await self._emit(
                    LLMThinkingChunk(
                        iteration=iteration,
                        content=text,
                        run_id=self._run_id,
                        parent_run_id=self._parent_run_id,
                    ),
                    on_event,
                )

            # ----------------------------------------------------------
            # Content
            # ----------------------------------------------------------

            content = message.get(
                "content",
            )

            if content:
                text = str(content)

                content_parts.append(
                    text,
                )

                await self._emit(
                    LLMContentChunk(
                        iteration=iteration,
                        content=text,
                        run_id=self._run_id,
                        parent_run_id=self._parent_run_id,
                    ),
                    on_event,
                )

            # ----------------------------------------------------------
            # Tool calls
            # ----------------------------------------------------------

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

        if inspect.isawaitable(
            response,
        ):
            response = await response

        if not isinstance(
            response,
            LLMResponse,
        ):
            raise TypeError(
                f"{type(self._llm).__name__}.chat() returned "
                f"{type(response).__name__}, expected LLMResponse"
            )

        # --------------------------------------------------------------
        # Thinking
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Content
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Response
        # --------------------------------------------------------------

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

    # ==================================================================
    # Messages
    # ==================================================================

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