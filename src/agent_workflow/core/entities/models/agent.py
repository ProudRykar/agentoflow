from __future__ import annotations

from typing import TYPE_CHECKING

import inspect
import json
import time

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import Any

from agent_workflow.core.context.checkpoint import build_checkpoint
from agent_workflow.core.context.context_assembler import ContextAssembler
if TYPE_CHECKING:
    # Annotation only: the import would be circular at runtime.
    from agent_workflow.core.entities.models.skills.skill import Skill

from agent_workflow.core.context.compaction import (
    Compactor,
    render_summary_block,
)
from agent_workflow.core.context.context_controller import ContextController
from agent_workflow.core.context.context_item import ContextItem
from agent_workflow.core.context.evidence import EvidenceReceipt
from agent_workflow.core.context.history import HistoryKind


from agent_workflow.core.context.memory import (
    retrieve_snapshot,
    retrieval_query,
    visible_entries,
)
from agent_workflow.core.context.synthesis import build_synthesis_guide
from agent_workflow.core.entities.models.agent_hook import AgentHook
from agent_workflow.core.entities.models.agent_orchestrator import (
    AgentOrchestrator,
    PhaseTransition,
)
from agent_workflow.core.entities.models.agent_trace import (
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
from agent_workflow.core.entities.models.plan_event import plan_event
from agent_workflow.core.entities.models.approval import ApprovalDeniedError
from agent_workflow.core.entities.models.context_manager import ContextManager
from agent_workflow.core.entities.models.guardrail_error import GuardrailDeniedError
from agent_workflow.core.entities.models.run_usage import RunUsage
from agent_workflow.core.entities.models.event_sink import emit_to
from agent_workflow.core.entities.models.attempt_baseline import (
    AttemptBaseline,
)
from agent_workflow.core.entities.models.llm import (
    extract_usage,
    LLMResponse,
    LLMToolCall,
)
from agent_workflow.core.entities.models.llm_client import LLMClient
from agent_workflow.core.entities.models.memory_manager import MemoryManager
from agent_workflow.core.entities.models.research_contract import ResearchResult
from agent_workflow.core.entities.models.task_contract import TaskContract
from agent_workflow.core.entities.models.tool import (
    ToolContext,
    ToolError,
    ToolResult,
)
from agent_workflow.core.entities.models.tool_definition import ToolDefinition
from agent_workflow.core.entities.models.tool_definition_builder import (
    ToolDefinitionBuilder,
)
from agent_workflow.core.entities.models.tool_executor import ToolExecutor
from agent_workflow.core.entities.models.tool_registry import ToolRegistry
from agent_workflow.core.entities.models.plugins.manager import PluginManager
from agent_workflow.core.entities.models.skills.manager import SkillManager


# A callback may be synchronous or asynchronous. ``on_event=list.append``
# is the obvious thing to write, and demanding an ``Awaitable`` made it
# fail at runtime with "object NoneType can't be used in 'await'
# expression" -- a message about the wrong line entirely.
EventCallback = Callable[
    [AgentEvent],
    Awaitable[None] | None,
]

# How many recent calls the cycle check looks at. Small on purpose: a
# longer window eventually matches two calls that merely resemble each
# other from unrelated earlier work.
_CYCLE_WINDOW = 4


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
        max_identical_failures: int = 2,
        price_per_million: tuple[float, float] | None = None,
        max_prompt_tokens: int | None = None,
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
        skill_manager: SkillManager | None = None,
        plugin_manager: PluginManager | None = None,
        compactor: Compactor | None = None,
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
        self._max_identical_failures = max_identical_failures
        # (prompt, completion) per million tokens. None unless the
        # caller knows what it is paying; a guess here would be worse
        # than no number, because it looks like a bill.
        self._price_per_million = price_per_million
        # A ceiling on measured prompt tokens for one run. max_iterations
        # bounds the number of turns but not their size: a conversation
        # that grows every turn reaches an expensive run long before the
        # iteration cap, and a local run has no provider to bill.
        self._max_prompt_tokens = max_prompt_tokens
        self._max_identical_successes = max(
            2, max_identical_failures
        )
        self._call_signatures: dict[str, int] = {}
        # Successful repeats are counted separately: one is
        # legitimate re-reading, several is a loop costing money.
        self._repeated_calls: dict[str, int] = {}
        # The recent order of calls, for cycle detection. Counted per
        # call and not per signature: ping-ponging between two tools
        # never repeats one signature often enough for the counters
        # above to notice.
        self._call_sequence: list[str] = []
        self._cycles = 0
        self._max_call_cycles = 2
        # Held on the instance rather than as a loop local: the call is
        # noted after the point where the next request is shaped, and a
        # local set in one iteration is gone by the time the next one
        # reads it.
        self._cycling = False
        # Measured tokens for this run. Provider-reported only,
        # so a run with no usage data totals zero rather than
        # claiming the estimate was measured.
        self._usage = RunUsage()
        # What the run being replaced had already gathered. A retry
        # must start from the state before the discarded attempt, not
        # inherit it: evidence it produced is evidence nobody asked to
        # keep, and leaving it in satisfies a research contract with
        # work the user has thrown away.
        self._attempt_baseline: AttemptBaseline | None = None
        self._allowed_tools = allowed_tools
        self._hooks = tuple(hooks)

        # A base for run ids, not an id for all time. Every turn mints
        # its own, because one id for the whole session made a run
        # record mean "this conversation" rather than "this attempt":
        # per-turn usage could not be told apart, and a streaming flag
        # for the current turn was cleared on every previous answer.
        self._run_prefix = run_id or "run"
        self._run_id = run_id
        self._run_counter = 0
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
        self._skill_manager = skill_manager
        self._plugin_manager = plugin_manager
        # Optional: a runtime with no summariser configured keeps the
        # old behaviour of dropping the window outright.
        self._compactor = compactor

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
    def registry(self) -> ToolRegistry:
        """The live tool registry.

        Exposed read-only so observability layers (web API, CLI)
        can inventory tools without reaching into internals.
        """

        return self._registry

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
    def context_manager(self) -> ContextManager:
        """The live conversation window.

        Exposed so a restored session can repopulate it without the
        application layer reaching into a private attribute.
        """

        return self._context_manager

    @property
    def memory(self) -> MemoryManager | None:
        return self._memory

    @property
    def skill_manager(self) -> SkillManager | None:
        return self._skill_manager

    @property
    def plugin_manager(self) -> PluginManager | None:
        return self._plugin_manager

    async def skill_catalog(self) -> str:
        """Render the available skills for the system prompt.

        Returns an empty string when no skills exist, so the
        caller can skip the block entirely.
        """

        if self._skill_manager is None:
            return ""

        skills = await self._skill_manager.list_available_metadata()

        if not skills:
            return ""

        lines = [
            "AVAILABLE SKILLS:",
            "",
            "These skills are registered and can be loaded with "
            "the skills.load tool. Loading a skill adds its "
            "instructions to the context. If a skill matches "
            "the task, load it before acting; if the user asks "
            "what you can do, read this list rather than "
            "guessing.",
            "",
        ]

        for skill in sorted(
            skills,
            key=lambda item: item["name"],
        ):
            lines.append(
                f"- {skill['name']} (v{skill['version']}): "
                f"{skill['description']}"
            )

        return "\n".join(lines)

    def plugin_catalog(self) -> str:
        """Render the connected plugins for the system prompt.

        Skills are enumerated by their own catalog. Plugins need
        this because their tools are flattened into the request
        without saying which package supplied them.
        """

        if self._plugin_manager is None:
            return ""

        plugins = self._plugin_manager.registry.list()

        if not plugins:
            return ""

        lines = [
            "CONNECTED PLUGINS:",
            "",
            "These plugins are installed and loaded. A tool "
            "prefixed with a plugin name comes from that "
            "plugin. Do not guess a plugin that is absent "
            "from this list.",
            "",
        ]

        for plugin in sorted(
            plugins,
            key=lambda item: item.name,
        ):
            context = self._plugin_manager.get_plugin_context(
                plugin.name,
            )

            tools = (
                len(context.registered_tools) if context else 0
            )
            skills = (
                len(context.registered_skills) if context else 0
            )

            detail = plugin.metadata.description or ""

            state = getattr(plugin.state, "value", plugin.state)

            lines.append(
                f"- {plugin.name} (v{plugin.version}, "
                f"{state}): {detail} "
                f"provides {tools} tool(s), {skills} skill(s)"
            )

            if plugin.metadata.capabilities:
                caps = ", ".join(plugin.metadata.capabilities)
                lines.append(f"  capabilities: {caps}")

        return "\n".join(lines)

    async def _memory_snapshot(self) -> tuple[ContextItem, ...]:
        if self._memory is None:
            return ()

        anchor = self._orchestrator.task_anchor

        if anchor is None:
            return ()

        # Read once and filter here rather than in the store: the store
        # would have to know about task semantics, and the visibility
        # rule is a property of retrieval, not of storage.
        entries = visible_entries(
            await self._memory.all(),
            anchor.task_id,
        )

        if not entries:
            return ()

        return retrieve_snapshot(
            entries,
            retrieval_query(
                f"{anchor.original_prompt} {anchor.objective}",
                self._context_manager.dialogue(),
            ),
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
        await emit_to(on_event, event)

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

        # Most of these transitions are the plan moving: a tool
        # finishing advances the step, synthesis opens one, completion
        # closes the last. Emitting the plan here rather than at each
        # call site means a new transition site cannot forget to report
        # the plan, which is how the plan would end up a phase behind
        # in the UI.
        await self._emit_plan(on_event)

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

    def _capture_baseline(self) -> None:
        """Record the state a retry would have to return to."""

        coverage = (
            self._orchestrator.task_progress.research_coverage
        )

        self._attempt_baseline = AttemptBaseline(
            evidence_ids=(
                self._orchestrator.evidence_store.ids
            ),
            fetched_urls=frozenset(coverage.fetched_urls),
            failed_urls=frozenset(coverage.failed_urls),
            discovered_urls=frozenset(
                coverage.discovered_urls
            ),
            max_depth_reached=coverage.max_depth_reached,
            total_bytes=coverage.total_bytes,
            research_completed=(
                self._orchestrator.task_progress.research_completed
            ),
        )

    def _restore_baseline(self) -> None:
        """Undo the discarded attempt's research.

        Research only. The plan keeps its completed steps: a retry of
        the wording should not also throw away the research that got
        the answer right in the first place.
        """

        baseline = self._attempt_baseline

        if baseline is None:
            return

        self._orchestrator.evidence_store.retain_only(
            baseline.evidence_ids
        )

        progress = self._orchestrator.task_progress

        coverage = progress.research_coverage

        coverage.fetched_urls = set(baseline.fetched_urls)
        coverage.failed_urls = set(baseline.failed_urls)
        coverage.discovered_urls = set(
            baseline.discovered_urls
        )
        coverage.max_depth_reached = baseline.max_depth_reached
        coverage.total_bytes = baseline.total_bytes

        progress.research_completed = baseline.research_completed

    def _mint_run_id(self) -> str:
        """A run id for the turn about to start.

        Counted rather than random: a replay of the event log should
        reproduce the same ids, and a test asserting on them should not
        have to match a uuid.
        """

        self._run_counter += 1

        self._run_id = f"{self._run_prefix}-{self._run_counter}"

        return self._run_id

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

        self._mint_run_id()

        self._capture_baseline()

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

        A restored session keeps the anchor it was given: forcing a
        new one would mint a task id that no longer matches the
        restored conversation and checkpoint. A live session still
        starts a fresh task, so a previous anchor never leaks into a
        new one.

        This is deliberately NOT on_agent_resumed().
        """

        self._prepare_task_contract(
            task_contract,
        )

        self._mint_run_id()

        self._capture_baseline()

        self._orchestrator.set_run_id(
            self._run_id,
        )

        # Only a session that already has an anchor keeps it, and that
        # happens exactly when it was restored from disk.
        if self._orchestrator.task_anchor is None:
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

    async def regenerate(
        self,
        context: ToolContext,
        on_event: EventCallback | None = None,
        hint: str = "",
    ) -> str:
        """Answer the last question again, differently.

        The user asked for the answer and got one they did not want.
        Resending the same question as a new message would put both in
        the conversation and leave the model treating the discarded
        attempt as context it should stay consistent with -- so it
        argues with itself instead of trying again. The attempt is
        therefore removed first, and the same anchor is reused: same
        task, same prepared work, one more attempt at the answer.

        ``hint`` says what to change ("shorter", "cite the source").
        It is recorded as the user's turn so it takes part in later
        context selection rather than being a one-off injection.

        Raises RuntimeError when the last thing said was the user's,
        since there is then no answer to replace -- re-answering is
        ``continue_run`` with the hint, and going through here would
        silently delete the user's own message.
        """

        dropped = self._context_manager.drop_trailing_assistant()

        if not dropped:
            raise RuntimeError(
                "Nothing to regenerate: the last message is not an "
                "assistant answer"
            )

        self._orchestrator.set_run_id(self._run_id)

        # A new run on the same anchor, not a resume: the previous run
        # finished, and the orchestrator is right to refuse resuming a
        # completed one. The question is whatever the user last asked,
        # since that is what is being answered again.
        self._restore_baseline()

        self._mint_run_id()

        self._capture_baseline()

        question = self._last_user_message()

        transition = self._orchestrator.on_agent_started(
            prompt=question,
            run_id=self._run_id,
        )

        await self._emit_phase_transition(transition, on_event)

        if hint:
            self._context_manager.add_user_message(hint)
            self._record_history(
                HistoryKind.USER_MESSAGE,
                hint,
            )

        await self._emit(
            AgentStarted(
                prompt=question,
                run_id=self._run_id,
                parent_run_id=self._parent_run_id,
                agent_id=self._agent_id,
                role=self._role,
                model=self._model,
                regenerated=True,
            ),
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

    def _last_user_message(self) -> str:
        """The last thing the user said, for the run transition."""

        for message in reversed(self._context_manager.dialogue()):
            if message.get("role") == "user":
                content = message.get("content")

                if isinstance(content, str):
                    return content

        return ""

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
    # Skills
    # ==================================================================

    async def list_skills(self) -> list[dict[str, Any]]:
        """
        List available skills with compact metadata.

        Returns metadata only (name, description, version) - not full instructions.
        """
        if self._skill_manager is None:
            return []

        return await self._skill_manager.list_available_metadata()

    async def load_skill(
        self,
        name: str,
    ) -> Skill:
        """
        Load and activate a skill by name.

        The skill's full instructions become available in the agent's context
        for subsequent LLM requests.
        """
        if self._skill_manager is None:
            raise RuntimeError("Skill system not initialized")

        skill = await self._skill_manager.activate(
            name,
            run_id=self._run_id,
            parent_run_id=self._parent_run_id,
        )

        return skill

    async def _compact_window(
        self,
        *,
        instruction: str = "",
    ):
        """Summarise the window the budget is about to drop.

        Returns None when compaction is off or there is nothing worth
        summarising. A failure propagates: carrying on without a
        summary is the amnesia this replaces, and silently degrading
        would hide it.
        """

        if not self._compactor:
            return None

        messages = self._context_manager.dialogue()

        if not messages:
            return None

        return await self._compactor.compact(
            list(messages),
            instruction=instruction,
        )

    def _seed_compaction(
        self,
        compaction,
    ) -> None:
        """Put the handover note at the head of the fresh window."""

        block = render_summary_block(
            compaction.summary,
            messages_compacted=compaction.messages_compacted,
            tokens_before=compaction.tokens_before,
        )

        self._context_manager.add_message(
            {
                "role": "user",
                "content": block,
            }
        )

    def get_active_skill_instructions(self) -> str:
        """Get concatenated instructions of all active skills."""
        if self._skill_manager is None:
            return ""

        return self._skill_manager.get_active_instructions()

    def is_skill_active(self, name: str) -> bool:
        """Check if a skill is currently active."""
        if self._skill_manager is None:
            return False

        return self._skill_manager.is_active(name)

    def mark_skill_used(self, name: str) -> None:
        """Mark a skill as used in the current run."""
        if self._skill_manager is None:
            return

        self._skill_manager.mark_used(
            name,
            run_id=self._run_id,
            parent_run_id=self._parent_run_id,
        )

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

    async def _emit_plan(
        self,
        on_event: EventCallback | None,
    ) -> None:
        """Publish the plan, or say there is none.

        Emitting unconditionally would flood the socket with a no-op on
        every iteration for runs that never planned anything, so an
        absent plan returns quietly.
        """

        plan = self._orchestrator.plan

        if plan is None:
            return

        await self._emit(
            plan_event(
                plan,
                self._run_id,
                self._parent_run_id,
                todo_list=self._orchestrator.todo_list,
            ),
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
        repeat_detected = False
        repeated_success = False
        self._usage = RunUsage()

        completion_gate_instruction: str | None = None
        consolidation_note: str | None = None

        guide_revision_done = False

        for iteration in range(
            1,
            self._max_iterations + 1,
        ):
            # ----------------------------------------------------------
            # Budget.
            #
            # Checked before the request, not after: the whole point is
            # to avoid spending the tokens that would take it over.
            # Stops with the run's own accounting attached, because
            # "stopped for budget" without a number is not actionable.
            # ----------------------------------------------------------

            if self._budget_exceeded():
                return await self._stop_for_budget(
                    on_event,
                )

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

            skill_catalog = await self.skill_catalog()

            plugin_catalog = self.plugin_catalog()

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
                consolidation_note=consolidation_note,
                completion_instruction=(
                    completion_gate_instruction
                ),
                checkpoint=build_checkpoint(
                    self._orchestrator,
                ),
                execution_plan=self._orchestrator.plan,
                todo_list=self._orchestrator.todo_list,
                history_selected=history_selected,
                tools=tools,
                skill_instructions=self.get_active_skill_instructions(),
                skill_catalog=skill_catalog,
                plugin_catalog=plugin_catalog,
            )

            # ------------------------------------------------------
            # Budget gate: at most one rollover per iteration.
            # The rebuilt request carries a one-shot history
            # selection from the closed window.
            # ------------------------------------------------------

            if self._controller.is_over_budget(request):
                # Compacted before the window is cleared. Rollover on
                # own throws the dialogue away, so a long run
                # reaches its limit and then behaves as if it had just
                # started. The note is written first, while the
                # messages it describes still exist.
                compaction = await self._compact_window(
                    instruction=_trailing_text(request),
                )

                rollover = self._controller.rollover(
                    self._orchestrator,
                    self._context_manager,
                )

                if compaction is not None and compaction.summary:
                    self._seed_compaction(compaction)

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
                    consolidation_note=consolidation_note,
                completion_instruction=(
                        completion_gate_instruction
                    ),
                    checkpoint=rollover.checkpoint,
                    execution_plan=self._orchestrator.plan,
                    todo_list=self._orchestrator.todo_list,
                    history_selected=history_selected,
                    tools=tools,
                    skill_instructions=self.get_active_skill_instructions(),
                    skill_catalog=skill_catalog,
                    plugin_catalog=plugin_catalog,
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
                    context_limit=self._controller.budget.available,
                    context_trimmed=request.trimmed,
                    counter_exact=getattr(
                        self._controller.counter,
                        "exact",
                        True,
                    ),
                    dropped_blocks=tuple(
                        request.dropped_blocks or ()
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

                # Asked once, not on every synthesis turn: repeated, it
                # turns "record what you learned" into something the
                # model satisfies by writing a note per turn until the
                # store is noise.
                if synthesis_transition is not None:
                    consolidation_note = CONSOLIDATION_NOTE

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
                    # The finished answer goes into the conversation.
                    # Without it the next turn shows the model two
                    # consecutive user messages and no trace of its own
                    # reply, so "why?" and "shorten that" have nothing to
                    # refer to -- and there is no answer to regenerate.
                    self._context_manager.add_assistant_message(
                        {
                            "role": "assistant",
                            "content": result,
                        },
                    )

                    await self._emit(
                        AgentFinished(
                            result=result,
                            run_id=self._run_id,
                            parent_run_id=self._parent_run_id,
                            agent_id=self._agent_id,
                            role=self._role,
                            model=self._model,
                            prompt_tokens=self._usage.prompt_tokens,
                            completion_tokens=(
                                self._usage.completion_tokens
                            ),
                            llm_calls=self._usage.calls,
                            llm_calls_without_usage=(
                                self._usage.calls_without_usage
                            ),
                            estimated_cost=self._usage.estimate_cost(
                                self._price_per_million
                            ),
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

            # Once the same call has failed repeatedly, the next
            # request is built without tools: the model has to
            # answer in prose instead of trying the same thing again.
            # An instruction alone is not enough, because a model
            # that is stuck often ignores it.
            if repeat_detected and self._call_signatures:
                completion_gate_instruction = (
                    self._repeat_instruction()
                )
                tools = ()
            elif self._cycling:
                # Ahead of the repeated-success warning on purpose. A
                # cycle is almost always *also* a set of repeats, and
                # the softer notice would win the branch every time,
                # so a genuine loop was never told what it was doing.
                # Answer-only, like the failure loop: a cycle will not
                # break itself, and each further turn makes the context
                # worse.
                completion_gate_instruction = (
                    self._cycle_instruction()
                )
                tools = ()
            elif repeated_success:
                # Told, not stopped. A repeated successful call is
                # warned about but still answered, because re-reading
                # after acting is legitimate; and it deliberately does
                # not feed the abort machinery above, which belongs to
                # the failure loop and changes when a run ends.
                completion_gate_instruction = (
                    self._repeat_success_instruction()
                )
            else:
                completion_gate_instruction = None

            if self._cycling:
                # Consumed. Left set, it would withdraw tools again on
                # every later turn of this run, including after the
                # model had moved on to different work.
                self._cycling = False

            # Failure loop only. A cycle already had its tools
            # withdrawn above, and aborting here instead would end the
            # run before the model ever saw the explanation -- it would
            # be told "stop calling tools" only by the refusal, never by
            # the request that carried no tools to begin with.
            if (
                repeat_detected
                and self._call_signatures
                and response.tool_calls
            ):
                # The model asked to retry anyway. Executing it
                # again is what produced the loop, so refuse and make
                # it speak instead: one more turn with no tools, and
                # if it still insists, the run ends with what we have.
                await self._emit(
                    ToolFinished(
                        iteration=iteration,
                        # The id from *this* response. Referencing
                        # the loop variable leaked a stale id from an
                        # earlier turn, so the synthetic card could
                        # not be matched to the call it describes.
                        tool_call_id=response.tool_calls[0].id,
                        tool_name=response.tool_calls[0].name,
                        output=None,
                        error_code="repeat_blocked",
                        error_message=(
                            "This exact call already failed "
                            "repeatedly and was not retried."
                        ),
                        run_id=self._run_id,
                        parent_run_id=self._parent_run_id,
                    ),
                    on_event,
                )

                repeat_detected = False
                repeated_success = False
                self._reset_repeats()

                final = await self._final_answer(
                    iteration,
                    messages,
                    on_event,
                )

                await self._emit(
                    AgentFinished(
                        result=final,
                        run_id=self._run_id,
                        parent_run_id=self._parent_run_id,
                        agent_id=self._agent_id,
                        role=self._role,
                        model=self._model,
                    ),
                    on_event,
                )

                return final

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

                    if result.error is not None:
                        if self._note_failed_call(tool_call):
                            repeat_detected = True

                    else:
                        self._clear_call_signatures()
                        repeat_detected = False

                        if self._note_repeat_call(tool_call):
                            repeated_success = True

                        self._note_call_sequence(tool_call)
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

                    if result.error is not None:
                        if self._note_failed_call(tool_call):
                            repeat_detected = True
                    else:
                        # A success means the agent is making
                        # progress; the budget starts over.
                        self._clear_call_signatures()
                        repeat_detected = False

                        if self._note_repeat_call(tool_call):
                            repeated_success = True

                        self._note_call_sequence(tool_call)

                        if tool_call.name == "todowrite":
                            # The checklist is part of the plan
                            # surface, so the panel has to see it now
                            # rather than at the next phase change --
                            # which for a long research run could be
                            # many turns away.
                            await self._emit_plan(on_event)

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

    async def _final_answer(
        self,
        iteration: int,
        messages: list[dict[str, Any]],
        on_event: EventCallback | None,
    ) -> str:
        """Ask for a tool-free answer, then fall back to a report.

        The loop has to end with something the user can read. A
        model that will not answer is described rather than retried,
        because retrying is what caused the loop.
        """

        instruction = self._repeat_instruction()

        summary = instruction + (
            " Respond now with the final answer only. Do not request "
            "any tool."
        )

        try:
            response = await self._chat_llm(
                iteration=iteration,
                messages=[
                    *messages,
                    {"role": "user", "content": summary},
                ],
                tools=(),
                on_event=on_event,
            )
        except Exception as exc:
            return (
                "The tool call failed repeatedly and the model could "
                f"not produce a final answer: {exc}"
            )

        text = (response.content or "").strip()

        if text:
            return text

        return (
            "The tool call failed repeatedly and the model produced no "
            "answer. The failing call was not retried again."
        )

    def _note_repeat_call(
        self,
        tool_call: Any,
    ) -> bool:
        """Track a *successful* identical call; True once it repeats.

        The failure guard exists because an agent stuck on an error
        retries forever. The mirror case was unguarded: a call that
        succeeds can be repeated exactly as often, and each repeat is
        a real request, real tokens and a real slot in the context,
        carrying back an answer the model has already read.

        Not an error -- re-reading something after acting on it is
        legitimate -- so this only warns once the same request has come
        back unchanged several times, and lets the model explain why it
        wants it again.
        """

        signature = f"{tool_call.name}:{_stable(tool_call.arguments)}"

        self._repeated_calls[signature] = (
            self._repeated_calls.get(signature, 0) + 1
        )

        return (
            self._repeated_calls[signature]
            >= self._max_identical_successes
        )

    def _note_call_sequence(
        self,
        tool_call: Any,
    ) -> bool:
        """Track the *order* of calls, not just each one alone.

        The existing guard counts a call against itself, so A, B, A, B
        looks like four first-time calls: each individual signature has
        been seen twice, but never three times, and a genuinely stuck
        agent ping-ponging between two tools walks straight through.
        Each pass costs a request and pushes the previous one out of
        the context window, so the loop gets tighter rather than
        shorter.

        An immediate return to the previous call is the signal. One
        bounce is ordinary -- checking a result, then adjusting -- so
        it takes two consecutive bounces to call it a cycle.
        """

        signature = f"{tool_call.name}:{_stable(tool_call.arguments)}"

        self._call_sequence.append(signature)

        # Only the tail matters; a full history would eventually match
        # two calls that merely resemble each other from earlier work.
        if len(self._call_sequence) > _CYCLE_WINDOW:
            del self._call_sequence[:-_CYCLE_WINDOW]

        if len(self._call_sequence) < 3:
            return False

        if (
            self._call_sequence[-1]
            == self._call_sequence[-3]
            and self._call_sequence[-1] != self._call_sequence[-2]
        ):
            self._cycles += 1
        else:
            self._cycles = 0

        self._cycling = self._cycles >= self._max_call_cycles

        return self._cycling

    def _cycle_instruction(self) -> str:
        """Name the cycle the model is in, since it cannot see it."""

        recent = list(self._call_sequence[-_CYCLE_WINDOW:])

        return (
            "TOOL CALL CYCLE DETECTED: recent tool calls are "
            f"{' -> '.join(recent)}. That pattern repeats without "
            "progress, and each pass costs a request and pushes the "
            "previous results out of your context.\n\n"
            "Stop cycling. Either commit to the best answer you have "
            "from what you already fetched, or state plainly what you "
            "are missing and what would resolve it."
        )

    def _repeat_success_instruction(self) -> str:
        """Explain that the same answer is already in hand."""

        return (
            "REPEATED CALL: you have already made this exact call and "
            "received its result, which is still in this conversation. "
            "Running it again returns the same thing at the cost of "
            "another request and more context.\n\n"
            "If you are calling it again because you meant to change "
            "something -- different arguments, a narrower query, a "
            "different tool -- do that instead. If you genuinely need "
            "it repeated, say why in your next message."
        )

    def _note_failed_call(self, tool_call: Any) -> bool:
        """Track a failed call; True once it is clearly a repeat.

        The agent had no signal that it was retrying the same
        request, so a tool that kept returning the same error was
        called again every iteration until the iteration cap aborted
        the run. Counting identical (tool, arguments) pairs that
        failed the same way turns that into an instruction the model
        can act on.
        """

        signature = f"{tool_call.name}:{_stable(tool_call.arguments)}"

        self._call_signatures[signature] = (
            self._call_signatures.get(signature, 0) + 1
        )

        return (
            self._call_signatures[signature]
            >= self._max_identical_failures
        )

    def _repeat_instruction(self) -> str:
        """Tell the model to stop retrying and answer instead."""

        return (
            "TOOL FAILURE LOOP DETECTED: the same tool call with the "
            "same arguments has now failed repeatedly with the same "
            "error. Retrying it again cannot succeed. Do not issue "
            "that call once more. Either use a different tool, change "
            "the arguments materially, or stop and report to the user "
            "what failed, which error you received, and what you "
            "could not accomplish."
        )

    def _clear_call_signatures(self) -> None:
        """Reset the *failure* counter only.

        Called after every success, because a success means the agent
        is making progress. It must not touch the repeat counter: that
        one is precisely about calls that succeeded.
        """

        self._call_signatures.clear()

    def _budget_exceeded(self) -> bool:
        """
        Whether this run has spent its token allowance.

        Prompt tokens are the thing that grows, and they are charged
        again on every call, so they are the only figure worth
        ceiling. Returns False when no ceiling is set, and when
        nothing has been measured -- an unmeasured run cannot be shown
        to have exceeded anything, and pretending otherwise would stop
        every provider that reports no usage.
        """

        if self._max_prompt_tokens is None:
            return False

        if not self._usage.measured:
            return False

        return self._usage.prompt_tokens >= self._max_prompt_tokens

    def _budget_stop_message(self) -> str:
        return (
            f"Stopped: this run reached its prompt-token allowance "
            f"of {self._max_prompt_tokens:,} "
            f"(used {self._usage.prompt_tokens:,}). "
            "Raise agent.max_prompt_tokens, or ask for a narrower "
            "task."
        )

    async def _stop_for_budget(
        self,
        on_event: EventCallback | None,
    ) -> str:
        """End the run on the token ceiling, accounting attached."""

        message = self._budget_stop_message()

        await self._emit(
            AgentFinished(
                result=message,
                run_id=self._run_id,
                parent_run_id=self._parent_run_id,
                agent_id=self._agent_id,
                role=self._role,
                model=self._model,
                prompt_tokens=self._usage.prompt_tokens,
                completion_tokens=(
                    self._usage.completion_tokens
                ),
                llm_calls=self._usage.calls,
                llm_calls_without_usage=(
                    self._usage.calls_without_usage
                ),
                estimated_cost=self._usage.estimate_cost(
                    self._price_per_million
                ),
            ),
            on_event,
        )

        return message

    def _reset_repeats(self) -> None:
        """Reset the per-signature counters at a turn boundary.

        Repeats are counted within a turn. The same call made again in
        the next turn is the agent coming back to something it decided
        to revisit, which is a decision rather than a loop.

        The call *sequence* is deliberately not reset here. A cycle is
        by definition several turns long, so clearing its window at the
        boundary would make it impossible to detect.
        """

        self._reset_cycles()

        self._call_signatures.clear()
        self._repeated_calls.clear()

    def _reset_cycles(self) -> None:
        """Forget the call pattern. Turn boundary only."""

        self._call_sequence.clear()
        self._cycles = 0
        self._cycling = False

    # ==================================================================
    # LLM
    # ==================================================================

    def _estimate_request(
        self,
        messages: list[dict[str, Any]],
    ) -> int:
        """What we think the request costs, for comparison with usage.

        Estimated from exactly the messages being sent, so the ratio
        against the provider's figure is meaningful rather than two
        unrelated numbers.
        """

        return sum(
            self._controller.counter.count(
                str(message.get("content", ""))
            )
            + 16
            for message in messages
        )

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
            # The streaming path is the one most runs take, so the
            # provider's count has to be read here too: measured only
            # on the non-streaming path would have made the meter look
            # estimated on every ordinary run.
            usage=extract_usage(
                raw_chunks[-1] if raw_chunks else {}
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
                # Next to the estimate for the same request: their
                # ratio is the only honest measure of whether the
                # budget arithmetic can be relied on.
                prompt_tokens=(
                    response.usage.prompt_tokens
                    if response.usage is not None
                    else None
                ),
                completion_tokens=(
                    response.usage.completion_tokens
                    if response.usage is not None
                    else None
                ),
                estimated_prompt_tokens=self._estimate_request(
                    messages
                ),
            ),
            on_event,
        )

        if response.usage is None:
            self._usage.note_unreported()
        else:
            self._usage.add(response.usage)

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
                # Next to the estimate for the same request: their
                # ratio is the only honest measure of whether the
                # budget arithmetic can be relied on.
                prompt_tokens=(
                    response.usage.prompt_tokens
                    if response.usage is not None
                    else None
                ),
                completion_tokens=(
                    response.usage.completion_tokens
                    if response.usage is not None
                    else None
                ),
                estimated_prompt_tokens=self._estimate_request(
                    messages
                ),
            ),
            on_event,
        )

        if response.usage is None:
            self._usage.note_unreported()
        else:
            self._usage.add(response.usage)

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

def _stable(value: Any) -> str:
    """A deterministic, order-independent rendering of arguments.

    Two calls that differ only in key order must count as the same
    retry, or a model that reorders its JSON would escape the guard.
    """

    try:
        return json.dumps(
            value,
            sort_keys=True,
            default=str,
        )
    except (TypeError, ValueError):
        return str(value)


def _trailing_text(request) -> str:
    """The instruction the user just gave, for the summary prompt.

    Taken from the assembled request rather than the dialogue because
    the window usually ends with a tool result at rollover time, and
    the last message is not the thing being asked for.
    """

    for message in reversed(request.to_message_list()):
        if message.get("role") != "user":
            continue

        content = message.get("content")

        if isinstance(content, str) and content.strip():
            return content.strip()

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("text"):
                    return str(block["text"]).strip()

    return ""


CONSOLIDATION_NOTE = """\
[BEFORE YOU ANSWER]

You are about to finish. If this session worked out something that \
took more than one attempt to learn -- a rule of a system, a \
convention of a codebase, a fact you had to try twice to get -- save \
it now with the remember tool, before you write your answer.

Nothing to save is a fine answer. Do not save the values of \
variables, tool output you could fetch again, or anything already \
stated in this conversation. If you did not learn anything, skip \
this and answer."""
