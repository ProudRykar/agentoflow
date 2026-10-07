from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_workflow.core.entities.models.agent_phase import AgentPhase


@dataclass(slots=True, frozen=True)
class AgentStarted:
    prompt: str
    run_id: str = ""
    parent_run_id: str | None = None
    agent_id: str = "main"
    role: str = "main"
    model: str = ""
    # A retry of the question already in the transcript, not a new one.
    # The prompt repeats an existing turn on purpose -- that is what is
    # being answered again -- so a consumer that treated it as a fresh
    # user message would show the same question twice.
    regenerated: bool = False


@dataclass(slots=True, frozen=True)
class AgentPhaseChanged:
    previous_phase: AgentPhase
    phase: AgentPhase
    reason: str
    iteration: int
    run_id: str = ""
    parent_run_id: str | None = None


@dataclass(slots=True, frozen=True)
class LLMRequested:
    iteration: int
    message_count: int
    tool_count: int
    run_id: str = ""
    parent_run_id: str | None = None
    estimated_tokens: int | None = None
    # The budget the estimate is measured against, so a client can
    # show real headroom instead of a bare number.
    context_limit: int | None = None
    # True once the assembler had to drop or trim blocks to fit.
    context_trimmed: bool = False
    # Whether estimated_tokens came from a real tokenizer or from a
    # length heuristic. A client showing a guess with the same
    # confidence as a measurement is how a silent underestimate turns
    # into a confusing rejection from the provider.
    counter_exact: bool = True
    # Names of the blocks that were dropped or cut, so "trimmed" can
    # say which rather than only that.
    dropped_blocks: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class LLMThinkingChunk:
    iteration: int
    content: str
    run_id: str = ""
    parent_run_id: str | None = None


@dataclass(slots=True, frozen=True)
class LLMContentChunk:
    iteration: int
    content: str
    run_id: str = ""
    parent_run_id: str | None = None


@dataclass(slots=True, frozen=True)
class LLMResponded:
    iteration: int
    content: str | None
    thinking: str | None
    tool_call_count: int
    run_id: str = ""
    parent_run_id: str | None = None
    # What the provider reported consuming, when it reported anything.
    # Present so the local estimate can be checked instead of trusted.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # The local estimate for the same request, next to the real
    # figure. Their ratio is the only honest measure of how much the
    # budget arithmetic can be relied on.
    estimated_prompt_tokens: int | None = None


@dataclass(slots=True, frozen=True)
class ToolStarted:
    iteration: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    run_id: str = ""
    parent_run_id: str | None = None


@dataclass(slots=True, frozen=True)
class ToolFinished:
    iteration: int
    tool_call_id: str
    tool_name: str
    output: str | None
    error_code: str | None
    error_message: str | None
    run_id: str = ""
    parent_run_id: str | None = None
    duration_seconds: float | None = None


@dataclass(slots=True, frozen=True)
class PlanUpdated:
    """The execution plan changed shape or state.

    The plan was already being built and advanced in the runtime, but
    nothing carried it to a browser: a client could only be told the
    phase moved, which is the same information as AgentPhaseChanged with
    the one part that mattered -- which of the steps -- thrown away.

    The whole plan travels on every change rather than a delta. A plan
    is a handful of steps, it changes on the order of once per phase,
    and a client that missed one event can rebuild the full picture from
    the next. Deltas would need acknowledgement and ordering the
    reconnect path does not currently guarantee.
    """

    objective: str
    steps: list[dict[str, Any]]
    revision: int
    current_step_id: str | None = None
    run_id: str = ""
    parent_run_id: str | None = None
    # The model's own checklist, in the same shape as ``steps`` so a
    # client renders one list component rather than two. None until the
    # model has written one: an empty list and "the model has not
    # planned yet" are different situations.
    todos: list[dict[str, Any]] | None = None


@dataclass(slots=True, frozen=True)
class AgentFinished:
    result: str
    run_id: str = ""
    parent_run_id: str | None = None
    agent_id: str = "main"
    role: str = "main"
    model: str = ""
    # Measured totals for the run, so its cost is knowable without
    # summing the stream afterwards. The prompt count is what grows
    # with a long dialogue, which is why the number matters locally as
    # well as on a paid endpoint.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    # Responses whose provider reported nothing. Zero totals with this
    # equal to llm_calls means "unmeasured", not "free" -- and the two
    # must not be reported the same way.
    llm_calls_without_usage: int = 0
    # None when no price is configured, rather than zero: a local model
    # is genuinely free and a mispriced remote one is not.
    estimated_cost: float | None = None


AgentEvent = (
    AgentStarted
    | AgentPhaseChanged
    | PlanUpdated
    | LLMRequested
    | LLMThinkingChunk
    | LLMContentChunk
    | LLMResponded
    | ToolStarted
    | ToolFinished
    | AgentFinished
)

@dataclass(slots=True, frozen=True)
class RunFailed:
    """Session-level failure, not an Agent event.

    Agent errors surface as exceptions from ``run``; the web layer
    needs them as a stream item so the browser can render them like any
    other message.

    It lives here rather than in ``session.py`` because the durable
    event log rebuilds every payload it stores through its registry,
    and a session-only class could never be replayed: a session whose
    last run failed came back with no record of it.
    """

    session_id: str
    error: str
