"""Runs the behavior half of the eval corpus.

Each ``expect`` string names a check rather than describing prose, and
the checks are the ones the session's real bugs produced. That is the
point: an assertion written while fixing a defect survives the next
refactor of the surrounding code, which a test written to match the
shape of the code at the time does not.

Deliberately not an LLM-in-the-loop test. A model that phrases things
differently should not turn the suite red, and a case that needs one
to check anything belongs in the live half with a human reading it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_workflow.core.context.context_assembler import (
    ContextAssembler,
)
from agent_workflow.core.entities.models.agent import Agent
from agent_workflow.core.entities.models.agent_trace import AgentFinished
from agent_workflow.core.entities.models.llm import (
    LLMToolCall,
    LLMResponse,
    TokenUsage,
)
from agent_workflow.core.entities.models.llm_client import LLMClient
from agent_workflow.core.entities.models.research_contract import (
    ResearchPage,
    ResearchResult,
)
from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolPolicy,
)
from agent_workflow.core.entities.models.tool_executor import ToolExecutor
from agent_workflow.core.entities.models.tool_registry import ToolRegistry
from agent_workflow.evals.cases import EvalCase

Check = Callable[["RunOutcome"], list[str]]


@dataclass(slots=True)
class RunOutcome:
    """Everything an assertion may look at."""

    result: str
    events: list[Any]
    final_prompt: list[dict[str, Any]]
    agent: Agent
    context: ToolContext



# ======================================================================
# Scripted models
# ======================================================================


class ScriptedLLM(LLMClient):
    """Replays a fixed list of responses, then answers plainly."""

    def __init__(
        self,
        responses: list[LLMResponse],
        usage: TokenUsage | None = None,
    ) -> None:
        self._usage = usage
        # Scripted responses carry usage too, not just the fallback.
        # A budget case scripted without it measures nothing, and a
        # ceiling cannot fire on a number that was never reported.
        self._responses = [
            LLMResponse(
                content=response.content,
                tool_calls=response.tool_calls,
                raw=response.raw,
                usage=usage,
            )
            for response in responses
        ]
        self.prompts: list[list[dict[str, Any]]] = []
        self.fallbacks = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[Any, ...] = (),
    ) -> LLMResponse:
        self.prompts.append([dict(message) for message in messages])

        # A request with no tools must be answered in prose. Following
        # the script regardless would make every loop-detection case
        # untestable: a model that keeps calling tools after they have
        # been withdrawn is not the thing being measured.
        if not tools:
            self.fallbacks += 1

            return LLMResponse(
                content="I am done.",
                tool_calls=(),
                raw={},
                usage=self._usage,
            )

        if self._responses:
            return self._responses.pop(0)

        self.fallbacks += 1

        return LLMResponse(
            content="I am done.",
            tool_calls=(),
            raw={},
            usage=self._usage,
        )


def _text(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=(),
        raw={},
    )


_CALL_SERIAL = 0


def _call(tool: str, **arguments: Any) -> LLMResponse:
    # Not named `name`: tool arguments legitimately have a field of
    # that name, and a keyword collision here is a confusing way to
    # discover it.
    global _CALL_SERIAL

    _CALL_SERIAL += 1

    return LLMResponse(
        content="",
        tool_calls=(
            LLMToolCall(
                # Unique per call. A shared id makes every result look
                # like a reply to the same call, which is exactly the
                # confusion this suite exists to catch.
                id=f"c{_CALL_SERIAL}",
                name=tool,
                arguments=arguments,
            ),
        ),
        raw={},
    )


# ======================================================================
# A tool to call
# ======================================================================


async def _noop(
    arguments: dict[str, Any],
    context: ToolContext,
) -> str:
    return "ok"


@dataclass(slots=True, frozen=True)
class _ToolArguments:
    """Accepts anything the scripted cases throw at it."""

    path: str = "x"
    name: str = "x"


def _registry_with_noop() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="noop",
            description="does nothing",
            input_type=_ToolArguments,
            handler=_noop,
            policy=ToolPolicy(
                permissions=frozenset(),
                timeout=1.0,
                max_output_size=200,
            ),
        ),
    )

    registry.register(
        Tool(
            name="alpha",
            description="tool alpha",
            input_type=_ToolArguments,
            handler=_noop,
            policy=ToolPolicy(
                permissions=frozenset(),
                timeout=1.0,
                max_output_size=200,
            ),
        ),
    )

    registry.register(
        Tool(
            name="beta",
            description="tool beta",
            input_type=_ToolArguments,
            handler=_noop,
            policy=ToolPolicy(
                permissions=frozenset(),
                timeout=1.0,
                max_output_size=200,
            ),
        ),
    )

    return registry


def _page(url: str) -> ResearchPage:
    return ResearchPage(
        url=url,
        depth=0,
        title=f"title {url}",
        content="body",
        links=(),
        content_bytes=4,
    )


def _research(url: str) -> ResearchResult:
    return ResearchResult(
        root_url=url,
        pages=(_page(url),),
        discovered_urls=(),
        failed_urls=(),
        max_depth_reached=0,
        total_bytes=4,
        page_limit_reached=False,
        byte_limit_reached=False,
    )


# ======================================================================
# Checks
# ======================================================================


def _answer_visible_to_next_turn(
    outcome: RunOutcome,
) -> list[str]:
    """The finished answer must reach the model's next request."""

    problems: list[str] = []

    prompts = getattr(outcome.agent, "_eval_prompts", [])

    if len(prompts) < 2:
        return ["the run made too few requests to check"]

    roles = [
        message.get("role")
        for message in prompts[1]
    ]

    if "assistant" not in roles:
        problems.append(
            "the second request contains no assistant turn, so the "
            "model cannot refer to its own answer"
        )

    return problems


def _regenerate_replaces_answer(outcome: RunOutcome) -> list[str]:
    problems: list[str] = []

    prompts = getattr(outcome.agent, "_eval_prompts", [])

    if len(prompts) < 2:
        return ["the retry made too few requests to check"]

    contents = [
        str(message.get("content"))
        for message in prompts[1]
        if message.get("role") == "assistant"
    ]

    if any("first answer" in content for content in contents):
        problems.append(
            "the discarded answer is still in the retry's context"
        )

    return problems


def _regenerate_drops_evidence(outcome: RunOutcome) -> list[str]:
    titles = {
        item.title
        for item in outcome.agent.orchestrator.evidence_store.all()
    }

    if "title https://example.com/discarded" in titles:
        return [
            "evidence gathered by the discarded attempt survived the "
            "retry"
        ]

    return []


def _todo_is_seedable(outcome: RunOutcome) -> list[str]:
    todos = outcome.agent.orchestrator.todo_list

    if todos.is_empty():
        return ["no checklist was seeded"]

    contents = [item.content for item in todos.items]

    if any("RESEARCH" in content.upper() for content in contents):
        return [
            "a greeting produced a research step: "
            f"{contents}"
        ]

    return []


def _budget_stops_run(outcome: RunOutcome) -> list[str]:
    if "allowance" not in outcome.result:
        return [
            f"the run was not stopped by its token allowance: "
            f"{outcome.result[:80]!r}"
        ]

    return []


def _usage_reported(outcome: RunOutcome) -> list[str]:
    finished = [
        event
        for event in outcome.events
        if isinstance(event, AgentFinished)
    ]

    if not finished:
        return ["no AgentFinished was emitted"]

    event = finished[-1]

    if event.prompt_tokens <= 0:
        return ["no prompt tokens were reported"]

    if event.estimated_cost is None:
        return ["no cost was reported despite a configured price"]

    return []


def _cycle_stops_run(outcome: RunOutcome) -> list[str]:
    prompts = getattr(outcome.agent, "_eval_prompts", [])

    notice = any(
        "TOOL CALL CYCLE DETECTED" in str(message.get("content"))
        for prompt in prompts
        for message in prompt
    )

    if not notice:
        return ["the model was never told it was cycling"]

    return []


def _repeat_is_allowed_once(outcome: RunOutcome) -> list[str]:
    prompts = getattr(outcome.agent, "_eval_prompts", [])

    executed = sum(
        1
        for prompt in prompts
        for message in prompt
        if message.get("role") == "tool"
    )

    # Warned, not blocked: the call has to still happen.
    if executed < 2:
        return [
            f"the repeat was blocked rather than warned about "
            f"({executed} tool result(s) seen)"
        ]

    return []


def _dropped_blocks_declared(outcome: RunOutcome) -> list[str]:
    last = outcome.final_prompt

    if not last:
        return ["no request was made"]

    declared = any(
        "NOT IN THIS REQUEST" in str(message.get("content"))
        for message in last
    )

    if not declared:
        return [
            "memory was dropped without telling the model what was "
            "lost"
        ]

    return []


def _unmeasured_is_labelled(outcome: RunOutcome) -> list[str]:
    finished = [
        event
        for event in outcome.events
        if isinstance(event, AgentFinished)
    ]

    if not finished:
        return ["no AgentFinished was emitted"]

    event = finished[-1]

    if event.llm_calls == 0:
        return ["calls made but none recorded"]

    if event.llm_calls_without_usage != event.llm_calls:
        return [
            "a silent provider was not recorded as unmeasured"
        ]

    if event.estimated_cost is not None:
        return [
            "an unmeasured run reported a cost, which is a guess "
            "wearing a bill's clothes"
        ]

    return []


def _todo_reaches_model(outcome: RunOutcome) -> list[str]:
    if not outcome.final_prompt:
        return ["no request was made"]

    rendered = any(
        "▼Todo" in str(message.get("content"))
        for message in outcome.final_prompt
    )

    if not rendered:
        return ["the checklist never reached the prompt"]

    return []


def _sync_callback_works(outcome: RunOutcome) -> list[str]:
    if not outcome.events:
        return ["the synchronous callback received nothing"]

    return []


CHECKS: dict[str, Check] = {
    "answer_visible_to_next_turn": _answer_visible_to_next_turn,
    "regenerate_replaces_answer": _regenerate_replaces_answer,
    "regenerate_drops_evidence": _regenerate_drops_evidence,
    "todo_is_seedable": _todo_is_seedable,
    "budget_stops_run": _budget_stops_run,
    "usage_reported": _usage_reported,
    "cycle_stops_run": _cycle_stops_run,
    "repeat_is_allowed_once": _repeat_is_allowed_once,
    "dropped_blocks_declared": _dropped_blocks_declared,
    "unmeasured_is_labelled": _unmeasured_is_labelled,
    "todo_reaches_model": _todo_reaches_model,
    "sync_callback_works": _sync_callback_works,
}


# ======================================================================
# Scenarios
# ======================================================================


async def _run(
    llm: LLMClient,
    tmp_path: Path,
    *,
    prepare: Callable[[Agent], None] | None = None,
    after_first_turn: Callable[[Agent], None] | None = None,
    second_turn: str | None = None,
    regenerate: bool = False,
    **agent_kwargs: Any,
) -> RunOutcome:
    registry = _registry_with_noop()

    agent = Agent(
        llm=llm,
        registry=registry,
        executor=ToolExecutor(registry),
        **agent_kwargs,
    )

    context = ToolContext(
        working_directory=tmp_path,
        environment={},
        allowed_path=(tmp_path,),
        permissions=frozenset(),
    )

    context.agent = agent

    events: list[Any] = []

    async def callback(event: Any) -> None:
        events.append(event)

    if prepare is not None:
        prepare(agent)

    result = await agent.run(
        prompt=tmp_path.name,
        context=context,
        on_event=callback,
    )

    if after_first_turn is not None:
        after_first_turn(agent)

    if second_turn is not None:
        result = await agent.continue_run(
            prompt=second_turn,
            context=context,
            on_event=callback,
        )

    if regenerate:
        result = await agent.regenerate(
            context=context,
            on_event=callback,
        )

    prompts = getattr(llm, "prompts", [])

    agent._eval_prompts = prompts  # type: ignore[attr-defined]
    return RunOutcome(
        result=result,
        events=events,
        final_prompt=prompts[-1] if prompts else [],
        agent=agent,
        context=context,
    )


async def evaluate(
    case: EvalCase,
    tmp_path: Path,
) -> list[str]:
    """Run one case, returning what went wrong. Empty means it held."""

    llm = ScriptedLLM(_responses_for(case))

    options: dict[str, Any] = {}

    match case.id:
        case "follow-up-needs-its-own-answer":
            llm = ScriptedLLM([_text("first answer"), _text("because")])
            options["second_turn"] = "why?"

        case "retry-replaces-rather-than-appends":
            llm = ScriptedLLM(
                [_text("first answer"), _text("second answer")]
            )
            options["regenerate"] = True

        case "retry-does-not-inherit-research":
            llm = ScriptedLLM(
                [_text("first answer"), _text("second answer")]
            )
            options["regenerate"] = True
            # Gathered by the attempt being replaced, which is the
            # only placement that tests anything: evidence seeded
            # before the run is part of the baseline and is meant to
            # survive.
            options["after_first_turn"] = (
                lambda agent: agent.orchestrator
                .store_research_evidence(
                    _research("https://example.com/discarded"),
                )
            )

        case "budget-stops-a-run":
            llm = ScriptedLLM(
                [_call("noop", path="a") for _ in range(20)],
                usage=TokenUsage(
                    prompt_tokens=500,
                    completion_tokens=20,
                ),
            )
            options["max_iterations"] = 8
            options["max_prompt_tokens"] = 1000
            options["price_per_million"] = (3.0, 15.0)

        case "cycle-is-caught":
            alternating = [
                _call(
                    "alpha" if index % 2 == 0 else "beta",
                    name=str(index % 2),
                )
                for index in range(20)
            ]
            llm = ScriptedLLM(alternating)
            options["max_iterations"] = 10

        case "repeating-a-successful-call-is-warned-not-blocked":
            llm = ScriptedLLM(
                [
                    _call("noop", path="same"),
                    _call("noop", path="same"),
                    _text("done"),
                ],
            )
            options["max_iterations"] = 6

        case "silent-provider-is-not-reported-as-free":
            llm = ScriptedLLM([_text("done")], usage=None)
            options["price_per_million"] = (3.0, 15.0)

        case "long-context-says-what-was-dropped":
            # Evidence is the easiest evictable block to overflow, and
            # the notice is emitted for any of the three.
            options["prepare"] = _fill_evidence
            options["assembler"] = _tiny_budget()

        case "todo-becomes-context":
            llm = ScriptedLLM([_text("planned")])
            options["second_turn"] = "go on"

    outcome = await _run(
        llm,
        tmp_path,
        **options,
    )

    problems: list[str] = []

    for expectation in case.expect:
        check = CHECKS.get(expectation)

        if check is None:
            problems.append(
                f"no check named {expectation!r}; the corpus and the "
                "runner have drifted apart"
            )
            continue

        problems.extend(check(outcome))

    return problems


def _fill_evidence(agent: Agent) -> None:
    """Overflow the evictable blocks so something has to be dropped."""

    for index in range(40):
        agent.orchestrator.store_research_evidence(
            ResearchResult(
                root_url=f"https://example.com/{index}",
                pages=(
                    ResearchPage(
                        url=f"https://example.com/{index}",
                        depth=0,
                        title=f"page {index}",
                        content="x" * 400,
                        links=(),
                        content_bytes=400,
                    ),
                ),
                discovered_urls=(),
                failed_urls=(),
                max_depth_reached=0,
                total_bytes=400,
                page_limit_reached=False,
                byte_limit_reached=False,
            ),
        )


def _tiny_budget() -> ContextAssembler:
    from agent_workflow.core.context.context_assembler import (
        ContextAssembler,
    )
    from agent_workflow.core.context.context_budget import ContextBudget

    return ContextAssembler(
        budget=ContextBudget(
            maximum_tokens=2_000,
            reserved_system=200,
            reserved_task=200,
            reserved_output=400,
        ),
    )


def _responses_for(case: EvalCase) -> list[LLMResponse]:
    return [_text("done")]
