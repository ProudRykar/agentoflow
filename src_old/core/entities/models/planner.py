from __future__ import annotations

import re
from urllib.parse import urlsplit

from core.entities.models.agent_phase import AgentPhase
from core.entities.models.agent_plan import AgentPlan
from core.entities.models.plan_step import (
    DelegationHint,
    PlanStep,
    PlanStepStatus,
)
from core.entities.models.research_contract import (
    ResearchContract,
    canonicalize_url,
)
from core.entities.models.subagent import SubagentPower
from core.entities.models.task_contract import TaskContract
from core.entities.models.task_plan import TaskPlan


class Planner:
    """
    Deterministic task and execution planner.

    Responsibilities:

    1. Parse user input into TaskPlan.
    2. Build an executable AgentPlan from TaskPlan + TaskContract.
    3. Revise AgentPlan when CompletionGate reports missing requirements.
    4. Restore a valid execution path after replanning.

    The planner:

    - does not execute tools;
    - does not execute the LLM;
    - does not decide semantic completion;
    - does not own TaskProgress;
    - does not replace CompletionGate.

    CompletionGate decides whether the task is semantically complete.
    Planner decides what should happen next when the current execution
    plan is insufficient.

    Research planning is deterministic.

    Explicit research requirements written by the user always take
    precedence over planner defaults.
    """

    # ------------------------------------------------------------------
    # Research policy defaults
    # ------------------------------------------------------------------

    # A URL alone means that at least the root page must be inspected.
    _DEFAULT_RESEARCH_MIN_PAGES = 1
    _DEFAULT_RESEARCH_MIN_DEPTH = 0

    # Verbs / nouns that indicate that the user expects an actual
    # investigation rather than a single-page fetch.
    _DEEP_RESEARCH_MIN_PAGES = 5
    _DEEP_RESEARCH_MIN_DEPTH = 1

    _URL_PATTERN = re.compile(
        r"""https?://[^\s<>\[\]{}"']+""",
        re.IGNORECASE,
    )

    # Examples:
    #
    #   минимум 5 страниц
    #   минимум 5 страниц
    #   не менее 5 страниц
    #   не меньше 5 страниц
    #   min 5 pages
    #   at least 5 pages
    #   not less than 5 pages
    #
    # Also accepts the number before "pages":
    #
    #   5 страниц минимум
    #   5 pages minimum
    #
    _PAGE_COUNT_PATTERN = re.compile(
        r"""
        (?:
            (?:
                минимум
                |не\s+менее
                |не\s+меньше
                |min
                |minimum
                |at\s+least
                |not\s+less\s+than
            )
            \s*
            (?P<prefix_value>\d+)
            (?:\s*(?:страниц?|pages?))?
            |
            (?P<suffix_value>\d+)
            \s*
            (?:страниц?|pages?)
            \s*
            (?:
                минимум
                |minimum
                |min
            )
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # Examples:
    #
    #   глубина 2
    #   минимальная глубина 2
    #   глубина минимум 2
    #   depth 2
    #   minimum depth 2
    #   depth at least 2
    #
    _DEPTH_PATTERN = re.compile(
        r"""
        (?:
            (?:
                (?:минимальная|минимум|min)?\s*
                глубина
                |depth
                |
                depth\s+
                (?:minimum|min|at\s+least|not\s+less\s+than)
            )
            \s*
            (?:
                минимум
                |minimum
                |min
                |at\s+least
                |not\s+less\s+than
            )?
            \s*
            (?P<value>\d+)
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # User verbs used to derive a concise objective.
    _VERB_PATTERNS = (
        r"^(изучи|изучить|обучи|обучить)\s+",
        r"^(сравни|сравнить|сравнение)\s+",
        r"^(найди|найти|поищи|поиск)\s+",
        r"^(напиши|написать|создай|создать)\s+",
        r"^(проанализируй|проанализировать|анализ)\s+",
        r"^(почитай|прочти|прочитать)\s+",
        r"^(исследуй|исследовать|исследование)\s+",
        r"^(расскажи|рассказать)\s+",
        (
            r"^(review|analyze|analyse|compare|research|"
            r"investigate|read|inspect|study)\s+"
        ),
    )

    # Research intent is intentionally broader than _VERB_PATTERNS.
    #
    # Examples:
    #
    #   проанализируй библиотеку
    #   исследуй документацию
    #   изучи этот фреймворк
    #   сделай обзор API
    #   analyze this framework
    #   research the documentation
    #
    _DEEP_RESEARCH_INTENT_PATTERN = re.compile(
        r"""
        (?:
            \b
            (?:
                проанализ\w*
                |анализ\w*
                |исслед\w*
                |изуч\w*
                |обзор\w*
                |разбер\w*
                |review\w*
                |analy[sz]\w*
                |research\w*
                |investigat\w*
                |inspect\w*
                |study\w*
            )
            \b
            |
            \b
            (?:
                библиотек\w*
                |фреймворк\w*
                |документац\w*
                |сайт\w*
                |репозитор\w*
                |framework\w*
                |library\w*
                |documentation
                |docs
                |repository
            )
            \b
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    _LEADING_GREETINGS = frozenset(
        {
            "привет",
            "hello",
            "здравствуйте",
            "добрый",
            "hi",
        }
    )

    # ------------------------------------------------------------------
    # Public planning API
    # ------------------------------------------------------------------

    def plan(
        self,
        prompt: str,
    ) -> TaskPlan:
        """
        Convert raw user input into an immutable TaskPlan.
        """

        if not isinstance(
            prompt,
            str,
        ):
            raise TypeError(
                "prompt must be a string",
            )

        objective = self._extract_objective(
            prompt,
        )

        research = self._extract_research(
            prompt,
        )

        return TaskPlan(
            objective=objective,
            research=research,
        )

    def build_execution_plan(
        self,
        task: TaskPlan,
        contract: TaskContract,
    ) -> AgentPlan:
        """
        Convert task requirements into a mutable runtime AgentPlan.

        The resulting plan has the following structure:

            research? -> execution -> verification? ->
            reflection? -> synthesis

        Invariants after this method returns:

        - all steps are PENDING except the first step;
        - exactly one step is ACTIVE;
        - synthesis exists;
        - synthesis is the final step;
        - step IDs are unique.
        """

        steps: list[PlanStep] = []

        if contract.requires_research:
            steps.append(
                PlanStep(
                    id="research",
                    description=self._research_description(
                        contract,
                    ),
                    phase=AgentPhase.RESEARCH,
                    delegation_hint=self._hint_for_phase(
                        AgentPhase.RESEARCH,
                        task.objective,
                    ),
                ),
            )

        steps.append(
            PlanStep(
                id="execution",
                description=task.objective,
                phase=AgentPhase.EXECUTION,
                delegation_hint=self._hint_for_phase(
                    AgentPhase.EXECUTION,
                    task.objective,
                ),
            ),
        )

        if contract.requires_verification:
            steps.append(
                PlanStep(
                    id="verification",
                    description=self._phase_description(
                        AgentPhase.VERIFICATION,
                    ),
                    phase=AgentPhase.VERIFICATION,
                    delegation_hint=self._hint_for_phase(
                        AgentPhase.VERIFICATION,
                        task.objective,
                    ),
                ),
            )

        if contract.requires_reflection:
            steps.append(
                PlanStep(
                    id="reflection",
                    description=self._phase_description(
                        AgentPhase.REFLECTION,
                    ),
                    phase=AgentPhase.REFLECTION,
                    delegation_hint=self._hint_for_phase(
                        AgentPhase.REFLECTION,
                        task.objective,
                    ),
                ),
            )

        steps.append(
            PlanStep(
                id="synthesis",
                description=self._phase_description(
                    AgentPhase.SYNTHESIS,
                ),
                phase=AgentPhase.SYNTHESIS,
            ),
        )

        plan = AgentPlan(
            objective=task.objective,
            steps=steps,
            revision=1,
            current_step_id=None,
        )

        self._activate_first_step(
            plan,
        )

        self._validate_plan(
            plan,
        )

        return plan

    def revise_for_missing_phase(
        self,
        plan: AgentPlan,
        phase: AgentPhase,
        reason: str,
    ) -> PlanStep:
        """
        Revise an existing execution plan after CompletionGate rejected
        completion.

        Existing completed work is preserved.

        The missing phase is inserted before synthesis.

        If that phase already exists and is not semantically completed,
        the existing step is reused.

        Synthesis is reopened and placed after the newly required work.

        Resulting structure:

            completed
            completed
            missing phase (ACTIVE)
            synthesis (PENDING)

        The method guarantees that the returned step is executable.
        """

        if phase in {
            AgentPhase.IDLE,
            AgentPhase.PLANNING,
            AgentPhase.COMPLETED,
            AgentPhase.BLOCKED,
            AgentPhase.WAITING_APPROVAL,
        }:
            raise ValueError(
                f"Phase {phase.value!r} cannot be inserted into "
                "an execution plan.",
            )

        plan.revision += 1

        synthesis = self._find_synthesis(
            plan,
        )

        if synthesis is None:
            synthesis = self._create_synthesis(
                plan,
            )

        # --------------------------------------------------------------
        # Preserve completed work.
        #
        # If the requested phase already exists and is completed,
        # there is no semantic reason to execute it again.
        # We can simply reopen synthesis.
        # --------------------------------------------------------------

        existing = self._find_non_synthesis_step(
            plan,
            phase,
        )

        if (
            existing is not None
            and existing.status == PlanStepStatus.COMPLETED
        ):
            self._reopen_synthesis(
                plan,
                reason=reason,
            )

            self._normalize_plan(
                plan,
            )

            return synthesis

        # --------------------------------------------------------------
        # Reuse an existing unfinished phase.
        # --------------------------------------------------------------

        if existing is not None:
            self._prepare_step_for_retry(
                existing,
                reason,
            )

            self._move_before_synthesis(
                plan,
                existing,
            )

            self._reopen_synthesis(
                plan,
                reason=reason,
            )

            plan.activate(
                existing,
            )

            self._normalize_plan(
                plan,
                active_step=existing,
            )

            self._validate_plan(
                plan,
            )

            return existing

        # --------------------------------------------------------------
        # Create a new missing phase.
        # --------------------------------------------------------------

        step = PlanStep(
            id=self._revision_step_id(
                plan,
                phase,
            ),
            description=self._phase_description(
                phase,
            ),
            phase=phase,
            delegation_hint=self._hint_for_phase(
                phase,
                plan.objective,
            ),
        )

        self._insert_before_synthesis(
            plan,
            step,
        )

        self._reopen_synthesis(
            plan,
            reason=reason,
        )

        plan.activate(
            step,
        )

        self._normalize_plan(
            plan,
            active_step=step,
        )

        self._validate_plan(
            plan,
        )

        return step

    def return_to_synthesis(
        self,
        plan: AgentPlan,
        reason: str = "вернуться к синтезу результата",
    ) -> PlanStep | None:
        """
        Restore synthesis as the executable step.

        Used after a newly required runtime phase has been completed.

        A previously completed synthesis step is deliberately reopened:
        the final answer must be synthesized again after the plan changed.
        """

        synthesis = self._find_synthesis(
            plan,
        )

        if synthesis is None:
            synthesis = self._create_synthesis(
                plan,
            )

        self._reopen_synthesis(
            plan,
            reason=reason,
        )

        plan.activate(
            synthesis,
        )

        self._normalize_plan(
            plan,
            active_step=synthesis,
        )

        self._validate_plan(
            plan,
        )

        return synthesis

    # ------------------------------------------------------------------
    # Plan normalization
    # ------------------------------------------------------------------

    def _activate_first_step(
        self,
        plan: AgentPlan,
    ) -> PlanStep:
        """
        Activate the first executable step of a newly created plan.
        """

        step = plan.next_pending_step()

        if step is None:
            raise ValueError(
                "Cannot activate empty execution plan.",
            )

        plan.activate(
            step,
        )

        return step

    def _normalize_plan(
        self,
        plan: AgentPlan,
        active_step: PlanStep | None = None,
    ) -> None:
        """
        Restore basic AgentPlan invariants.

        The planner does not rewrite completed work.

        It only ensures:

        - there is one active step;
        - steps before active are completed/skipped;
        - steps after active are pending;
        - synthesis remains last.
        """

        if not plan.steps:
            raise ValueError(
                "Execution plan cannot be empty.",
            )

        synthesis = self._find_synthesis(
            plan,
        )

        if synthesis is None:
            synthesis = self._create_synthesis(
                plan,
            )

        # Synthesis must always be the final step.
        self._move_to_end(
            plan,
            synthesis,
        )

        if active_step is None:
            active_step = plan.current_step()

        if active_step is None:
            active_step = plan.next_pending_step()

        if active_step is None:
            # Every step is completed.
            # There is no executable work left.
            plan.current_step_id = None
            return

        # Make sure active step is actually active.
        if active_step.status != PlanStepStatus.ACTIVE:
            active_step.start()

        plan.current_step_id = active_step.id

        active_index = plan.steps.index(
            active_step,
        )

        for index, step in enumerate(
            plan.steps,
        ):
            if step is active_step:
                continue

            if index < active_index:
                if step.status in {
                    PlanStepStatus.PENDING,
                    PlanStepStatus.ACTIVE,
                    PlanStepStatus.FAILED,
                }:
                    step.status = PlanStepStatus.COMPLETED

            else:
                if step.status == PlanStepStatus.ACTIVE:
                    step.status = PlanStepStatus.PENDING

                if step.status == PlanStepStatus.FAILED:
                    step.status = PlanStepStatus.PENDING

    # ------------------------------------------------------------------
    # Step manipulation
    # ------------------------------------------------------------------

    def _prepare_step_for_retry(
        self,
        step: PlanStep,
        reason: str,
    ) -> None:
        """
        Convert a failed/pending step into an executable step.

        The previous failure is preserved in the plan through the
        incremented attempt counter and the new execution result.
        """

        if step.status == PlanStepStatus.FAILED:
            step.error = reason

        if step.status in {
            PlanStepStatus.FAILED,
            PlanStepStatus.PENDING,
        }:
            step.start()

    def _reopen_synthesis(
        self,
        plan: AgentPlan,
        reason: str,
    ) -> None:
        """
        Reopen synthesis because the runtime plan changed.

        Synthesis must be executed again after new evidence or a new
        requirement appears.
        """

        synthesis = self._find_synthesis(
            plan,
        )

        if synthesis is None:
            return

        if synthesis.status == PlanStepStatus.COMPLETED:
            synthesis.status = PlanStepStatus.PENDING

        elif synthesis.status == PlanStepStatus.FAILED:
            synthesis.status = PlanStepStatus.PENDING

        synthesis.error = None

        if synthesis.result is not None:
            synthesis.result = None

    def _insert_before_synthesis(
        self,
        plan: AgentPlan,
        step: PlanStep,
    ) -> None:
        synthesis = self._find_synthesis(
            plan,
        )

        if synthesis is None:
            plan.append(
                step,
            )
            return

        index = plan.steps.index(
            synthesis,
        )

        plan.steps.insert(
            index,
            step,
        )

    def _move_before_synthesis(
        self,
        plan: AgentPlan,
        step: PlanStep,
    ) -> None:
        synthesis = self._find_synthesis(
            plan,
        )

        if synthesis is None:
            return

        if step not in plan.steps:
            return

        plan.steps.remove(
            step,
        )

        index = plan.steps.index(
            synthesis,
        )

        plan.steps.insert(
            index,
            step,
        )

    def _move_to_end(
        self,
        plan: AgentPlan,
        step: PlanStep,
    ) -> None:
        if step not in plan.steps:
            return

        plan.steps.remove(
            step,
        )

        plan.steps.append(
            step,
        )

    # ------------------------------------------------------------------
    # Step lookup
    # ------------------------------------------------------------------

    def _find_synthesis(
        self,
        plan: AgentPlan,
    ) -> PlanStep | None:
        for step in plan.steps:
            if step.phase == AgentPhase.SYNTHESIS:
                return step

        return None

    def _find_non_synthesis_step(
        self,
        plan: AgentPlan,
        phase: AgentPhase,
    ) -> PlanStep | None:
        for step in plan.steps:
            if (
                step.phase == phase
                and step.phase != AgentPhase.SYNTHESIS
            ):
                return step

        return None

    def _create_synthesis(
        self,
        plan: AgentPlan,
    ) -> PlanStep:
        synthesis = PlanStep(
            id="synthesis",
            description=self._phase_description(
                AgentPhase.SYNTHESIS,
            ),
            phase=AgentPhase.SYNTHESIS,
        )

        plan.append(
            synthesis,
        )

        return synthesis

    def _revision_step_id(
        self,
        plan: AgentPlan,
        phase: AgentPhase,
    ) -> str:
        base = phase.value

        existing_ids = {
            step.id
            for step in plan.steps
        }

        if base not in existing_ids:
            return base

        revision = plan.revision

        candidate = (
            f"{base}-revision-{revision}"
        )

        counter = 2

        while candidate in existing_ids:
            candidate = (
                f"{base}-revision-{revision}-{counter}"
            )

            counter += 1

        return candidate

    # ------------------------------------------------------------------
    # Descriptions
    # ------------------------------------------------------------------

    def _phase_description(
        self,
        phase: AgentPhase,
    ) -> str:
        descriptions = {
            AgentPhase.RESEARCH: (
                "Дополнить исследование."
            ),
            AgentPhase.EXECUTION: (
                "Продолжить выполнение задачи."
            ),
            AgentPhase.DEBUGGING: (
                "Исправить обнаруженную проблему."
            ),
            AgentPhase.VERIFICATION: (
                "Проверить результат."
            ),
            AgentPhase.REFLECTION: (
                "Переосмыслить результат и устранить недочёты."
            ),
            AgentPhase.SYNTHESIS: (
                "Сформировать итоговый ответ."
            ),
        }

        return descriptions.get(
            phase,
            f"Выполнить фазу {phase.value}.",
        )

    # Production verbs: the task creates or changes code.
    # Read-only verbs (read, find, show) deliberately do not
    # trigger a coder hint: such steps stay with the main agent.
    _CODING_SIGNALS = frozenset({
        "напиши",
        "написать",
        "создай",
        "создать",
        "реализуй",
        "реализовать",
        "рефактор",
        "рефакторинг",
        "исправь",
        "исправить",
        "почини",
        "покрой тестами",
        "напиши тесты",
        "код",
        "тест",
        "баг",
        "write code",
        "implement",
        "create",
        "refactor",
        "fix",
        "write tests",
        "code",
        "tests",
        "bug",
    })

    @classmethod
    def _hint_for_phase(
        cls,
        phase: AgentPhase,
        objective: str,
    ) -> DelegationHint | None:
        """
        Advisory delegation hint for one plan step.

        Rendered into context by ContextAssembler. Never acted
        upon by the runtime; the model decides.
        """

        if phase is AgentPhase.RESEARCH:
            return DelegationHint(
                role="researcher",
                power=SubagentPower.LOW,
                reason=(
                    "independent source traversal with "
                    "isolated context"
                ),
            )

        if phase is AgentPhase.VERIFICATION:
            return DelegationHint(
                role="reviewer",
                power=SubagentPower.MEDIUM,
                reason=(
                    "independent result checking against "
                    "the contract"
                ),
            )

        if (
            phase is AgentPhase.EXECUTION
            and cls._has_coding_signals(objective)
        ):
            return DelegationHint(
                role="coder",
                power=SubagentPower.MEDIUM,
                reason=(
                    "focused code writing with a limited "
                    "tool set"
                ),
            )

        return None

    @classmethod
    def _has_coding_signals(
        cls,
        objective: str,
    ) -> bool:
        lowered = objective.lower()

        return any(
            signal in lowered
            for signal in cls._CODING_SIGNALS
        )

    def _research_description(
        self,
        contract: TaskContract,
    ) -> str:
        research = contract.research

        if research is None:
            return (
                "Провести необходимое исследование."
            )

        return (
            "Исследовать указанные источники и "
            "достичь требуемого покрытия."
        )

    # ------------------------------------------------------------------
    # Plan validation
    # ------------------------------------------------------------------

    def _validate_plan(
        self,
        plan: AgentPlan,
    ) -> None:
        """
        Validate planner-owned invariants.

        This deliberately raises ValueError rather than silently fixing
        a corrupted plan.
        """

        if not plan.steps:
            raise ValueError(
                "Execution plan must contain at least one step.",
            )

        # --------------------------------------------------------------
        # Unique IDs
        # --------------------------------------------------------------

        ids = [
            step.id
            for step in plan.steps
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Execution plan contains duplicate step IDs.",
            )

        # --------------------------------------------------------------
        # Exactly one synthesis
        # --------------------------------------------------------------

        synthesis_steps = [
            step
            for step in plan.steps
            if step.phase == AgentPhase.SYNTHESIS
        ]

        if len(synthesis_steps) != 1:
            raise ValueError(
                "Execution plan must contain exactly one synthesis step.",
            )

        if plan.steps[-1] is not synthesis_steps[0]:
            raise ValueError(
                "Synthesis must be the final execution step.",
            )

        # --------------------------------------------------------------
        # Current step consistency
        # --------------------------------------------------------------

        current = plan.current_step()

        active_steps = [
            step
            for step in plan.steps
            if step.status == PlanStepStatus.ACTIVE
        ]

        if current is None:
            if active_steps:
                raise ValueError(
                    "Plan has an active step but no current_step_id.",
                )

            return

        if len(active_steps) != 1:
            raise ValueError(
                "Execution plan must contain exactly one ACTIVE step.",
            )

        if active_steps[0] is not current:
            raise ValueError(
                "current_step_id does not point to the ACTIVE step.",
            )

        # --------------------------------------------------------------
        # Steps after active cannot already be active/failed.
        # --------------------------------------------------------------

        current_index = plan.steps.index(
            current,
        )

        for step in plan.steps[
            current_index + 1 :
        ]:
            if step.status == PlanStepStatus.ACTIVE:
                raise ValueError(
                    "A step after the current step is ACTIVE.",
                )

            if step.status == PlanStepStatus.FAILED:
                raise ValueError(
                    "FAILED steps must be normalized before execution.",
                )

    # ------------------------------------------------------------------
    # Objective extraction
    # ------------------------------------------------------------------

    def _extract_objective(
        self,
        prompt: str,
    ) -> str:
        cleaned = self._remove_urls(
            prompt,
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        ).strip()

        cleaned = self._strip_leading_greeting(
            cleaned,
        )

        objective = self._find_verb_object(
            cleaned,
        )

        if objective:
            return objective

        if cleaned:
            return cleaned

        if self._has_urls(prompt):
            return (
                "Исследовать указанные в источниках материалы."
            )

        return "Выполнить поставленную задачу."

    def _strip_leading_greeting(
        self,
        text: str,
    ) -> str:
        words = text.split()

        if not words:
            return ""

        first = words[0].lower().strip(
            ",.!?:;—-",
        )

        if first in self._LEADING_GREETINGS:
            return " ".join(
                words[1:],
            ).strip()

        return text

    def _find_verb_object(
        self,
        text: str,
    ) -> str | None:
        for pattern in self._VERB_PATTERNS:
            cleaned = re.sub(
                pattern,
                "",
                text,
                count=1,
                flags=re.IGNORECASE,
            )

            if cleaned != text:
                cleaned = re.sub(
                    r"\s+",
                    " ",
                    cleaned,
                ).strip()

                if cleaned:
                    return cleaned

        return None

    # ------------------------------------------------------------------
    # Research extraction
    # ------------------------------------------------------------------

    def _extract_research(
        self,
        prompt: str,
    ) -> ResearchContract | None:
        """
        Convert URLs + research intent into a deterministic
        ResearchContract.

        Rules:

        1. No URL -> no automatic web research contract.
        2. Explicit page/depth requirements override defaults.
        3. Deep research intent gets stronger defaults.
        4. Every root URL must be fetched.
        5. Multiple root URLs always require at least one fetched page
           per root.
        """

        urls = self._extract_urls(
            prompt,
        )

        if not urls:
            return None

        explicit_page_count = (
            self._extract_min_pages(
                prompt,
            )
        )

        explicit_min_depth = (
            self._extract_min_depth(
                prompt,
            )
        )

        deep_research = (
            self._is_deep_research_request(
                prompt,
            )
        )

        if explicit_page_count is not None:
            min_pages = explicit_page_count
        elif deep_research:
            min_pages = self._DEEP_RESEARCH_MIN_PAGES
        else:
            min_pages = self._DEFAULT_RESEARCH_MIN_PAGES

        # With multiple root URLs, at least one page from every root
        # must be fetched because require_all_roots=True.
        min_pages = max(
            min_pages,
            len(urls),
        )

        if explicit_min_depth is not None:
            min_depth = explicit_min_depth
        elif deep_research:
            min_depth = self._DEEP_RESEARCH_MIN_DEPTH
        else:
            min_depth = self._DEFAULT_RESEARCH_MIN_DEPTH

        return ResearchContract(
            root_urls=tuple(urls),
            required_urls=(),
            min_pages=min_pages,
            min_depth=min_depth,
            require_all_roots=True,
        )

    def _is_deep_research_request(
        self,
        prompt: str,
    ) -> bool:
        return bool(
            self._DEEP_RESEARCH_INTENT_PATTERN.search(
                prompt,
            ),
        )

    def _extract_urls(
        self,
        prompt: str,
    ) -> tuple[str, ...]:
        raw_urls = self._URL_PATTERN.findall(
            prompt,
        )

        normalized: list[str] = []
        seen: set[str] = set()

        for raw_url in raw_urls:
            raw_url = raw_url.rstrip(
                ".,;:!?)]}",
            )

            try:
                canonical = canonicalize_url(
                    raw_url,
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if not canonical:
                continue

            parsed = urlsplit(
                canonical,
            )

            if parsed.scheme not in {
                "http",
                "https",
            }:
                continue

            if not parsed.netloc:
                continue

            if canonical in seen:
                continue

            seen.add(
                canonical,
            )

            normalized.append(
                canonical,
            )

        return tuple(
            normalized,
        )

    def _extract_min_pages(
        self,
        prompt: str,
    ) -> int | None:
        match = self._PAGE_COUNT_PATTERN.search(
            prompt,
        )

        if match is None:
            return None

        raw_value = (
            match.group(
                "prefix_value",
            )
            or match.group(
                "suffix_value",
            )
        )

        if raw_value is None:
            return None

        try:
            value = int(
                raw_value,
            )
        except (
            TypeError,
            ValueError,
        ):
            return None

        return max(
            1,
            value,
        )

    def _extract_min_depth(
        self,
        prompt: str,
    ) -> int | None:
        match = self._DEPTH_PATTERN.search(
            prompt,
        )

        if match is None:
            return None

        raw_value = match.group(
            "value",
        )

        if raw_value is None:
            return None

        try:
            value = int(
                raw_value,
            )
        except (
            TypeError,
            ValueError,
        ):
            return None

        return max(
            0,
            value,
        )

    def _has_urls(
        self,
        prompt: str,
    ) -> bool:
        return bool(
            self._URL_PATTERN.search(
                prompt,
            ),
        )

    def _remove_urls(
        self,
        prompt: str,
    ) -> str:
        return self._URL_PATTERN.sub(
            " ",
            prompt,
        )