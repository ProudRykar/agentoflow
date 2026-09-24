from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.context.checkpoint import (
    TaskCheckpoint,
    build_checkpoint,
)
from core.context.context_budget import ContextBudget
from core.context.context_item import (
    ContextItem,
    ContextSource,
)
from core.context.conversation import ConversationManager
from core.context.history import (
    HistoryItem,
    HistoryKind,
    HistoryStore,
    InMemoryHistoryStore,
)
from core.context.llm_request import LLMRequestContext
from core.context.tokens import (
    ApproximateTokenCounter,
    TokenCounter,
)

if TYPE_CHECKING:
    from core.entities.models.agent_orchestrator import (
        AgentOrchestrator,
    )

HISTORY_RESTORE_LIMIT = 5


@dataclass(slots=True, frozen=True)
class RolloverResult:
    """Outcome of one context-window rollover."""

    checkpoint: TaskCheckpoint
    window_id: str
    previous_window_id: str
    history_context: tuple[ContextItem, ...]
    cleared_messages: int


class ContextController:
    """Owns windows, checkpoints, history, and rollover gates.

    Invariants enforced here, not by convention:

    - CHECKPOINT SAVE FAILED -> NEW CONTEXT FORBIDDEN.
    - CHECKPOINT RESTORE FAILED -> RUN BLOCKED.

    Rollover order is therefore: save -> read-back verify ->
    record -> clear -> new window. Nothing is cleared before
    the checkpoint is proven retrievable.
    """

    def __init__(
        self,
        budget: ContextBudget | None = None,
        counter: TokenCounter | None = None,
        history: HistoryStore | None = None,
    ) -> None:
        self._budget = budget or ContextBudget()
        self._counter = (
            counter or ApproximateTokenCounter()
        )
        self._history: HistoryStore = (
            history or InMemoryHistoryStore()
        )
        self._checkpoints: dict[str, TaskCheckpoint] = {}
        self._window_counter: int = 0

    @property
    def budget(self) -> ContextBudget:
        return self._budget

    @property
    def counter(self) -> TokenCounter:
        return self._counter

    @property
    def history(self) -> HistoryStore:
        return self._history

    @property
    def current_window_id(self) -> str:
        return f"window-{max(self._window_counter, 1):02d}"

    def new_window(self) -> str:
        self._window_counter += 1

        return f"window-{self._window_counter:02d}"

    # ==================================================================
    # Checkpoints
    # ==================================================================

    def save_checkpoint(
        self,
        orchestrator: AgentOrchestrator,
    ) -> TaskCheckpoint:
        try:
            checkpoint = build_checkpoint(orchestrator)
        except Exception as exc:
            raise RuntimeError(
                "Checkpoint save failed, "
                "new context is forbidden: "
                f"{exc}"
            ) from exc

        self._checkpoints[checkpoint.task_id] = checkpoint

        return checkpoint

    def restore_checkpoint(
        self,
        task_id: str,
    ) -> TaskCheckpoint:
        try:
            return self._checkpoints[task_id]
        except KeyError as exc:
            raise RuntimeError(
                "Checkpoint restore failed, "
                "run is blocked: "
                f"no checkpoint for {task_id}"
            ) from exc

    # ==================================================================
    # History
    # ==================================================================

    def record(
        self,
        task_id: str,
        run_id: str,
        kind: HistoryKind,
        content: str,
        reference: str | None = None,
    ) -> HistoryItem:
        return self._history.append(
            task_id,
            run_id,
            self.current_window_id,
            kind,
            content,
            reference,
        )

    # ==================================================================
    # Budget
    # ==================================================================

    def request_tokens(
        self,
        request: LLMRequestContext,
    ) -> int:
        total = 0

        for message in request.messages:
            total += self._counter.count(
                str(message.get("content", ""))
            )
            total += 16

        return total

    def is_over_budget(
        self,
        request: LLMRequestContext,
    ) -> bool:
        return (
            self.request_tokens(request)
            > self._budget.available
        )

    # ==================================================================
    # Rollover
    # ==================================================================

    def rollover(
        self,
        orchestrator: AgentOrchestrator,
        conversation: ConversationManager,
    ) -> RolloverResult:
        """
        Checkpoint, verify, then open a fresh context window.

        The trailing user message (the current instruction) is
        preserved; everything older is cleared. A one-shot
        history selection of the previous window is returned
        for the next assembled request.
        """

        # Gate 1: save must succeed before anything is cleared.
        checkpoint = self.save_checkpoint(orchestrator)

        # Gate 2: the checkpoint must be retrievable, or the
        # run is blocked instead of continuing blind.
        read_back = self.restore_checkpoint(
            checkpoint.task_id
        )

        if read_back != checkpoint:
            raise RuntimeError(
                "Checkpoint restore failed, run is blocked: "
                "stored checkpoint does not match"
            )

        task_id = checkpoint.task_id
        previous_window = self.current_window_id

        history_context = self._select_restore_context(
            task_id,
            previous_window,
        )

        self.record(
            task_id=task_id,
            run_id=orchestrator.execution_context.run_id,
            kind=HistoryKind.CHECKPOINT,
            content=checkpoint.render(),
            reference=previous_window,
        )

        dialogue = conversation.dialogue()

        keep: list[dict[str, Any]] = []

        if dialogue and dialogue[-1].get("role") == "user":
            keep = [dialogue[-1]]

        conversation.clear()

        for message in keep:
            conversation.add_message(message)

        window_id = self.new_window()

        return RolloverResult(
            checkpoint=checkpoint,
            window_id=window_id,
            previous_window_id=previous_window,
            history_context=history_context,
            cleared_messages=len(dialogue) - len(keep),
        )

    def _select_restore_context(
        self,
        task_id: str,
        window_id: str,
    ) -> tuple[ContextItem, ...]:
        items = self._history.window_items(
            task_id,
            window_id,
        )

        textual = [
            item
            for item in items
            if item.kind
            in (
                HistoryKind.USER_MESSAGE,
                HistoryKind.ASSISTANT_MESSAGE,
            )
            and item.content.strip()
        ]

        selected = textual[-HISTORY_RESTORE_LIMIT:]

        return tuple(
            ContextItem(
                source=ContextSource.HISTORY,
                content=item.content,
                reference=item.window_id,
            )
            for item in selected
        )
