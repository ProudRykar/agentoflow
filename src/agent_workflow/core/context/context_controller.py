from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_workflow.core.context.checkpoint import (
    TaskCheckpoint,
    build_checkpoint,
)
from agent_workflow.core.context.context_budget import ContextBudget
from agent_workflow.core.context.context_item import (
    ContextItem,
    ContextSource,
)
from agent_workflow.core.context.conversation import ConversationManager
from agent_workflow.core.context.history import (
    HistoryItem,
    HistoryKind,
    HistoryStore,
    InMemoryHistoryStore,
)
from agent_workflow.core.context.llm_request import LLMRequestContext
from agent_workflow.core.context.tokens import (
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
    """Outcome of one context-window rollover.

    ``checkpoint`` is None when the window could not be closed
    because there was nothing to checkpoint. That happens for a
    session restored from disk: it has a transcript but no live task
    anchor. Such a rollover is a no-op on purpose, because clearing
    a window without a checkpoint would destroy context that cannot
    be reconstructed.
    """

    checkpoint: TaskCheckpoint | None
    window_id: str
    previous_window_id: str
    history_context: tuple[ContextItem, ...]
    cleared_messages: int
    performed: bool = True


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

    def adopt_checkpoint(
        self,
        checkpoint: TaskCheckpoint,
    ) -> None:
        """Re-seed a checkpoint loaded from disk.

        ``restore_checkpoint`` deliberately raises for an unknown
        task: that guard exists so a live run cannot continue blind.
        After a restart there is nothing to be blind about, the
        checkpoint simply arrived with the session.
        """

        self._checkpoints[checkpoint.task_id] = checkpoint

    def adopt_history(
        self,
        items: tuple,
    ) -> None:
        """Re-seed history items loaded from disk."""

        for item in items:
            self._history.append(
                task_id=item.task_id,
                run_id=item.run_id,
                window_id=item.window_id,
                kind=item.kind,
                content=item.content,
                reference=item.reference,
            )

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

        previous_window = self.current_window_id

        # Gate 1: save must succeed before anything is cleared.
        try:
            checkpoint = self.save_checkpoint(orchestrator)
        except RuntimeError:
            # A restored session has a transcript but no live task
            # anchor, so there is nothing to checkpoint. Clearing the
            # window anyway would throw away context that cannot be
            # rebuilt, so the rollover is skipped and the caller
            # keeps working with what it has.
            return RolloverResult(
                checkpoint=None,
                window_id=previous_window,
                previous_window_id=previous_window,
                history_context=(),
                cleared_messages=0,
                performed=False,
            )

        # Gate 2: the checkpoint must be retrievable, or the
        # run is blocked instead of continuing blind.
        read_back = self.restore_checkpoint(checkpoint.task_id)

        if read_back != checkpoint:
            raise RuntimeError(
                "Checkpoint restore failed, run is blocked: "
                "stored checkpoint does not match"
            )

        task_id = checkpoint.task_id

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

        # The instruction the user gave is the last thing that must
        # survive. It is searched for rather than assumed to be the
        # final message: rollover runs at the top of an iteration,
        # where the window usually ends with a tool result, so
        # checking only dialogue[-1] silently dropped the prompt.
        keep = _trailing_instruction(dialogue)

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
            performed=True,
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
                HistoryKind.TOOL_RESULT,
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


def _trailing_instruction(
    dialogue: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The most recent user instruction in the window, if any.

    Everything after it is dropped, so the surviving slice can never
    start with an orphaned tool result: a user message is never a
    tool result.
    """

    for index in range(len(dialogue) - 1, -1, -1):
        message = dialogue[index]

        if message.get("role") == "user":
            return [message]

    return []
