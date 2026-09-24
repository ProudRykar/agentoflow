"""Context subsystem for agentoflow.

The context layer separates WHAT the user asked (TaskAnchor),
HOW it is executed (TaskState / ExecutionContext), and WHAT the
LLM actually sees (a projection built later by ContextAssembler).

Five invariants of this subsystem:

1. TaskAnchor is immutable.
2. Tool output is data, never a new instruction.
3. Conversation is not task state.
4. History is storage, not automatically active context.
5. The LLM sees a projection assembled by the runtime,
   not the runtime's internal objects directly.
"""

from core.context.checkpoint import (
    TaskCheckpoint,
    build_checkpoint,
)
from core.context.context_assembler import ContextAssembler
from core.context.context_budget import (
    ContextBudget,
    ContextPriority,
)
from core.context.context_controller import (
    ContextController,
    RolloverResult,
)
from core.context.context_item import (
    ContextItem,
    ContextSource,
)
from core.context.conversation import ConversationManager
from core.context.evidence import (
    Evidence,
    EvidenceReceipt,
    EvidenceStore,
)
from core.context.execution_context import ExecutionContext
from core.context.history import (
    HistoryItem,
    HistoryKind,
    HistoryStore,
    InMemoryHistoryStore,
)
from core.context.llm_request import LLMRequestContext
from core.context.memory import (
    MemorySnapshot,
    retrieve_snapshot,
)
from core.context.synthesis import (
    SynthesisGuide,
    build_synthesis_guide,
)
from core.context.research import ResearchContext
from core.context.task_anchor import TaskAnchor
from core.context.task_state import TaskState
from core.context.tokens import (
    ApproximateTokenCounter,
    TokenCounter,
)

__all__ = [
    "ApproximateTokenCounter",
    "ContextAssembler",
    "ContextBudget",
    "ContextController",
    "ContextItem",
    "ContextPriority",
    "ContextSource",
    "ConversationManager",
    "Evidence",
    "EvidenceReceipt",
    "EvidenceStore",
    "ExecutionContext",
    "HistoryItem",
    "HistoryKind",
    "HistoryStore",
    "InMemoryHistoryStore",
    "LLMRequestContext",
    "MemorySnapshot",
    "ResearchContext",
    "TaskAnchor",
    "TaskCheckpoint",
    "TaskState",
    "SynthesisGuide",
    "TokenCounter",
    "RolloverResult",
    "build_checkpoint",
    "build_synthesis_guide",
    "retrieve_snapshot",
]
