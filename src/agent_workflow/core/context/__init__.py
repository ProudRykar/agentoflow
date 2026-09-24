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

from agent_workflow.core.context.checkpoint import (
    TaskCheckpoint,
    build_checkpoint,
)
from agent_workflow.core.context.context_assembler import ContextAssembler
from agent_workflow.core.context.context_budget import (
    ContextBudget,
    ContextPriority,
)
from agent_workflow.core.context.context_controller import (
    ContextController,
    RolloverResult,
)
from agent_workflow.core.context.context_item import (
    ContextItem,
    ContextSource,
)
from agent_workflow.core.context.conversation import ConversationManager
from agent_workflow.core.context.evidence import (
    Evidence,
    EvidenceReceipt,
    EvidenceStore,
)
from agent_workflow.core.context.execution_context import ExecutionContext
from agent_workflow.core.context.history import (
    HistoryItem,
    HistoryKind,
    HistoryStore,
    InMemoryHistoryStore,
)
from agent_workflow.core.context.llm_request import LLMRequestContext
from agent_workflow.core.context.memory import (
    MemorySnapshot,
    retrieve_snapshot,
)
from agent_workflow.core.context.synthesis import (
    SynthesisGuide,
    build_synthesis_guide,
)
from agent_workflow.core.context.research import ResearchContext
from agent_workflow.core.context.task_anchor import TaskAnchor
from agent_workflow.core.context.task_state import TaskState
from agent_workflow.core.context.tokens import (
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
