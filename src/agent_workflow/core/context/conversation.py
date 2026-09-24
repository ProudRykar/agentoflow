"""Conversation ownership.

``ConversationManager`` is the new canonical name for what used
to be ``ContextManager``: it owns ONLY the dialogue
(user / assistant / tool messages) and is NOT task state.
"""

from agent_workflow.core.entities.models.context_manager import (
    ContextManager,
)

ConversationManager = ContextManager

__all__ = [
    "ConversationManager",
]
