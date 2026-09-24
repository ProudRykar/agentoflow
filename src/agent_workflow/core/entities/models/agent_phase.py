from __future__ import annotations

from enum import StrEnum


class AgentPhase(StrEnum):
    IDLE = "idle"
    PLANNING = "planning"
    RESEARCH = "research"
    EXECUTION = "execution"
    DEBUGGING = "debugging"
    VERIFICATION = "verification"
    REFLECTION = "reflection"
    SYNTHESIS = "synthesis"
    WAITING_APPROVAL = "waiting_approval"
    BLOCKED = "blocked"
    COMPLETED = "completed"