from __future__ import annotations

from dataclasses import dataclass

from agent_workflow.core.entities.models.agent_phase import AgentPhase


@dataclass(slots=True)
class AgentState:
    phase: AgentPhase = AgentPhase.IDLE

    iteration: int = 0
    tool_calls: int = 0

    last_tool_name: str | None = None
    last_tool_succeeded: bool | None = None
    last_error: str | None = None

    started: bool = False
    finished: bool = False

    phase_reason: str = ""

    def reset(self) -> None:
        self.phase = AgentPhase.IDLE

        self.iteration = 0
        self.tool_calls = 0

        self.last_tool_name = None
        self.last_tool_succeeded = None
        self.last_error = None

        self.started = False
        self.finished = False

        self.phase_reason = ""