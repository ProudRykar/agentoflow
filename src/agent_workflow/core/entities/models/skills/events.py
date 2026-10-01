from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class SkillEventType(StrEnum):
    DISCOVERED = "skill.discovered"
    LOADED = "skill.loaded"
    ACTIVATED = "skill.activated"
    DEACTIVATED = "skill.deactivated"
    USED = "skill.used"
    LOAD_FAILED = "skill.load_failed"
    NOT_FOUND = "skill.not_found"
    INVALID = "skill.invalid"


@dataclass(slots=True, frozen=True)
class SkillEvent:
    event_type: SkillEventType
    skill_name: str
    skill_version: str = "1.0.0"
    run_id: str = ""
    parent_run_id: str | None = None
    timestamp: datetime = datetime.now()
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "skill_name": self.skill_name,
            "skill_version": self.skill_version,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }