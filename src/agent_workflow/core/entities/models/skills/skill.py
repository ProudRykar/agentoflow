from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class Skill:
    name: str
    description: str
    version: str = "1.0.0"
    instructions: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Skill name cannot be empty")
        if not self.description:
            raise ValueError("Skill description cannot be empty")

    @property
    def display_name(self) -> str:
        return self.name

    def summary(self) -> str:
        lines = self.description.strip().splitlines()
        return lines[0] if lines else self.description


@dataclass(slots=True, frozen=True)
class SkillMetadata:
    name: str
    description: str
    version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "metadata": self.metadata,
        }