from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class PluginEventType(StrEnum):
    DISCOVERED = "plugin.discovered"
    LOADED = "plugin.loaded"
    INITIALIZED = "plugin.initialized"
    ACTIVATED = "plugin.activated"
    DEACTIVATED = "plugin.deactivated"
    FAILED = "plugin.failed"
    UNLOADED = "plugin.unloaded"


@dataclass(slots=True, frozen=True)
class PluginEvent:
    event_type: PluginEventType
    plugin_name: str
    plugin_version: str = "0.0.0"
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
            "plugin_name": self.plugin_name,
            "plugin_version": self.plugin_version,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }