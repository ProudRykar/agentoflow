from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class PluginCapability(str):
    TOOLS = "tools"
    SKILLS = "skills"
    WORKFLOWS = "workflows"
    COMMANDS = "commands"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    EXTERNAL_API = "external_api"


@dataclass(slots=True, frozen=True)
class PluginMetadata:
    name: str
    version: str
    description: str = ""
    author: str = ""
    homepage: str = ""
    license: str = ""
    capabilities: tuple[PluginCapability, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "homepage": self.homepage,
            "license": self.license,
            "capabilities": list(self.capabilities),
        }


class PluginState(str):
    DISCOVERED = "discovered"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    DEACTIVATED = "deactivated"
    FAILED = "failed"
    DISABLED = "disabled"
    UNLOADED = "unloaded"


@dataclass(slots=True)
class Plugin:
    metadata: PluginMetadata
    state: PluginState = PluginState.DISCOVERED
    module: Any = None
    instance: Any = None
    error: str | None = None
    source_path: Path | None = None

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def version(self) -> str:
        return self.metadata.version

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "state": self.state,
            "error": self.error,
        }


class PluginError(Exception):
    pass


class PluginNotFoundError(PluginError):
    pass


class PluginAlreadyRegisteredError(PluginError):
    pass


class PluginLoadError(PluginError):
    def __init__(self, name: str, message: str) -> None:
        self.name = name
        super().__init__(f"Failed to load plugin '{name}': {message}")


class PluginInitializationError(PluginError):
    def __init__(self, name: str, message: str) -> None:
        self.name = name
        super().__init__(f"Failed to initialize plugin '{name}': {message}")