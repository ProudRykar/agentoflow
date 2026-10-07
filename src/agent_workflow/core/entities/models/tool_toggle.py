from __future__ import annotations

import logging

from agent_workflow.core.entities.models.tool import Tool
from agent_workflow.core.entities.models.tool_registry import ToolRegistry
from agent_workflow.core.infrastructure.config import ToolsConfig


logger = logging.getLogger(__name__)


class ToolToggle:
    """Turns tools off and on without a restart.

    Disabling a tool has to remove it from the registry, because a
    tool the model can still call is not disabled. Re-enabling it
    needs the tool object to still exist somewhere, so every tool the
    session knows about is retained here even while it is hidden.

    This mirrors how MCP servers are controlled at runtime: the
    registry reflects what the model may call right now, and the
    toggle owns what could be turned back on.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        config: ToolsConfig | None = None,
    ) -> None:
        self._registry = registry
        self._config = config or ToolsConfig()

        # name -> tool, for every tool seen, hidden or not. MCP tools
        # arrive after the server connects, so this is filled in
        # incrementally rather than from the constructor.
        self._known: dict[str, Tool] = {
            tool.name: tool for tool in registry.all()
        }

        self._disabled: set[str] = set(
            self._config.disabled
        )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def observe(self) -> None:
        """Record tools that appeared since the last look.

        Without this a tool registered later -- an MCP tool, a
        plugin -- could not be switched back on, because its object
        was never retained.
        """

        for tool in self._registry.all():
            self._known.setdefault(tool.name, tool)

    def known_tools(self) -> tuple[Tool, ...]:
        self.observe()

        return tuple(
            self._known[name] for name in sorted(self._known)
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def known_tools_by_name(self) -> dict[str, Tool]:
        self.observe()

        return dict(self._known)

    def is_disabled(self, name: str) -> bool:
        return name in self._disabled

    def disabled_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._disabled))

    def enabled_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in sorted(self._known)
            if name not in self._disabled
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def disable(self, name: str) -> bool:
        """Remove a tool from the registry.

        Returns False when the name is not a tool this session knows,
        so the caller can answer 404 instead of silently succeeding.
        """

        self.observe()

        tool = self._known.get(name)

        if tool is None:
            logger.warning(
                "Cannot disable %r: not a known tool", name
            )
            return False

        self._disabled.add(name)

        if self._registry.has(name):
            self._registry.unregister(name)

        return True

    def enable(self, name: str) -> bool:
        """Put a previously disabled tool back into the registry."""

        self.observe()

        tool = self._known.get(name)

        if tool is None:
            logger.warning(
                "Cannot enable %r: not a known tool", name
            )
            return False

        self._disabled.discard(name)

        if not self._registry.has(name):
            self._registry.register(tool)

        return True

    def apply(self) -> None:
        """Remove everything currently disabled.

        Called after late registration, since a tool that arrives
        disabled must not appear.
        """

        for name in self._disabled:
            if self._registry.has(name):
                self._registry.unregister(name)

    def config(self) -> ToolsConfig:
        """The current set, shaped for persisting."""

        return ToolsConfig(
            disabled=tuple(sorted(self._disabled))
        )