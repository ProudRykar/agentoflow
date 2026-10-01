from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agent_workflow.core.entities.models.skills.events import (
    SkillEvent,
    SkillEventType,
)
from agent_workflow.core.entities.models.skills.loader import (
    SkillLoader,
)
from agent_workflow.core.entities.models.skills.registry import (
    SkillRegistry,
)
from agent_workflow.core.entities.models.skills.skill import Skill


SkillEventCallback = Callable[
    [SkillEvent],
    Awaitable[None],
]


class SkillManager:
    def __init__(
        self,
        skills_directory: Path,
        registry: SkillRegistry | None = None,
        on_event: SkillEventCallback | None = None,
    ) -> None:
        self._skills_directory = skills_directory
        self._registry = registry or SkillRegistry()
        self._loader = SkillLoader(self._registry)
        self._on_event = on_event

        self._active_skills: set[str] = set()
        self._loaded_skills: set[str] = set()
        self._used_skills: set[str] = set()

        self._directory_marker = (
            f"dir:{self._skills_directory}"
        )

    @property
    def registry(self) -> SkillRegistry:
        return self._registry

    @property
    def active_skills(self) -> tuple[str, ...]:
        return tuple(sorted(self._active_skills))

    @property
    def loaded_skills(self) -> tuple[str, ...]:
        return tuple(sorted(self._loaded_skills))

    @property
    def used_skills(self) -> tuple[str, ...]:
        return tuple(sorted(self._used_skills))

    async def _emit(
        self,
        event_type: SkillEventType,
        skill_name: str,
        skill_version: str = "1.0.0",
        run_id: str = "",
        parent_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._on_event is None:
            return

        event = SkillEvent(
            event_type=event_type,
            skill_name=skill_name,
            skill_version=skill_version,
            run_id=run_id,
            parent_run_id=parent_run_id,
            metadata=metadata or {},
        )

        await self._on_event(event)

    async def discover(self) -> list[Skill]:
        """Scan the skills directory and register what is found.

        Idempotent: skills already registered by a plugin or by
        a previous call are left alone, and a file whose skill is
        present upstream is not re-registered.
        """

        if self._registry.is_scanned(self._directory_marker):
            return self._registry.list()

        # Load discovered skills into the registry. Registration
        # is guarded by _directory_marker so repeated calls do
        # not raise SkillAlreadyRegisteredError.
        for skill in self._loader.load_directory(
            self._skills_directory,
        ):
            await self._emit(
                SkillEventType.DISCOVERED,
                skill.name,
                skill.version,
                metadata=skill.metadata,
            )

        self._registry.mark_scanned(self._directory_marker)

        return self._registry.list()

    async def list_available(self) -> list[Skill]:
        await self.discover()
        return self._registry.list()

    async def list_available_metadata(self) -> list[dict[str, Any]]:
        await self.discover()
        return [meta.to_dict() for meta in self._registry.list_metadata()]

    async def load(
        self,
        name: str,
        run_id: str = "",
        parent_run_id: str | None = None,
    ) -> Skill:
        if name in self._loaded_skills:
            skill = self._registry.get(name)
            await self._emit(
                SkillEventType.LOADED,
                skill.name,
                skill.version,
                run_id=run_id,
                parent_run_id=parent_run_id,
                metadata={"cached": True},
            )
            return skill

        # Auto-discover if not yet registered
        if not self._registry.has(name):
            await self.discover()

        skill = self._registry.get(name)
        self._loaded_skills.add(name)

        await self._emit(
            SkillEventType.LOADED,
            skill.name,
            skill.version,
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

        return skill

    async def activate(
        self,
        name: str,
        run_id: str = "",
        parent_run_id: str | None = None,
    ) -> Skill:
        skill = await self.load(name, run_id, parent_run_id)

        if name in self._active_skills:
            await self._emit(
                SkillEventType.ACTIVATED,
                skill.name,
                skill.version,
                run_id=run_id,
                parent_run_id=parent_run_id,
                metadata={"already_active": True},
            )
            return skill

        self._active_skills.add(name)

        await self._emit(
            SkillEventType.ACTIVATED,
            skill.name,
            skill.version,
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

        return skill

    async def deactivate(
        self,
        name: str,
        run_id: str = "",
        parent_run_id: str | None = None,
    ) -> None:
        if name not in self._active_skills:
            return

        skill = self._registry.get(name)
        self._active_skills.discard(name)

        await self._emit(
            SkillEventType.DEACTIVATED,
            skill.name,
            skill.version,
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

    def mark_used(
        self,
        name: str,
        run_id: str = "",
        parent_run_id: str | None = None,
    ) -> None:
        if name not in self._active_skills:
            return

        if name in self._used_skills:
            return

        skill = self._registry.get(name)
        self._used_skills.add(name)

        import asyncio

        asyncio.create_task(
            self._emit(
                SkillEventType.USED,
                skill.name,
                skill.version,
                run_id=run_id,
                parent_run_id=parent_run_id,
            )
        )

    def get_active_instructions(self) -> str:
        if not self._active_skills:
            return ""

        parts: list[str] = []

        for name in sorted(self._active_skills):
            skill = self._registry.get(name)
            parts.append(
                f"# Skill: {skill.name} (v{skill.version})\n\n{skill.instructions}"
            )

        return "\n\n---\n\n".join(parts)

    def is_active(self, name: str) -> bool:
        return name in self._active_skills

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded_skills

    def is_used(self, name: str) -> bool:
        return name in self._used_skills

    def clear_run_state(self) -> None:
        self._active_skills.clear()
        self._loaded_skills.clear()
        self._used_skills.clear()