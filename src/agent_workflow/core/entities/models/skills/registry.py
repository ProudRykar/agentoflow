from __future__ import annotations

from typing import Any

from agent_workflow.core.entities.models.skills.skill import Skill, SkillMetadata


class SkillRegistryError(Exception):
    pass


class SkillNotFoundError(SkillRegistryError):
    pass


class SkillAlreadyRegisteredError(SkillRegistryError):
    pass


class InvalidSkillError(SkillRegistryError):
    pass


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

        # Opaque markers recording which sources have already been
        # scanned. Prevents duplicate registration on re-scan.
        self._scanned: set[str] = set()

    def mark_scanned(
        self,
        marker: str,
    ) -> None:
        self._scanned.add(marker)

    def is_scanned(
        self,
        marker: str,
    ) -> bool:
        return marker in self._scanned

    def register(
        self,
        skill: Skill,
    ) -> None:
        if skill.name in self._skills:
            raise SkillAlreadyRegisteredError(
                f"Skill '{skill.name}' is already registered"
            )

        self._skills[skill.name] = skill

    def unregister(
        self,
        name: str,
    ) -> None:
        if name not in self._skills:
            raise SkillNotFoundError(
                f"Skill '{name}' is not registered"
            )

        del self._skills[name]

    def get(
        self,
        name: str,
    ) -> Skill:
        try:
            return self._skills[name]
        except KeyError:
            raise SkillNotFoundError(
                f"Skill '{name}' is not registered"
            ) from None

    def has(
        self,
        name: str,
    ) -> bool:
        return name in self._skills

    def list(
        self,
    ) -> tuple[Skill, ...]:
        return tuple(self._skills.values())

    def list_metadata(
        self,
    ) -> tuple[SkillMetadata, ...]:
        return tuple(
            SkillMetadata(
                name=skill.name,
                description=skill.description,
                version=skill.version,
                metadata=skill.metadata,
            )
            for skill in self._skills.values()
        )

    def find(
        self,
        query: str,
    ) -> tuple[Skill, ...]:
        query_lower = query.lower()
        return tuple(
            skill
            for skill in self._skills.values()
            if query_lower in skill.name.lower()
            or query_lower in skill.description.lower()
        )