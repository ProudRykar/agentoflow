from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from agent_workflow.core.entities.models.skills.registry import (
    InvalidSkillError,
    SkillRegistry,
)
from agent_workflow.core.entities.models.skills.skill import Skill


class SkillLoadError(Exception):
    def __init__(
        self,
        path: Path,
        message: str,
    ) -> None:
        self.path = path
        super().__init__(f"Failed to load skill from {path}: {message}")


FRONTMATTER_PATTERN = re.compile(
    r"^---\n(.*?)\n---\n",
    re.DOTALL,
)


REQUIRED_FIELDS = ("name", "description")


def parse_skill_file(
    path: Path,
) -> Skill:
    content = path.read_text(encoding="utf-8")

    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        raise SkillLoadError(
            path,
            "Missing frontmatter (expected --- ... ---)",
        )

    frontmatter_text = match.group(1)
    instructions = content[match.end() :].strip()

    try:
        frontmatter = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as exc:
        raise SkillLoadError(
            path,
            f"Invalid YAML in frontmatter: {exc}",
        ) from exc

    if not isinstance(frontmatter, dict):
        raise SkillLoadError(
            path,
            "Frontmatter must be a mapping",
        )

    for field_name in REQUIRED_FIELDS:
        if field_name not in frontmatter:
            raise SkillLoadError(
                path,
                f"Missing required field: {field_name}",
            )
        if not frontmatter[field_name]:
            raise SkillLoadError(
                path,
                f"Required field '{field_name}' cannot be empty",
            )

    name = str(frontmatter["name"])
    description = str(frontmatter["description"])
    version = str(frontmatter.get("version", "1.0.0"))

    metadata: dict[str, Any] = {}
    for key, value in frontmatter.items():
        if key not in {"name", "description", "version"}:
            metadata[key] = value

    return Skill(
        name=name,
        description=description,
        version=version,
        instructions=instructions,
        metadata=metadata,
        source=path,
    )


class SkillLoader:
    def __init__(
        self,
        registry: SkillRegistry,
    ) -> None:
        self._registry = registry

    def load(
        self,
        path: Path,
    ) -> Skill:
        skill = parse_skill_file(path)
        self._registry.register(skill)
        return skill

    def load_directory(
        self,
        directory: Path,
        pattern: str = "*/SKILL.md",
    ) -> list[Skill]:
        skills: list[Skill] = []

        for skill_path in directory.glob(pattern):
            try:
                skill = self.load(skill_path)
                skills.append(skill)
            except SkillLoadError:
                raise
            except Exception as exc:
                raise SkillLoadError(
                    skill_path,
                    f"Unexpected error: {exc}",
                ) from exc

        return skills

    def discover(
        self,
        directory: Path,
        pattern: str = "*/SKILL.md",
    ) -> list[SkillMetadata]:
        from agent_workflow.core.entities.models.skills.skill import (
            SkillMetadata,
        )

        discovered: list[SkillMetadata] = []

        for skill_path in directory.glob(pattern):
            try:
                content = skill_path.read_text(encoding="utf-8")
                match = FRONTMATTER_PATTERN.match(content)
                if not match:
                    continue

                frontmatter_text = match.group(1)
                frontmatter = yaml.safe_load(frontmatter_text)

                if not isinstance(frontmatter, dict):
                    continue

                if "name" not in frontmatter or "description" not in frontmatter:
                    continue

                discovered.append(
                    SkillMetadata(
                        name=str(frontmatter["name"]),
                        description=str(frontmatter["description"]),
                        version=str(frontmatter.get("version", "1.0.0")),
                        metadata={
                            k: v
                            for k, v in frontmatter.items()
                            if k not in {"name", "description", "version"}
                        },
                    )
                )
            except Exception:
                continue

        return discovered