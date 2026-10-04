from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agent_workflow.core.infrastructure.paths import (
    AgentWorkflowPaths,
)


SKILL_FILENAME = "SKILL.md"

SKILL_PATTERN = "*/" + SKILL_FILENAME

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 500

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class SkillError(Exception):
    """A skill could not be read or written."""


class SkillValidationError(SkillError):
    """The submitted skill is not valid."""


@dataclass(slots=True, frozen=True)
class SkillSummary:
    name: str
    description: str
    version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)
    path: str = ""
    builtin: bool = False


@dataclass(slots=True, frozen=True)
class SkillDocument:
    name: str
    description: str
    version: str
    instructions: str
    metadata: dict[str, Any] = field(default_factory=dict)
    path: str = ""


def validate_name(name: str) -> str:
    """Normalise and check a skill directory name.

    The name becomes a directory, so anything that could escape the
    skills root is rejected outright.
    """

    cleaned = str(name or "").strip().lower()

    if not cleaned:
        raise SkillValidationError("Skill name is required")

    if len(cleaned) > MAX_NAME_LENGTH:
        raise SkillValidationError(
            f"Skill name must be at most {MAX_NAME_LENGTH} characters"
        )

    if not _NAME_RE.match(cleaned):
        raise SkillValidationError(
            "Skill name may contain lowercase letters, digits, "
            "'-' and '_', and must start with a letter or digit"
        )

    return cleaned


class SkillsService:
    """CRUD over the on-disk skills library.

    Skills are plain Markdown files with YAML frontmatter, the same
    format the loader already reads, so anything written here is
    immediately loadable by the agent with no import step.
    """

    def __init__(
        self,
        skills_directory: Path | None = None,
        paths: AgentWorkflowPaths | None = None,
    ) -> None:
        if skills_directory is not None:
            self._directory = skills_directory
        else:
            resolved = paths or AgentWorkflowPaths()

            self._directory = resolved.home / "skills"

    @property
    def directory(self) -> Path:
        return self._directory

    # ==================================================================
    # Listing
    # ==================================================================

    def list(self) -> list[SkillSummary]:
        summaries: dict[str, SkillSummary] = {}

        for path in self._skill_files():
            try:
                document = self._read_file(path)
            except SkillError:
                continue

            summaries[document.name] = SkillSummary(
                name=document.name,
                description=document.description,
                version=document.version,
                metadata=document.metadata,
                path=str(path),
            )

        return sorted(
            summaries.values(),
            key=lambda item: item.name,
        )

    def get(
        self,
        name: str,
    ) -> SkillDocument:
        return self._read_file(
            self._skill_path(validate_name(name))
        )

    def exists(
        self,
        name: str,
    ) -> bool:
        try:
            return self._skill_path(
                validate_name(name)
            ).is_file()
        except SkillValidationError:
            return False

    # ==================================================================
    # Writing
    # ==================================================================

    def create(
        self,
        name: str,
        *,
        description: str,
        instructions: str,
        version: str = "1.0.0",
        metadata: dict[str, Any] | None = None,
    ) -> SkillDocument:
        cleaned = validate_name(name)

        if self.exists(cleaned):
            raise SkillValidationError(
                f"Skill '{cleaned}' already exists"
            )

        return self._write(
            cleaned,
            description=description,
            instructions=instructions,
            version=version,
            metadata=metadata,
        )

    def update(
        self,
        name: str,
        *,
        description: str,
        instructions: str,
        version: str = "1.0.0",
        metadata: dict[str, Any] | None = None,
    ) -> SkillDocument:
        cleaned = validate_name(name)

        if not self.exists(cleaned):
            raise SkillValidationError(
                f"Skill '{cleaned}' does not exist"
            )

        return self._write(
            cleaned,
            description=description,
            instructions=instructions,
            version=version,
            metadata=metadata,
        )

    def upsert(
        self,
        name: str,
        *,
        description: str,
        instructions: str,
        version: str = "1.0.0",
        metadata: dict[str, Any] | None = None,
    ) -> SkillDocument:
        cleaned = validate_name(name)

        if self.exists(cleaned):
            return self.update(
                cleaned,
                description=description,
                instructions=instructions,
                version=version,
                metadata=metadata,
            )

        return self.create(
            cleaned,
            description=description,
            instructions=instructions,
            version=version,
            metadata=metadata,
        )

    def delete(
        self,
        name: str,
    ) -> bool:
        cleaned = validate_name(name)

        directory = self._skill_dir(cleaned)

        if not (directory / SKILL_FILENAME).is_file():
            return False

        try:
            (directory / SKILL_FILENAME).unlink()

            # Remove the directory only when the skill owned it
            # entirely, so a skill sharing a folder is not broken.
            if not any(directory.iterdir()):
                directory.rmdir()

        except OSError as exc:
            raise SkillError(
                f"Cannot delete skill '{cleaned}': {exc}"
            ) from exc

        return True

    # ==================================================================
    # Internals
    # ==================================================================

    def _skill_dir(
        self,
        name: str,
    ) -> Path:
        directory = (self._directory / name).resolve()

        root = self._directory.resolve()

        # Belt and braces: validate_name already forbids separators.
        if root not in directory.parents:
            raise SkillValidationError(
                "Refusing to write outside the skills directory"
            )

        return directory

    def _skill_path(
        self,
        name: str,
    ) -> Path:
        return self._skill_dir(name) / SKILL_FILENAME

    def _skill_files(self) -> list[Path]:
        if not self._directory.is_dir():
            return []

        try:
            return sorted(
                self._directory.glob(SKILL_PATTERN)
            )
        except OSError:
            return []

    def _write(
        self,
        name: str,
        *,
        description: str,
        instructions: str,
        version: str,
        metadata: dict[str, Any] | None,
    ) -> SkillDocument:
        summary = str(description or "").strip()

        if not summary:
            raise SkillValidationError(
                "Skill description is required"
            )

        if len(summary) > MAX_DESCRIPTION_LENGTH:
            raise SkillValidationError(
                "Skill description is too long"
            )

        body = str(instructions or "").strip()

        if not body:
            raise SkillValidationError(
                "Skill instructions cannot be empty"
            )

        frontmatter: dict[str, Any] = {
            "name": name,
            "description": summary,
            "version": str(version or "1.0.0"),
        }

        for key, value in (metadata or {}).items():
            if key in {"name", "description", "version"}:
                continue

            frontmatter[key] = value

        try:
            rendered = yaml.safe_dump(
                frontmatter,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )
        except yaml.YAMLError as exc:
            raise SkillValidationError(
                f"Invalid skill metadata: {exc}"
            ) from exc

        text = f"---\n{rendered}---\n\n{body}\n"

        target = self._skill_path(name)

        self._write_atomic(target, text)

        return SkillDocument(
            name=name,
            description=summary,
            version=str(version or "1.0.0"),
            instructions=body,
            metadata={
                key: value
                for key, value in (metadata or {}).items()
                if key not in {"name", "description", "version"}
            },
            path=str(target),
        )

    def _read_file(
        self,
        path: Path,
    ) -> SkillDocument:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise SkillError(
                f"Skill file not found: {path.name}"
            ) from exc
        except OSError as exc:
            raise SkillError(
                f"Cannot read skill: {exc}"
            ) from exc

        from agent_workflow.core.entities.models.skills.loader import (
            parse_skill_file,
        )

        try:
            skill = parse_skill_file(path)
        except Exception as exc:
            raise SkillError(
                f"Invalid skill file {path.name}: {exc}"
            ) from exc

        return SkillDocument(
            name=skill.name,
            description=skill.description,
            version=skill.version,
            instructions=skill.instructions,
            metadata=dict(skill.metadata),
            path=str(path),
        )

    def _write_atomic(
        self,
        path: Path,
        text: str,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        handle, temporary = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{SKILL_FILENAME}.",
            suffix=".tmp",
        )

        try:
            with os.fdopen(
                handle,
                "w",
                encoding="utf-8",
            ) as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())

            os.replace(temporary, path)

        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
