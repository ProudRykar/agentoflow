from __future__ import annotations

import os
import re
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_workflow.core.infrastructure.paths import (
    AgentWorkflowPaths,
)
from agent_workflow.core.infrastructure.toml_writer import (
    TOMLWriteError,
    dumps,
)


class SettingsError(Exception):
    """Settings cannot be read or written."""


class SettingsValidationError(SettingsError):
    """The submitted document is not valid for its target."""


# Only these files may be edited through the web layer. Anything
# else in the config directory stays off limits, so a compromised
# request cannot reach arbitrary paths.
CONFIG_FILE = "config.toml"
MODELS_FILE = "models.toml"

EDITABLE_FILES = (CONFIG_FILE, MODELS_FILE)

REDACTED = "***"

# Keys whose values are replaced before a document leaves the
# process. Values are never returned, so a client cannot read a
# secret back out of the API.
SECRET_KEYS = frozenset({
    "api_key",
    "apikey",
    "authorization",
    "token",
    "password",
    "secret",
    "client_secret",
    "access_key",
    "private_key",
})


@dataclass(slots=True)
class SettingsDocument:
    name: str
    path: Path
    data: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    exists: bool = False


def is_secret_key(key: str) -> bool:
    """True when a key name suggests it holds a credential.

    CamelCase is folded to snake_case first, so ``clientSecret`` and
    ``api-key`` are both recognised.
    """

    raw = str(key).strip()

    folded = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])",
        "_",
        raw,
    )

    normalized = folded.lower().replace("-", "_")

    if normalized in SECRET_KEYS:
        return True

    return any(
        normalized.endswith(f"_{suffix}")
        for suffix in ("token", "secret", "password", "key")
    )


def redact(value: Any) -> Any:
    """Replace secret-looking values, recursing into containers."""

    if isinstance(value, dict):
        return {
            key: (
                REDACTED
                if is_secret_key(key) and item not in (None, "")
                else redact(item)
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [redact(item) for item in value]

    return value


class SettingsService:
    """Read and write the harness TOML files.

    Writes are atomic and validated before they replace anything:
    a document that the loader would reject never reaches disk, and
    a backup of the previous contents is kept.
    """

    def __init__(
        self,
        paths: AgentWorkflowPaths | None = None,
        backups: int = 5,
    ) -> None:
        self._paths = paths or AgentWorkflowPaths()
        self._backups = max(0, backups)

    @property
    def config_path(self) -> Path:
        return self._paths.config

    @property
    def models_path(self) -> Path:
        return self._paths.models

    def path_for(
        self,
        name: str,
    ) -> Path:
        resolved = self._resolve(name)

        return resolved

    def _resolve(
        self,
        name: str,
    ) -> Path:
        if name not in EDITABLE_FILES:
            raise SettingsError(
                f"Only {', '.join(EDITABLE_FILES)} may be edited"
            )

        # Both names are constants, but resolve anyway so the
        # returned path is guaranteed inside the config home.
        candidate = (self._paths.home / name).resolve()

        if candidate.parent != self._paths.home.resolve():
            raise SettingsError("Refusing to write outside config home")

        return candidate

    # ==================================================================
    # Reading
    # ==================================================================

    def read(
        self,
        name: str,
        *,
        reveal_secrets: bool = False,
    ) -> SettingsDocument:
        path = self._resolve(name)

        if not path.exists():
            return SettingsDocument(
                name=name,
                path=path,
                exists=False,
            )

        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SettingsError(
                f"Cannot read {name}: {exc}"
            ) from exc

        try:
            data = tomllib.loads(raw)
        except tomllib.TOMLDecodeError as exc:
            raise SettingsValidationError(
                f"{name} is not valid TOML: {exc}"
            ) from exc

        return SettingsDocument(
            name=name,
            path=path,
            data=data if reveal_secrets else redact(data),
            raw=raw,
            exists=True,
        )

    def raw_text(
        self,
        name: str,
        *,
        reveal_secrets: bool = False,
    ) -> str:
        """The file as text, for a syntax-highlighting editor.

        The raw text is returned verbatim when secrets are allowed,
        and with secret values masked otherwise.
        """

        path = self._resolve(name)

        if not path.exists():
            return ""

        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SettingsError(
                f"Cannot read {name}: {exc}"
            ) from exc

        if reveal_secrets:
            return raw

        return dumps(redact(tomllib.loads(raw)))

    # ==================================================================
    # Writing
    # ==================================================================

    def write(
        self,
        name: str,
        data: dict[str, Any],
        *,
        keep_secrets: dict[str, Any] | None = None,
    ) -> SettingsDocument:
        """Validate then atomically replace a settings file.

        Secret values the caller could not see are restored from
        the stored document, so editing a redacted document in the
        UI never silently deletes a token.
        """

        path = self._resolve(name)

        if not isinstance(data, dict):
            raise SettingsValidationError(
                "Settings payload must be an object"
            )

        merged = _merge_secrets(
            data,
            self._stored(name, keep_secrets),
        )

        try:
            text = dumps(merged)
        except TOMLWriteError as exc:
            raise SettingsValidationError(str(exc)) from exc

        self._validate(name, text)

        self._backup(path)
        self._write_atomic(path, text)

        return SettingsDocument(
            name=name,
            path=path,
            data=redact(merged),
            raw=text,
            exists=True,
        )

    def write_text(
        self,
        name: str,
        text: str,
        *,
        keep_secrets: dict[str, Any] | None = None,
    ) -> SettingsDocument:
        """Write raw TOML text from a text editor."""

        path = self._resolve(name)

        try:
            parsed = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise SettingsValidationError(
                f"Invalid TOML: {exc}"
            ) from exc

        merged = _merge_secrets(
            parsed,
            self._stored(name, keep_secrets),
        )

        try:
            rendered = dumps(merged)
        except TOMLWriteError as exc:
            raise SettingsValidationError(str(exc)) from exc

        self._validate(name, rendered)

        self._backup(path)
        self._write_atomic(path, rendered)

        return SettingsDocument(
            name=name,
            path=path,
            data=redact(merged),
            raw=rendered,
            exists=True,
        )

    # ==================================================================
    # Internals
    # ==================================================================

    def _stored(
        self,
        name: str,
        override: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """The document as stored, used to restore secrets.

        Read straight from disk rather than trusting the client to
        echo secrets back, so a redacted document can be saved
        without a caller having to know a secret was involved.
        """

        if override:
            return override

        path = self._resolve(name)

        if not path.exists():
            return {}

        try:
            return tomllib.loads(
                path.read_text(encoding="utf-8")
            )
        except (OSError, tomllib.TOMLDecodeError):
            return {}

    def _validate(
        self,
        name: str,
        text: str,
    ) -> None:
        """Reject a document the harness could not load."""

        if name == CONFIG_FILE:
            from agent_workflow.core.infrastructure.config import (
                ConfigLoader,
            )

            handle = tempfile.NamedTemporaryFile(
                "wb",
                suffix=".toml",
                delete=False,
            )

            try:
                handle.write(text.encode("utf-8"))
                handle.close()

                ConfigLoader().load(Path(handle.name))
            except Exception as exc:
                raise SettingsValidationError(
                    f"Configuration would not load: {exc}"
                ) from exc
            finally:
                Path(handle.name).unlink(missing_ok=True)

            return

        if name == MODELS_FILE:
            from agent_workflow.core.infrastructure.model_catalog import (
                ModelCatalogLoader,
            )

            handle = tempfile.NamedTemporaryFile(
                "wb",
                suffix=".toml",
                delete=False,
            )

            try:
                handle.write(text.encode("utf-8"))
                handle.close()

                ModelCatalogLoader().load(Path(handle.name))
            except Exception as exc:
                raise SettingsValidationError(
                    f"Model catalog would not load: {exc}"
                ) from exc
            finally:
                Path(handle.name).unlink(missing_ok=True)

    def _backup(self, path: Path) -> None:
        if self._backups <= 0 or not path.exists():
            return

        try:
            existing = sorted(
                path.parent.glob(f"{path.name}.bak.*"),
                key=lambda item: item.stat().st_mtime,
            )
        except OSError:
            return

        for stale in existing[: max(0, len(existing) - self._backups + 1)]:
            try:
                stale.unlink()
            except OSError:
                continue

        target = path.with_name(
            f"{path.name}.bak.{os.getpid()}"
        )

        try:
            target.write_bytes(path.read_bytes())
        except OSError:
            # A missing backup must not block the write.
            return

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
            prefix=f".{path.name}.",
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

    # ==================================================================
    # Model helpers
    # ==================================================================

    def models(self) -> list[dict[str, Any]]:
        document = self.read(MODELS_FILE)

        raw = document.data.get("models", [])

        if not isinstance(raw, list):
            return []

        return [
            dict(item)
            for item in raw
            if isinstance(item, dict)
        ]

    def add_model(
        self,
        model: dict[str, Any],
    ) -> SettingsDocument:
        document = self.read(
            MODELS_FILE,
            reveal_secrets=True,
        )

        data = dict(document.data)
        models = data.get("models", [])

        if not isinstance(models, list):
            models = []

        name = str(model.get("name", "")).strip()

        if not name:
            raise SettingsValidationError("Model name is required")

        for index, existing in enumerate(models):
            if (
                isinstance(existing, dict)
                and existing.get("name") == name
            ):
                models[index] = model
                break
        else:
            models.append(model)

        data["models"] = models

        return self.write(MODELS_FILE, data)

    def remove_model(
        self,
        name: str,
    ) -> SettingsDocument:
        document = self.read(
            MODELS_FILE,
            reveal_secrets=True,
        )

        data = dict(document.data)
        models = data.get("models", [])

        if not isinstance(models, list):
            models = []

        remaining = [
            item
            for item in models
            if not (
                isinstance(item, dict)
                and item.get("name") == name
            )
        ]

        if len(remaining) == len(models):
            raise SettingsValidationError(
                f"Model not found: {name}"
            )

        data["models"] = remaining

        return self.write(MODELS_FILE, data)


def _merge_secrets(
    submitted: dict[str, Any],
    existing: dict[str, Any],
) -> dict[str, Any]:
    """Restore redacted values from the document on disk.

    A redacted value is a placeholder the client could not read, so
    the stored value wins. Real edits to other keys are preserved.
    """

    if not existing:
        return submitted

    merged = dict(submitted)

    for key, value in existing.items():
        current = merged.get(key)

        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = _merge_secrets(current, value)
            continue

        # Arrays of tables, such as MCP servers, hold secrets too;
        # they are merged positionally.
        if (
            isinstance(value, list)
            and isinstance(current, list)
        ):
            merged[key] = _merge_lists(current, value)
            continue

        if key in merged and merged[key] == REDACTED:
            merged[key] = value
            continue

        if key not in merged and is_secret_key(key):
            merged[key] = value

    return merged


def _merge_lists(
    submitted: list[Any],
    existing: list[Any],
) -> list[Any]:
    merged: list[Any] = []

    for index, item in enumerate(submitted):
        if index >= len(existing):
            merged.append(item)
            continue

        reference = existing[index]

        if isinstance(item, dict) and isinstance(reference, dict):
            merged.append(_merge_secrets(item, reference))
        else:
            merged.append(item)

    # Entries the client dropped are not resurrected: that is a
    # real deletion, not an edit.
    return merged
