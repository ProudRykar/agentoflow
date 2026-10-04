from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_workflow.core.infrastructure.paths import (
    AgentWorkflowPaths,
)


@dataclass(slots=True, frozen=True)
class SessionRecord:
    """Durable identity of a session.

    Only what is needed to rebuild the session is stored. The live
    The live Agent, its tool registry and plugin state are **not**
    persisted: a restart therefore restores a session's identity,
    sandbox root and metadata, while the transcript itself is kept
    separately in the event log so the client can replay it.
    """

    session_id: str
    working_directory: str
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)
    title: str = ""
    # Added after the first release: records written before it
    # simply load as "auto", which is the safe default.
    title_source: str = "auto"


class SessionStore:
    """Small JSON registry of known sessions.

    Writes are atomic so a crash mid-write cannot leave a truncated
    file that breaks startup. A corrupt file is treated as empty
    rather than fatal: losing the registry must not stop the server
    from booting.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    @classmethod
    def default(cls) -> "SessionStore":
        return cls(AgentWorkflowPaths().sessions)

    def load(self) -> list[SessionRecord]:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return []

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []

        if not isinstance(payload, list):
            return []

        records: list[SessionRecord] = []

        for item in payload:
            if not isinstance(item, dict):
                continue

            session_id = item.get("session_id")

            if not isinstance(session_id, str) or not session_id:
                continue

            records.append(
                SessionRecord(
                    session_id=session_id,
                    working_directory=str(
                        item.get("working_directory", "")
                    ),
                    created_at=float(
                        item.get("created_at", 0.0) or 0.0
                    ),
                    metadata=(
                        item.get("metadata")
                        if isinstance(item.get("metadata"), dict)
                        else {}
                    ),
                    title=str(item.get("title", "") or ""),
                    title_source=(
                        str(item.get("title_source", "") or "")
                        or "auto"
                    ),
                )
            )

        return records

    def save(self, records: list[SessionRecord]) -> None:
        self._path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = json.dumps(
            [asdict(record) for record in records],
            ensure_ascii=False,
            indent=2,
        )

        handle, temporary = tempfile.mkstemp(
            dir=str(self._path.parent),
            prefix=".sessions-",
            suffix=".tmp",
        )

        try:
            with os.fdopen(
                handle,
                "w",
                encoding="utf-8",
            ) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())

            os.replace(temporary, self._path)

        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def upsert(self, record: SessionRecord) -> None:
        records = [
            existing
            for existing in self.load()
            if existing.session_id != record.session_id
        ]

        records.append(record)

        self.save(records)

    def remove(self, session_id: str) -> None:
        records = self.load()

        remaining = [
            record
            for record in records
            if record.session_id != session_id
        ]

        if len(remaining) != len(records):
            self.save(remaining)
