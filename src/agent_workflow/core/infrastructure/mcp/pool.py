from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agent_workflow.core.entities.models.tool import Tool
from agent_workflow.core.infrastructure.config import MCPServerConfigEntry


logger = logging.getLogger(__name__)


def _digest(values: dict[str, str]) -> str:
    """A stable digest of a mapping.

    Only the digest is kept, so a pooled key never holds an API token
    in memory longer than the entry it came from, and an equal
    mapping always yields an equal key regardless of order.
    """

    if not values:
        return ""

    encoded = json.dumps(
        sorted(values.items()),
        separators=(",", ":"),
    )

    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ServerKey:
    """Identity of a shareable server process.

    Keyed by what actually determines the process, not by the name: a
    session that renames a server it changed should not collide with
    another session's copy, and two names for one command should.
    """

    transport: str

    target: str

    args: tuple[str, ...] = ()

    env_digest: str = ""

    headers_digest: str = ""

    startup_timeout: float = 20.0

    request_timeout: float = 60.0

    @classmethod
    def of(cls, entry: MCPServerConfigEntry) -> ServerKey:
        transport = getattr(entry, "transport", "stdio")

        return cls(
            transport=transport,
            target=(
                entry.url if transport == "http" else entry.command
            ),
            args=tuple(entry.args),
            env_digest=_digest(dict(entry.env)),
            headers_digest=_digest(
                dict(getattr(entry, "headers", {}) or {})
            ),
            startup_timeout=entry.startup_timeout,
            request_timeout=entry.request_timeout,
        )


@dataclass(slots=True)
class SharedProcess:
    """One live process and whoever is borrowing it."""

    client: Any = None

    server_info: Any = None

    descriptors: tuple[Any, ...] = ()

    tools: tuple[Tool, ...] = ()

    holders: set[str] = field(default_factory=set)

    # Set once the owner has finished connecting. Waiters await this
    # rather than polling, so a second session does not spin while the
    # first is still handshaking.
    ready: asyncio.Event = field(default_factory=asyncio.Event)

    # Set when the owner failed, so waiters raise instead of hanging.
    failure: str = ""


class MCPProcessPool:
    """One server process, many sessions.

    Each session used to spawn its own MCP client, so a stdio server
    that is really a container cost one container per session. Two
    sessions and one server is merely wasteful; a dozen sessions and
    four servers is what brings a machine down.

    A process is shared by every session asking for the same server
    definition and closed once the last of them releases it. What
    stays per session is the tool registration: each session keeps its
    own registry, so one session hiding a tool cannot affect another.
    """

    def __init__(self) -> None:
        self._entries: dict[ServerKey, SharedProcess] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def process_count(self) -> int:
        """Live processes, as opposed to one per session per server."""

        return len(self._entries)

    def borrower_counts(self) -> dict[str, int]:
        """How many sessions share each process."""

        return {
            key.target: len(shared.holders)
            for key, shared in self._entries.items()
        }

    # ------------------------------------------------------------------
    # Borrowing
    # ------------------------------------------------------------------

    async def acquire(
        self,
        entry: MCPServerConfigEntry,
        session_id: str,
        factory: Any,
    ) -> tuple[Any, tuple[Any, ...], Any]:
        """Return a connected client plus what the handshake found.

        ``factory`` builds the client, so the manager keeps ownership
        of client construction.
        """

        key = ServerKey.of(entry)

        while True:
            async with self._lock:
                shared = self._entries.get(key)

                if shared is None:
                    shared = SharedProcess()
                    self._entries[key] = shared
                    owner = True
                else:
                    owner = False

            if owner:
                try:
                    client = factory()

                    server_info = await client.connect()
                    descriptors = await client.list_tools()
                except Exception as exc:
                    detail = str(exc)

                    tail = ""

                    try:
                        tail = client.stderr_tail()
                    except Exception:
                        pass

                    if tail:
                        detail = f"{detail} | server stderr: {tail}"

                    async with self._lock:
                        if self._entries.get(key) is shared:
                            del self._entries[key]

                        shared.failure = detail

                    shared.ready.set()

                    try:
                        await client.close()
                    except Exception:
                        pass

                    raise

                shared.client = client
                shared.server_info = server_info
                shared.descriptors = tuple(descriptors)
                shared.ready.set()

            else:
                await shared.ready.wait()

                if shared.failure:
                    raise MCPSharedProcessError(shared.failure)

                if shared.client is None:
                    # Released between the wait and the check.
                    await asyncio.sleep(0)
                    continue

            async with self._lock:
                current = self._entries.get(key)

                if current is not shared or shared.client is None:
                    await asyncio.sleep(0)
                    continue

                shared.holders.add(session_id)

                return (
                    shared.client,
                    shared.descriptors,
                    shared.server_info,
                )

    def publish_tools(
        self,
        entry: MCPServerConfigEntry,
        tools: tuple[Tool, ...],
    ) -> None:
        """Record the tools a freshly connected process exposes."""

        shared = self._entries.get(ServerKey.of(entry))

        if shared is not None:
            shared.tools = tools

    async def release(
        self,
        entry: MCPServerConfigEntry,
        session_id: str,
    ) -> None:
        """Drop this session's claim, closing an unclaimed process."""

        key = ServerKey.of(entry)

        async with self._lock:
            shared = self._entries.get(key)

            if shared is None:
                return

            shared.holders.discard(session_id)

            if shared.holders or shared.client is None:
                return

            client = shared.client
            del self._entries[key]

        try:
            await client.close()
        except Exception:
            logger.warning(
                "Closing the shared MCP process for %r failed",
                key.target,
                exc_info=True,
            )

    async def close_all(self) -> None:
        """Close everything, whatever still holds it."""

        async with self._lock:
            shared = list(self._entries.values())
            self._entries.clear()

        for entry in shared:
            if entry.client is None:
                continue

            try:
                await entry.client.close()
            except Exception:
                logger.warning(
                    "Closing a shared MCP process failed",
                    exc_info=True,
                )


class MCPSharedProcessError(RuntimeError):
    """The process this session waited for could not be started."""