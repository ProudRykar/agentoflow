from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any


@dataclass(slots=True, frozen=True)
class ApprovalRequest:
    tool_name: str
    arguments: dict[str, Any]
    permission: str
    reason: str

    # Assigned by ApprovalController when the request becomes
    # active. Callers (TUI, Web UI) echo it back so a decision
    # cannot be applied to the wrong request.
    approval_id: str = ""


@dataclass(slots=True, frozen=True)
class ApprovalRequested:
    """Emitted when the controller starts waiting for a decision."""

    approval_id: str
    tool_name: str
    arguments: dict[str, Any]
    permission: str
    reason: str


@dataclass(slots=True, frozen=True)
class ApprovalResolved:
    """Emitted when the pending request is decided or cancelled."""

    approval_id: str
    approved: bool


ApprovalHandler = Callable[
    [ApprovalRequest],
    Awaitable[bool],
]


class ApprovalDeniedError(Exception):
    """Raised when the user denies an approval request."""


StateChangeCallback = Callable[[], None]

ApprovalEventCallback = Callable[
    [ApprovalRequested | ApprovalResolved],
    Awaitable[None],
]


class ApprovalController:
    """Blocks the agent until a decision arrives.

    The controller is UI-agnostic: it knows nothing about
    Textual, HTTP or WebSocket. A surface (TUI or web) either
    polls ``active``/``request`` or subscribes to ``on_event``.
    """

    def __init__(
        self,
        on_change: StateChangeCallback | None = None,
    ) -> None:
        self._on_change = on_change
        self._on_event: ApprovalEventCallback | None = None

        self._future: asyncio.Future[bool] | None = None
        self._request: ApprovalRequest | None = None

    @property
    def active(self) -> bool:
        return self._future is not None

    @property
    def request(self) -> ApprovalRequest | None:
        return self._request

    @property
    def approval_id(self) -> str:
        if self._request is None:
            return ""

        return self._request.approval_id

    def set_on_change(
        self,
        callback: StateChangeCallback | None,
    ) -> None:
        self._on_change = callback

    def set_on_event(
        self,
        callback: ApprovalEventCallback | None,
    ) -> None:
        self._on_event = callback

    async def __call__(
        self,
        request: ApprovalRequest,
    ) -> bool:
        if self._future is not None:
            raise RuntimeError(
                "Another approval request is already active",
            )

        loop = asyncio.get_running_loop()

        request = replace(
            request,
            approval_id=str(uuid.uuid4()),
        )

        self._future = loop.create_future()
        self._request = request

        self._notify()
        await self._emit(
            ApprovalRequested(
                approval_id=request.approval_id,
                tool_name=request.tool_name,
                arguments=dict(request.arguments),
                permission=request.permission,
                reason=request.reason,
            ),
        )

        try:
            return await self._future
        finally:
            self._request = None
            self._future = None
            self._notify()

    def allow(
        self,
        approval_id: str = "",
    ) -> bool:
        return self._resolve(True, approval_id)

    def deny(
        self,
        approval_id: str = "",
    ) -> bool:
        return self._resolve(False, approval_id)

    async def cancel(self) -> bool:
        """Deny the pending request so the agent stops waiting.

        Used when the session is closed or a run is cancelled
        while it waits for a decision, so the agent never stays
        parked on a future nobody will resolve.
        """

        if not self.active:
            return False

        self.deny(self.approval_id)

        return True

    def _resolve(
        self,
        value: bool,
        approval_id: str,
    ) -> bool:
        future = self._future
        request = self._request

        if future is None or future.done():
            return False

        # A stale decision (from a previous request, or a client
        # that reconnected) must not resolve a newer request.
        if approval_id and request is not None:
            if approval_id != request.approval_id:
                return False

        future.set_result(value)

        loop = asyncio.get_running_loop()

        loop.create_task(
            self._emit(
                ApprovalResolved(
                    approval_id=(
                        request.approval_id if request else ""
                    ),
                    approved=value,
                ),
            ),
        )

        return True

    async def _emit(
        self,
        event: ApprovalRequested | ApprovalResolved,
    ) -> None:
        callback = self._on_event

        if callback is None:
            return

        try:
            await callback(event)
        except Exception:
            pass

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change()
