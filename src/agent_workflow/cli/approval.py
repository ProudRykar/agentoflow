from __future__ import annotations

import asyncio
from typing import Callable

from agent_workflow.core.entities.models.approval import ApprovalRequest


StateChangeCallback = Callable[[], None]


class ApprovalController:
    def __init__(
        self,
        on_change: StateChangeCallback | None = None,
    ) -> None:
        self._on_change = on_change

        self._future: asyncio.Future[bool] | None = None
        self._request: ApprovalRequest | None = None

    @property
    def active(self) -> bool:
        return self._future is not None

    @property
    def request(self) -> ApprovalRequest | None:
        return self._request

    def set_on_change(
        self,
        callback: StateChangeCallback,
    ) -> None:
        self._on_change = callback

    async def __call__(
        self,
        request: ApprovalRequest,
    ) -> bool:
        if self._future is not None:
            raise RuntimeError(
                "Another approval request is already active",
            )

        loop = asyncio.get_running_loop()

        self._future = loop.create_future()
        self._request = request

        self._notify()

        try:
            return await self._future
        finally:
            self._request = None
            self._future = None
            self._notify()

    def allow(self) -> None:
        self._resolve(True)

    def deny(self) -> None:
        self._resolve(False)

    def _resolve(
        self,
        value: bool,
    ) -> None:
        future = self._future

        if future is None or future.done():
            return

        future.set_result(value)

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change()