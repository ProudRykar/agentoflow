from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from typing import Any


@dataclass(slots=True, frozen=True)
class ApprovalRequest:
    tool_name: str
    arguments: dict[str, Any]
    permission: str
    reason: str


ApprovalHandler = Callable[
    [ApprovalRequest],
    Awaitable[bool],
]


class ApprovalDeniedError(Exception):
    """Raised when the user denies an approval request."""