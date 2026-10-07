"""Delivering an event to a consumer that may not be async.

The annotation demanded an ``Awaitable``, so the obvious
``on_event=events.append`` was rejected by the type checker and then
failed at runtime with "object NoneType can't be used in 'await'
expression" -- a message about the wrong line, from the wrong layer.

Kept in its own module because the obvious home (agent.py) is
imported *by* the plugin loader, so anything it exported could not be
imported back from there without a cycle.

Both forms are supported. A synchronous consumer blocks the loop for
the length of its own work, which is what handing it a plain function
asked for.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

EventSink = Callable[[Any], Any]


async def emit_to(
    sink: EventSink | None,
    event: Any,
) -> None:
    """Deliver one event, awaiting the result only if there is one."""

    if sink is None:
        return

    result = sink(event)

    if inspect.isawaitable(result):
        await result
