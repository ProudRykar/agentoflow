"""Several calls must be able to be in flight at once.

The stdio transport used to hold a lock across the whole round trip and
each waiter read the pipe itself, discarding any line whose id did not
match its own. That combination made concurrency impossible: a reply to
a queued request was thrown away, and the request it belonged to sat
until its timeout.

These tests pin the two properties that fix requires, because both fail
silently rather than loudly. Out-of-order replies get attributed to the
wrong caller, and a transport that is still serial passes a timing test
only when the machine happens to be fast.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

from agent_workflow.core.infrastructure.mcp.client import (
    MCPClient,
    MCPServerConfig,
)


STUB = str(Path(__file__).parent / "fixtures" / "mcp_stub_server.py")

PER_CALL = 0.30
CALLS = 6


def config(**kwargs) -> MCPServerConfig:
    return MCPServerConfig(
        name="stub",
        command=sys.executable,
        args=(STUB,),
        **kwargs,
    )


def text_of(response) -> str:
    blocks = response.get("content") or []
    return blocks[0].get("text", "") if blocks else ""


async def test_concurrent_calls_overlap_in_wall_clock() -> None:
    """Six 300ms calls must not take six times 300ms.

    Generous bounds on purpose: the point is to fail when calls are
    serialised, not to measure how fast this machine is.
    """

    calls = [("echo", {"tag_id": f"t-{PER_CALL}"})] * CALLS

    async with MCPClient(config()) as client:
        started = time.monotonic()
        results = await client.call_tools_concurrently(
            calls, limit=CALLS
        )
        elapsed = time.monotonic() - started

    serial = PER_CALL * CALLS

    assert len(results) == CALLS
    # Anything at or past the serial time means they queued.
    assert elapsed < serial * 0.7, (
        f"{CALLS} x {PER_CALL}s took {elapsed:.2f}s, "
        f"about {serial:.2f}s: the calls are not overlapping"
    )


async def test_each_reply_reaches_the_caller_that_asked() -> None:
    """Distinct arguments must come back with their own answers.

    The stub sleeps in proportion to its argument, so the slowest call
    answers first. Routing by arrival order rather than by id therefore
    hands every caller someone else's answer.
    """

    delays = [0.30, 0.24, 0.18, 0.12, 0.06, 0.02]

    calls = [("echo", {"tag_id": f"t-{d}"}) for d in delays]

    async with MCPClient(config()) as client:
        results = await client.call_tools_concurrently(
            calls, limit=len(calls)
        )

    for (name, args), response in zip(calls, results):
        assert isinstance(response, dict), response
        assert text_of(response) == f"echo:{args['tag_id']}"


async def test_a_slow_call_does_not_block_a_fast_one() -> None:
    """Fast calls must finish while the slow one is still running.

    The batch as a whole still waits for the slowest, because
    ``gather`` is not a streaming result. What must not happen is the
    two quick calls sitting idle behind the slow one, so this measures
    when each one actually completed rather than the total.
    """

    calls = [
        ("echo", {"tag_id": "t-0.60"}),
        ("echo", {"tag_id": "t-0.01"}),
        ("echo", {"tag_id": "t-0.01"}),
    ]

    finished: dict[str, float] = {}
    started = time.monotonic()

    async with MCPClient(config()) as client:
        original = client.call_tool

        async def timed(name, arguments, _original=original):
            result = await _original(name, arguments)
            finished[arguments["tag_id"]] = time.monotonic() - started
            return result

        client.call_tool = timed  # type: ignore[method-assign]

        results = await client.call_tools_concurrently(
            calls, limit=3
        )

    assert all(isinstance(r, dict) for r in results)
    # The quick ones are done before the slow one is, with room to
    # spare; serially they would both land after 0.60s.
    assert finished["t-0.01"] < 0.30, (
        f"quick calls finished at {finished['t-0.01']:.2f}s"
    )
    assert finished["t-0.60"] > 0.55


async def test_the_limit_is_respected() -> None:
    """A batch wider than the cap must not open N sockets at once."""

    calls = [("echo", {"tag_id": "t-0.10"})] * 6

    async with MCPClient(config()) as client:
        started = time.monotonic()
        results = await client.call_tools_concurrently(
            calls, limit=2
        )
        elapsed = time.monotonic() - started

    assert len(results) == 6
    # 2 at a time over 6 x 100ms is ~300ms; all 6 at once would be
    # ~100ms.
    assert elapsed > 0.20, f"took {elapsed:.2f}s, limit looks ignored"


async def test_a_failing_call_does_not_discard_the_rest() -> None:
    """One bad item must not throw away the others' results.

    The calls that succeeded already happened on the far side; an
    exception here cannot un-write them.
    """

    calls = [
        ("echo", {"tag_id": "t-0.01"}),
        ("no_such_tool", {"tag_id": "t-0.01"}),
        ("echo", {"tag_id": "t-0.01"}),
    ]

    async with MCPClient(config()) as client:
        results = await client.call_tools_concurrently(
            calls, limit=3
        )

    assert len(results) == 3
    # The server reports an unknown tool as an error payload rather
    # than a transport failure, so the first and third still hold
    # their answers.
    assert text_of(results[0]) == "echo:t-0.01"
    assert text_of(results[2]) == "echo:t-0.01"


async def test_an_empty_batch_is_a_no_op() -> None:
    async with MCPClient(config()) as client:
        assert await client.call_tools_concurrently([]) == []


async def test_a_zero_limit_is_refused() -> None:
    async with MCPClient(config()) as client:
        with pytest.raises(ValueError, match="limit"):
            await client.call_tools_concurrently(
                [("echo", {"tag_id": "t-0.01"})], limit=0
            )


async def test_closing_releases_callers_rather_than_stranding_them() -> None:
    """A shutdown must not leave waiters on their own timeout.

    The request timeout here is 30s. Left unawaited, the caller would
    sit for the full 30 after the connection went away, which reads as
    a hang and hides the fact that the server is gone.
    """

    async with MCPClient(config(request_timeout=30.0)) as client:
        waiting = asyncio.ensure_future(
            client.call_tool("echo", {"tag_id": "t-1.20"})
        )

        await asyncio.sleep(0.15)

        started = time.monotonic()
        await client.close()

        with pytest.raises(Exception, match="closed|exited"):
            await asyncio.wait_for(waiting, timeout=5.0)

        elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"waited {elapsed:.2f}s after close"
