"""Say what actually went wrong when the network is unreachable.

An agent whose web tools return a bare timeout retries. It rephrases
the query, widens the search, changes tool, and burns minutes waiting
for DNS that is not going to answer. Every one of those attempts costs
a full connect timeout and teaches the caller nothing, because the
error never said that the host was unreachable.

Two things are done here instead:

- **The failure is named.** "DNS could not resolve duckduckgo.com" and
  "the host accepted the connection but did not answer" lead to
  different next steps, and neither of them is "try another phrasing".
- **It is remembered briefly.** A hard connect failure is not
  transient; retrying it four times in a row is four timeouts. A
  short cooldown turns the second and later attempts into an immediate
  answer carrying the same diagnosis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


# How long a hard connectivity failure suppresses further attempts.
#
# Long enough to cover a burst of retries inside one turn, short enough
# that a genuinely restored network is picked up without a restart.
COOLDOWN_SECONDS = 60.0


class NetworkUnavailable(RuntimeError):
    """Outbound network is not usable, and the reason is known."""


def _host_of(url: str | None) -> str:
    if not url:
        return "the host"

    try:
        return httpx.URL(url).host or url
    except Exception:  # pragma: no cover - httpx is permissive here
        return url


def describe(exc: BaseException, url: str | None = None) -> str:
    """Turn an httpx failure into something actionable.

    The exception text is often empty, which is why the class name is
    always included: an empty message here is indistinguishable from no
    message at all to whoever reads it.
    """

    host = _host_of(url)

    if isinstance(exc, httpx.ConnectTimeout):
        return (
            f"could not connect to {host} in time. The host is not "
            "answering on the expected port."
        )

    if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
        return (
            f"connected to {host} but it sent no response in time. "
            "The host accepted the connection and then stalled."
        )

    if isinstance(exc, httpx.ConnectError):
        detail = str(exc).lower()

        if any(
            marker in detail
            for marker in (
                "name or service not known",
                "nodename nor servname",
                "temporary failure in name resolution",
                "no address associated",
                "getaddrinfo",
            )
        ):
            return (
                f"DNS could not resolve {host}. Nothing can reach the "
                "open internet from here, so no web tool will work "
                "until name resolution is fixed."
            )

        return (
            f"could not connect to {host}. The connection was refused "
            "or the route is blocked."
        )

    if isinstance(exc, httpx.ProxyError):
        return f"the proxy for {host} refused the connection."

    if isinstance(exc, httpx.TooManyRedirects):
        return f"{host} redirected too many times."

    detail = str(exc).strip()
    name = type(exc).__name__

    return f"{host}: {name}" + (f": {detail}" if detail else " (no detail)")


_HINT = (
    " If this persists, work from local sources instead: files, "
    "databases and tools that talk to a service on this machine."
)


@dataclass(slots=True)
class _Outage:
    at: float
    message: str


_state: _Outage | None = None


def blocked_message(url: str | None = None) -> str | None:
    """The reason to refuse without waiting, or None to proceed."""

    global _state

    if _state is None:
        return None

    if time.monotonic() - _state.at > COOLDOWN_SECONDS:
        return None

    return f"{_state.message} (repeated calls within {int(COOLDOWN_SECONDS)}s were not attempted)"


def note_failure(exc: BaseException, url: str | None = None) -> str:
    """Record a hard connectivity failure and return its description."""

    global _state

    message = describe(exc, url) + _HINT

    # Only a failure to connect is treated as an outage. A read timeout
    # means the host was reachable, which is a different story and not
    # grounds for refusing the next attempt.
    if isinstance(
        exc,
        (httpx.ConnectError, httpx.ConnectTimeout, httpx.ProxyError),
    ):
        _state = _Outage(at=time.monotonic(), message=message)

    return message


def note_success() -> None:
    """Clear a recorded outage once a request works."""

    global _state

    _state = None


def reset() -> None:
    """Forget any recorded outage. For tests."""

    global _state

    _state = None


__all__ = [
    "COOLDOWN_SECONDS",
    "NetworkUnavailable",
    "blocked_message",
    "describe",
    "note_failure",
    "note_success",
    "reset",
]