from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse


class WebPolicyError(Exception):
    """Raised when a web request violates the network policy."""


class WebPolicy:
    ALLOWED_SCHEMES = frozenset({"http", "https"})
    ALLOWED_PORTS = frozenset({80, 443})

    async def validate_url(self, url: str) -> None:
        parsed = urlparse(url)

        if parsed.scheme not in self.ALLOWED_SCHEMES:
            raise WebPolicyError(
                "Only http:// and https:// URLs are allowed"
            )

        if not parsed.hostname:
            raise WebPolicyError("URL must contain a hostname")

        if parsed.username is not None or parsed.password is not None:
            raise WebPolicyError(
                "Credentials in URLs are not allowed"
            )

        if parsed.fragment:
            # Fragment is never sent to the server.
            pass

        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80

        if port not in self.ALLOWED_PORTS:
            raise WebPolicyError(
                f"Port {port} is not allowed"
            )

        hostname = parsed.hostname.rstrip(".").lower()

        if hostname in {
            "localhost",
            "localhost.localdomain",
            "ip6-localhost",
            "ip6-loopback",
        }:
            raise WebPolicyError(
                f"Host '{hostname}' is not allowed"
            )

        if hostname.endswith(".localhost"):
            raise WebPolicyError(
                f"Host '{hostname}' is not allowed"
            )

        await self._validate_resolved_addresses(hostname, port)

    async def _validate_resolved_addresses(
        self,
        hostname: str,
        port: int,
    ) -> None:
        try:
            addresses = await asyncio.to_thread(
                socket.getaddrinfo,
                hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise WebPolicyError(
                f"Could not resolve host '{hostname}'"
            ) from exc

        if not addresses:
            raise WebPolicyError(
                f"Host '{hostname}' resolved to no addresses"
            )

        for address in addresses:
            raw_ip = address[4][0]

            try:
                ip = ipaddress.ip_address(raw_ip)
            except ValueError as exc:
                raise WebPolicyError(
                    f"Invalid resolved address for '{hostname}'"
                ) from exc

            if self._is_blocked_address(ip):
                raise WebPolicyError(
                    f"Host '{hostname}' resolves to blocked address "
                    f"{ip}"
                )

    @staticmethod
    def _is_blocked_address(
        ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    ) -> bool:
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        )