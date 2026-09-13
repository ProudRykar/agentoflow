from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

from core.entities.models.tool import ToolContext
from core.entities.models.web_policy import WebPolicy


@dataclass(slots=True, frozen=True)
class WebFetchInput:
    url: str


MAX_RESPONSE_SIZE = 1_000_000
MAX_TEXT_SIZE = 50_000
MAX_REDIRECTS = 5
REQUEST_TIMEOUT = 10.0

USER_AGENT = "agentoflow/0.1"

ACCEPT_HEADER = (
    "text/html,"
    "text/plain,"
    "application/json,"
    "application/xml,"
    "text/xml;q=0.9,"
    "*/*;q=0.1"
)

ALLOWED_CONTENT_TYPES = frozenset({
    "text/html",
    "text/plain",
    "application/json",
    "application/xml",
    "text/xml",
    "application/xhtml+xml",
})


class _HTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)

        self.title = ""
        self.text_parts: list[str] = []
        self.links: list[str] = []

        self._inside_title = False
        self._skip_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()

        if tag == "title":
            self._inside_title = True

        if tag in {"script", "style", "noscript", "template"}:
            self._skip_depth += 1
            return

        if tag == "a":
            for name, value in attrs:
                if name.lower() == "href" and value:
                    self.links.append(value)
                    break

        if not self._skip_depth and tag in {
            "p",
            "div",
            "section",
            "article",
            "main",
            "header",
            "footer",
            "li",
            "ul",
            "ol",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "br",
        }:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag == "title":
            self._inside_title = False

        if tag in {"script", "style", "noscript", "template"}:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return

        if not self._skip_depth and tag in {
            "p",
            "div",
            "section",
            "article",
            "main",
            "li",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return

        text = data.strip()

        if not text:
            return

        if self._inside_title:
            self.title += text

        self.text_parts.append(text)

    def text(self) -> str:
        lines = [
            " ".join(line.split())
            for line in "".join(self.text_parts).splitlines()
        ]

        return "\n".join(
            line
            for line in lines
            if line
        )


async def _read_response(
    response: httpx.Response,
    limit: int,
) -> bytes:
    chunks: list[bytes] = []
    total = 0

    async for chunk in response.aiter_bytes():
        total += len(chunk)

        if total > limit:
            raise ValueError(
                f"Response exceeds maximum size of {limit} bytes"
            )

        chunks.append(chunk)

    return b"".join(chunks)


def _normalize_links(
    links: list[str],
    base_url: str,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for link in links:
        absolute = urljoin(base_url, link)
        parsed = urlparse(absolute)

        if parsed.scheme not in {"http", "https"}:
            continue

        normalized = absolute.split("#", 1)[0]

        if normalized in seen:
            continue

        seen.add(normalized)
        result.append(normalized)

        if len(result) >= 50:
            break

    return result


def _decode_body(
    body: bytes,
    response: httpx.Response,
) -> str:
    encoding = response.encoding or "utf-8"

    try:
        return body.decode(
            encoding,
            errors="replace",
        )
    except LookupError:
        return body.decode(
            "utf-8",
            errors="replace",
        )


async def web_fetch(
    arguments: WebFetchInput,
    context: ToolContext,
) -> str:
    del context

    policy = WebPolicy()
    current_url = arguments.url

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT,
        follow_redirects=False,
        trust_env=False,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": ACCEPT_HEADER,
        },
    ) as client:
        for redirect_number in range(MAX_REDIRECTS + 1):
            await policy.validate_url(current_url)

            async with client.stream(
                "GET",
                current_url,
            ) as response:
                if response.is_redirect:
                    location = response.headers.get("location")

                    if not location:
                        raise ValueError(
                            "Redirect response has no Location header"
                        )

                    if redirect_number >= MAX_REDIRECTS:
                        raise ValueError(
                            f"Too many redirects "
                            f"(maximum {MAX_REDIRECTS})"
                        )

                    current_url = urljoin(
                        str(response.url),
                        location,
                    )
                    continue

                content_type = (
                    response.headers.get("content-type", "")
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                )

                if content_type not in ALLOWED_CONTENT_TYPES:
                    raise ValueError(
                        "Unsupported content type: "
                        f"{content_type or 'unknown'}"
                    )

                response.raise_for_status()

                body = await _read_response(
                    response,
                    MAX_RESPONSE_SIZE,
                )

            text = _decode_body(
                body,
                response,
            )

            title = ""
            links: list[str] = []

            if content_type in {
                "text/html",
                "application/xhtml+xml",
            }:
                parser = _HTMLParser()
                parser.feed(text)

                clean_text = parser.text()
                title = parser.title.strip()
                links = _normalize_links(
                    parser.links,
                    str(response.url),
                )
            else:
                clean_text = text.strip()

            if len(clean_text) > MAX_TEXT_SIZE:
                clean_text = (
                    clean_text[:MAX_TEXT_SIZE]
                    + "\n[content truncated]"
                )

            output_parts = [
                f"url: {response.url}",
                f"status: {response.status_code}",
                f"content_type: {content_type}",
            ]

            if title:
                output_parts.append(
                    f"title: {title}"
                )

            if clean_text:
                output_parts.append(
                    f"[content]\n{clean_text}"
                )

            if links:
                output_parts.append(
                    "[links]\n"
                    + "\n".join(links)
                )

            return "\n".join(output_parts)

    raise RuntimeError("Unreachable redirect state")