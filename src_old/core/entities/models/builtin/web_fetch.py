from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx
from trafilatura import extract, extract_metadata

from core.entities.models.research_contract import (
    ResearchPage,
    ResearchResult,
)
from core.entities.models.tool import ToolContext
from core.entities.models.web_policy import WebPolicy


@dataclass(slots=True, frozen=True)
class WebFetchInput:
    url: str


MAX_RESPONSE_SIZE = 16_000_000
MAX_TEXT_SIZE = 100_000
MAX_REDIRECTS = 5
MAX_LINKS = 100
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

ALLOWED_CONTENT_TYPES = frozenset(
    {
        "text/html",
        "text/plain",
        "application/json",
        "application/xml",
        "text/xml",
        "application/xhtml+xml",
    }
)


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(
            convert_charrefs=True,
        )

        self.links: list[tuple[str, str]] = []

        self._current_href: str | None = None
        self._current_text: list[str] = []
        self._inside_skip_tag = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()

        if tag in {
            "script",
            "style",
            "noscript",
            "template",
        }:
            self._inside_skip_tag += 1
            return

        if self._inside_skip_tag:
            return

        if tag != "a":
            return

        href: str | None = None

        for name, value in attrs:
            if name.lower() == "href" and value:
                href = value
                break

        if href is None:
            return

        self._current_href = href
        self._current_text = []

    def handle_data(
        self,
        data: str,
    ) -> None:
        if self._inside_skip_tag:
            return

        if self._current_href is None:
            return

        text = data.strip()

        if text:
            self._current_text.append(text)

    def handle_endtag(
        self,
        tag: str,
    ) -> None:
        tag = tag.lower()

        if tag in {
            "script",
            "style",
            "noscript",
            "template",
        }:
            if self._inside_skip_tag > 0:
                self._inside_skip_tag -= 1

            return

        if tag != "a":
            return

        if self._current_href is None:
            return

        text = " ".join(
            self._current_text,
        ).strip()

        if text:
            self.links.append(
                (
                    self._current_href,
                    text,
                ),
            )

        self._current_href = None
        self._current_text = []


async def _read_response(
    response: httpx.Response,
    limit: int,
) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    total = 0
    truncated = False

    async for chunk in response.aiter_bytes():
        remaining = limit - total

        if remaining <= 0:
            truncated = True
            break

        if len(chunk) > remaining:
            chunks.append(
                chunk[:remaining],
            )
            total += remaining
            truncated = True
            break

        chunks.append(chunk)
        total += len(chunk)

    return (
        b"".join(chunks),
        truncated,
    )


def _decode_body(
    body: bytes,
    response: httpx.Response,
) -> str:
    encoding = (
        response.encoding
        or "utf-8"
    )

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


def _extract_links(
    html: str,
    base_url: str,
) -> list[tuple[str, str]]:
    parser = _LinkParser()
    parser.feed(html)

    base = urlparse(base_url)

    result: list[tuple[str, str]] = []
    seen: set[str] = set()

    for href, label in parser.links:
        absolute = urljoin(
            base_url,
            href,
        )

        parsed = urlparse(
            absolute,
        )

        if parsed.scheme not in {
            "http",
            "https",
        }:
            continue

        if (
            parsed.hostname is None
            or parsed.hostname.lower()
            != (base.hostname or "").lower()
        ):
            continue

        normalized = absolute.split(
            "#",
            1,
        )[0]

        if normalized == base_url.split(
            "#",
            1,
        )[0]:
            continue

        if normalized in seen:
            continue

        clean_label = " ".join(
            label.split(),
        )

        if not clean_label:
            continue

        seen.add(normalized)

        result.append(
            (
                clean_label,
                normalized,
            ),
        )

        if len(result) >= MAX_LINKS:
            break

    return result


def _extract_html(
    text: str,
    url: str,
) -> tuple[str, str, list[tuple[str, str]]]:
    title = ""

    try:
        metadata = extract_metadata(
            text,
            default_url=url,
        )

        if metadata is not None:
            title = (
                metadata.title
                or ""
            ).strip()

    except Exception:
        title = ""

    clean_text = extract(
        text,
        url=url,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        include_links=True,
        deduplicate=True,
        favor_precision=True,
    )

    if clean_text is None:
        clean_text = ""

    clean_text = clean_text.strip()

    links = _extract_links(
        text,
        url,
    )

    return (
        clean_text,
        title,
        links,
    )


def _extract_non_html(
    text: str,
) -> str:
    return text.strip()


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
        for redirect_number in range(
            MAX_REDIRECTS + 1,
        ):
            await policy.validate_url(
                current_url,
            )

            async with client.stream(
                "GET",
                current_url,
            ) as response:
                if response.is_redirect:
                    location = response.headers.get(
                        "location",
                    )

                    if not location:
                        raise ValueError(
                            "Redirect response has no Location header",
                        )

                    if redirect_number >= MAX_REDIRECTS:
                        raise ValueError(
                            "Too many redirects "
                            f"(maximum {MAX_REDIRECTS})",
                        )

                    current_url = str(
                        response.url.join(
                            location,
                        ),
                    )

                    continue

                content_type = (
                    response.headers.get(
                        "content-type",
                        "",
                    )
                    .split(
                        ";",
                        1,
                    )[0]
                    .strip()
                    .lower()
                )

                if content_type not in ALLOWED_CONTENT_TYPES:
                    raise ValueError(
                        "Unsupported content type: "
                        f"{content_type or 'unknown'}",
                    )

                response.raise_for_status()

                body, response_truncated = (
                    await _read_response(
                        response,
                        MAX_RESPONSE_SIZE,
                    )
                )

                final_url = str(
                    response.url,
                )

                status_code = response.status_code

            text = _decode_body(
                body,
                response,
            )

            title = ""
            raw_links: list[tuple[str, str]] = []

            if content_type in {
                "text/html",
                "application/xhtml+xml",
            }:
                (
                    clean_text,
                    title,
                    raw_links,
                ) = _extract_html(
                    text,
                    final_url,
                )
            else:
                clean_text = _extract_non_html(
                    text,
                )

            content_truncated = False

            if len(clean_text) > MAX_TEXT_SIZE:
                clean_text = (
                    clean_text[:MAX_TEXT_SIZE]
                    + "\n[content truncated]"
                )

                content_truncated = True

            links = tuple(
                url
                for _, url in raw_links
            )

            page = ResearchPage(
                url=final_url,
                depth=0,
                title=title,
                content=clean_text,
                links=links,
                content_bytes=len(body),
            )

            result = ResearchResult(
                root_url=final_url,
                pages=(page,),
                discovered_urls=links,
                failed_urls=(),
                max_depth_reached=0,
                total_bytes=len(body),
                page_limit_reached=False,
                byte_limit_reached=response_truncated,
            )

            return result.to_json()

    raise RuntimeError(
        "Unreachable redirect state",
    )