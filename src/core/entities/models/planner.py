from __future__ import annotations

import re
from urllib.parse import urlsplit

from core.entities.models.research_contract import (
    ResearchContract,
    canonicalize_url,
)
from core.entities.models.task_plan import TaskPlan


class Planner:
    """
    Lightweight deterministic planner.

    The planner does not perform research itself.

    It extracts:
        - objective
        - explicitly mentioned URLs
        - minimum page count
        - minimum crawl depth

    The actual research is performed by the trusted web_crawl tool.
    """

    _URL_PATTERN = re.compile(
        r"https?://[^\s<>\[\]{}\"']+",
        re.IGNORECASE,
    )

    _PAGE_COUNT_PATTERN = re.compile(
        r"(?:"
        r"минимум"
        r"|min"
        r"|not\s+less\s+than"
        r"|at\s+least"
        r")\s*"
        r"(\d+)\s*"
        r"(?:страниц?|pages?)?",
        re.IGNORECASE,
    )

    _DEPTH_PATTERN = re.compile(
        r"(?:глубина|depth)\s*"
        r"(?:минимум|min|not\s+less\s+than|at\s+least)?\s*"
        r"(\d+)",
        re.IGNORECASE,
    )

    _VERB_PATTERNS = (
        r"^(изучи|изучить|обучи|обучить)\s+",
        r"^(сравни|сравнить|сравнение)\s+",
        r"^(найди|найти|поищи|поиск)\s+",
        r"^(напиши|написать|создай|создать)\s+",
        r"^(проанализируй|проанализировать|анализ)\s+",
        r"^(почитай|прочти|прочитать)\s+",
        r"^(исследуй|исследовать|исследование)\s+",
        r"^(расскажи|рассказать)\s+",
        r"^(review|analyze|analyse|compare|research|investigate|read|inspect)\s+",
    )

    _LEADING_GREETINGS = frozenset(
        {
            "привет",
            "hello",
            "здравствуйте",
            "добрый",
            "hi",
        }
    )

    def plan(
        self,
        prompt: str,
    ) -> TaskPlan:
        objective = self._extract_objective(
            prompt,
        )

        research = self._extract_research(
            prompt,
        )

        return TaskPlan(
            objective=objective,
            research=research,
        )

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _extract_objective(
        self,
        prompt: str,
    ) -> str:
        cleaned = self._remove_urls(
            prompt,
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        ).strip()

        cleaned = self._strip_leading_greeting(
            cleaned,
        )

        objective = self._find_verb_object(
            cleaned,
        )

        if objective:
            return objective

        if cleaned:
            return cleaned

        if self._has_urls(
            prompt,
        ):
            return (
                "Исследовать указанные "
                "в источниках материалы."
            )

        return "Выполнить поставленную задачу."

    def _strip_leading_greeting(
        self,
        text: str,
    ) -> str:
        words = text.split()

        if not words:
            return ""

        first = words[0].lower().strip(
            ",.!?:;—-",
        )

        if first in self._LEADING_GREETINGS:
            return " ".join(
                words[1:],
            ).strip()

        return text

    def _find_verb_object(
        self,
        text: str,
    ) -> str | None:
        for pattern in self._VERB_PATTERNS:
            cleaned = re.sub(
                pattern,
                "",
                text,
                count=1,
                flags=re.IGNORECASE,
            )

            if cleaned != text:
                cleaned = re.sub(
                    r"\s+",
                    " ",
                    cleaned,
                ).strip()

                if cleaned:
                    return cleaned

        return None

    # ------------------------------------------------------------------
    # Research
    # ------------------------------------------------------------------

    def _extract_research(
        self,
        prompt: str,
    ) -> ResearchContract | None:
        urls = self._extract_urls(
            prompt,
        )

        if not urls:
            return None

        explicit_page_count = (
            self._extract_min_pages(
                prompt,
            )
        )

        min_pages = max(
            explicit_page_count,
            len(urls),
        )

        min_depth = self._extract_min_depth(
            prompt,
        )

        return ResearchContract(
            root_urls=tuple(
                urls,
            ),
            required_urls=(),
            min_pages=min_pages,
            min_depth=min_depth,
            require_all_roots=True,
        )

    def _extract_urls(
        self,
        prompt: str,
    ) -> tuple[str, ...]:
        raw_urls = self._URL_PATTERN.findall(
            prompt,
        )

        normalized: list[str] = []
        seen: set[str] = set()

        for raw_url in raw_urls:
            raw_url = raw_url.rstrip(
                ".,;:!?)]}",
            )

            canonical = canonicalize_url(
                raw_url,
            )

            if not canonical:
                continue

            parsed = urlsplit(
                canonical,
            )

            if parsed.scheme not in {
                "http",
                "https",
            }:
                continue

            if not parsed.netloc:
                continue

            if canonical in seen:
                continue

            seen.add(
                canonical,
            )

            normalized.append(
                canonical,
            )

        return tuple(
            normalized,
        )

    def _extract_min_pages(
        self,
        prompt: str,
    ) -> int:
        match = self._PAGE_COUNT_PATTERN.search(
            prompt,
        )

        if match is None:
            return 1

        try:
            value = int(
                match.group(1),
            )
        except (
            TypeError,
            ValueError,
        ):
            return 1

        return max(
            1,
            value,
        )

    def _extract_min_depth(
        self,
        prompt: str,
    ) -> int:
        match = self._DEPTH_PATTERN.search(
            prompt,
        )

        if match is None:
            return 0

        try:
            value = int(
                match.group(1),
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0

        return max(
            0,
            value,
        )

    def _has_urls(
        self,
        prompt: str,
    ) -> bool:
        return bool(
            self._URL_PATTERN.search(
                prompt,
            )
        )

    def _remove_urls(
        self,
        prompt: str,
    ) -> str:
        return self._URL_PATTERN.sub(
            " ",
            prompt,
        )