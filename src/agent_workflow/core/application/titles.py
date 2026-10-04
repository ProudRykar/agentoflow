from __future__ import annotations

import asyncio
import re
from typing import Any

from agent_workflow.core.entities.models.llm_client import LLMClient


MAX_TITLE_LENGTH = 60

SYSTEM_PROMPT = (
    "You name chat conversations. Given the user's first message, "
    "reply with a title of at most five words that describes the "
    "task. Reply with the title only: no quotes, no punctuation at "
    "the end, no explanation."
)

# Field the title travels in, so a streamed reply cannot be mistaken
# for one.
_TITLE_RE = re.compile(r"[Tt][Ii][Tt][Ll][Ee]\s*:\s*(.+)")


def derive_title(prompt: str) -> str:
    """Fallback title taken from the prompt itself.

    Used when the model is unavailable so a session is never left
    untitled.
    """

    text = " ".join(str(prompt or "").split())

    if not text:
        return "New chat"

    # Prefer a leading imperative or noun phrase.
    sentence = re.split(r"[.!?\n]", text, maxsplit=1)[0]

    words = sentence.split()

    if len(words) > 8:
        sentence = " ".join(words[:8])

    sentence = sentence.strip(" ,;:-")

    if not sentence:
        return "New chat"

    if len(sentence) > MAX_TITLE_LENGTH:
        sentence = sentence[:MAX_TITLE_LENGTH].rstrip()

    if not sentence:
        sentence = "New chat"

    return sentence[0].upper() + sentence[1:]


def _clean(raw: str) -> str:
    text = " ".join(str(raw or "").split())

    match = _TITLE_RE.match(text)

    if match:
        text = match.group(1)

    # Strip wrapping quotes and trailing punctuation the model may
    # add despite instructions. Both are trimmed repeatedly because
    # a reply like '"Title".' needs the quote removed again after
    # the full stop.
    for _ in range(3):
        text = text.strip().strip("\"'`*").strip()
        text = text.rstrip(".,;:!").strip()

    if len(text) > MAX_TITLE_LENGTH:
        text = text[:MAX_TITLE_LENGTH].rstrip()

    if not text:
        return "New chat"

    return text[0].upper() + text[1:]


async def generate_title(
    llm: LLMClient,
    prompt: str,
    *,
    timeout: float = 20.0,
) -> str:
    """Ask the model for a short title.

    Never raises: a failure falls back to :func:`derive_title` so
    titling can never block or break a run.
    """

    text = str(prompt or "").strip()

    if not text:
        return "New chat"

    # Bound the prompt sent for titling.
    excerpt = text[:1500]

    try:
        response = await asyncio.wait_for(
            llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": excerpt},
                ],
                tools=(),
            ),
            timeout=timeout,
        )
    except (asyncio.TimeoutError, Exception):
        return derive_title(text)

    content = getattr(response, "content", None)

    if not isinstance(content, str) or not content.strip():
        return derive_title(text)

    return _clean(content)


def build_title_messages(
    prompt: str,
) -> list[dict[str, Any]]:
    """Exposed for tests and for the system prompt preview."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt[:1500]},
    ]
