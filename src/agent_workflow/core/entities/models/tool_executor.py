from __future__ import annotations

import asyncio
from typing import Any

from agent_workflow.core.entities.models.arguments import (
    ArgumentDecoder,
    ArgumentDecoderError,
)
from agent_workflow.core.entities.models.builtin.execute_shell import (
    ShellCommandError,
)
from agent_workflow.core.entities.models.tool import Tool, ToolContext
from agent_workflow.core.entities.models.tool_registry import ToolRegistry
from agent_workflow.core.entities.models.tool_result import ToolError, ToolResult


# Appended to truncated output. Names the tool, says how much was
# dropped, and tells the model what to do about it -- a bare ellipsis
# would let it answer as though it had seen everything.
TRUNCATION_MARKER = (
    "\n\n[TRUNCATED] The output of '{tool}' was {shown} of "
    "{total} characters ({lost} dropped, limit {limit}). What "
    "follows is the beginning, not the whole result -- narrow the "
    "query, filter, or page the call rather than treating this as "
    "complete."
)


def _truncate_marked(
    text: str,
    limit: int,
    tool_name: str,
) -> str:
    """Cut to ``limit`` and say so, inside the limit.

    The marker counts against the budget, and its own length depends
    on the numbers it prints, so the body length and the marker length
    have to be solved together. One pass is not enough: rendering with
    placeholders and then substituting the real figures makes the
    marker longer than the space that was reserved, and the result
    overshoots the limit by a few characters. Two more passes settle
    it -- the numbers only change when the body length changes, which
    it no longer does after the first correction.
    """

    if limit <= 0:
        # Degenerate but constructible: nothing fits, not even the
        # note saying something was cut.
        return ""

    body_room = max(0, limit - len(marker_of(tool_name, 0, len(text), 0, limit)) - 2)

    for _attempt in range(3):
        body = text[:body_room]
        marker = marker_of(
            tool_name,
            len(body),
            len(text),
            len(text) - len(body),
            limit,
        )

        corrected = max(
            0,
            limit - len(marker) - 2,
        )

        if corrected == body_room:
            break

        body_room = corrected

    body = text[:body_room]

    marker = marker_of(
        tool_name,
        len(body),
        len(text),
        len(text) - len(body),
        limit,
    )

    if len(body) + 2 + len(marker) <= limit:
        return f"{body}\n\n{marker}"

    # The limit is too small for the full note. A shorter one still
    # carries the fact that data was dropped, which is the part the
    # model needs in order not to answer as though it had seen all of
    # it; the guidance about narrowing the query is what gets lost.
    short = f"[TRUNCATED: {len(body)} of {len(text)} characters]"

    room = max(0, limit - len(short) - 2)

    return f"{text[:room]}\n\n{short}"


def marker_of(
    tool_name: str,
    shown: int,
    total: int,
    lost: int,
    limit: int,
) -> str:
    return TRUNCATION_MARKER.format(
        tool=tool_name,
        shown=shown,
        total=total,
        lost=lost,
        limit=limit,
    )


class ToolExecutor:
    """Runs a tool under its policy."""

    def __init__(
        self,
        registry: ToolRegistry,
        argument_decoder: ArgumentDecoder | None = None,
        *,
        output_allowance: int | None = None,
    ) -> None:
        self._registry = registry
        # Cap for a single result, per request, derived from the
        # context budget.
        #
        # A per-tool constant cannot express this: the research tools
        # allow 120k characters, which is several times a default 32k
        # token window, so one result could be all the budget before
        # the assembler saw it and had to evict everything else. The
        # per-tool value stays as a ceiling -- a tool may be stricter
        # than the budget allows -- and this is the other bound.
        self._output_allowance = output_allowance
        self._argument_decoder = (
            argument_decoder
            or ArgumentDecoder()
        )

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        try:
            tool = self._registry.get(
                tool_name,
            )
        except Exception as exc:
            return ToolResult(
                error=ToolError(
                    message=str(exc),
                    code="tool_not_found",
                    retryable=False,
                ),
            )

        try:
            self._check_permissions(
                tool,
                context,
            )
        except PermissionError as exc:
            return ToolResult(
                error=ToolError(
                    message=str(exc),
                    code="permission_denied",
                    retryable=False,
                ),
            )

        try:
            decoded_arguments = (
                self._argument_decoder.decode(
                    arguments,
                    tool.input_type,
                )
            )
        except ArgumentDecoderError as exc:
            return ToolResult(
                error=ToolError(
                    message=str(exc),
                    code=exc.code,
                    retryable=False,
                ),
            )

        try:
            output = await asyncio.wait_for(
                tool.handler(
                    decoded_arguments,
                    context,
                ),
                timeout=tool.policy.timeout,
            )

        except asyncio.TimeoutError:
            return ToolResult(
                error=ToolError(
                    message=(
                        f"Tool '{tool_name}' timed out after "
                        f"{tool.policy.timeout} seconds"
                    ),
                    code="timeout",
                    retryable=True,
                ),
            )

        except ShellCommandError as exc:
            return ToolResult(
                error=ToolError(
                    message=str(exc),
                    code="command_failed",
                    retryable=True,
                ),
            )

        except Exception as exc:
            return ToolResult(
                error=ToolError(
                    message=str(exc),
                    code="execution_error",
                    retryable=True,
                ),
            )

        output_text = str(output)

        # Over-long output is truncated, not refused.
        #
        # Refusing lost the whole result: a legitimate 25k-character
        # answer -- a tag list, a research digest, a batch reply --
        # became an error and the model got nothing, then had to
        # re-run the call with narrower filters to get a fragment it
        # could have been handed immediately. The assembler already
        # truncates context blocks this way, with a visible marker;
        # refusing here was the same problem solved by discarding it.
        #
        # The marker is not decoration. It tells the model the data is
        # partial, so it narrows its next query instead of reporting a
        # confident answer built on a cut-off list.
        if len(output_text) > tool.policy.max_output_size:
            output_text = _truncate_marked(
                output_text,
                tool.policy.max_output_size,
                tool_name,
            )

        # Whichever bound is stricter wins, so one oversized result
        # cannot consume the whole request budget whatever the tool
        # asked for.
        allowance = self._output_allowance

        if (
            allowance is not None
            and len(output_text) > allowance
        ):
            output_text = _truncate_marked(
                output_text,
                allowance,
                tool_name,
            )

        return ToolResult(
            output=output_text,
        )

    @staticmethod
    def _check_permissions(
        tool: Tool[Any, Any],
        context: ToolContext,
    ) -> None:
        available_permissions = (
            context.permissions
            | context.approved_permissions
        )

        missing = (
            tool.policy.permissions
            - available_permissions
        )

        if missing:
            raise PermissionError(
                "Missing permissions: "
                + ", ".join(sorted(missing))
            )