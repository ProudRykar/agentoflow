from __future__ import annotations

from dataclasses import dataclass

from agent_workflow.core.context.task_state import TaskState


@dataclass(slots=True, frozen=True)
class SynthesisGuide:
    """Checklist shaping the final answer. Advisory, but the
    code rule is enforced by a blocking verification pass."""

    sections: tuple[str, ...]
    rules: tuple[str, ...]

    def render(self) -> str:
        lines = ["[SYNTHESIS GUIDE]"]

        if self.sections:
            lines.append("Required sections:")

            lines.extend(
                f"{index}. {section}"
                for index, section in enumerate(
                    self.sections,
                    start=1,
                )
            )

        if self.rules:
            lines.append("Rules:")

            lines.extend(
                f"- {rule}" for rule in self.rules
            )

        return "\n".join(lines)


GROUNDING_RULE = (
    "Reproduce API patterns, signatures, and commands only "
    "as seen in [EVIDENCE] excerpts. If a pattern is not in "
    "the evidence, omit it or mark it explicitly as unverified. "
    "Never invent API shapes from prior knowledge."
)

CITATION_RULE = (
    "End every factual section with the source URLs taken "
    "from the evidence blocks above. Resources without URLs "
    "are not resources."
)

REQUIREMENTS_RULE = (
    "Carry installation commands, version requirements, and "
    "launch commands over verbatim when the evidence "
    "contains them."
)


def build_synthesis_guide(
    task_state: TaskState,
) -> SynthesisGuide:
    """Deterministic guide for the final answer.

    Research tasks get sections; everything gets grounding.
    Table shape for tools is deliberately NOT mandated:
    citations are required, layout is the model's choice.
    """

    contract = task_state.contract

    if contract.requires_research or contract.research is not None:
        return SynthesisGuide(
            sections=(
                "Overview: what it is, in two or three sentences.",
                "Tools and capabilities actually fetched, each "
                "with its source URL.",
                "Step-by-step usage guide grounded in fetched pages.",
                "Resources: every item with its URL.",
            ),
            rules=(
                GROUNDING_RULE,
                CITATION_RULE,
                REQUIREMENTS_RULE,
            ),
        )

    return SynthesisGuide(
        sections=(),
        rules=(GROUNDING_RULE,),
    )
