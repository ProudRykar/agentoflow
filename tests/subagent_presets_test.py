from core.entities.models.builtin.subagent import (
    SubagentRunInput,
    apply_role_preset,
)
from core.entities.models.model_catalog import ModelCapability
from core.entities.models.subagent import SubagentPower


def _input(**overrides: object) -> SubagentRunInput:
    base: dict[str, object] = {
        "role": "researcher",
        "objective": "find docs",
    }
    base.update(overrides)

    return SubagentRunInput(**base)  # type: ignore[arg-type]


def test_researcher_preset_fills_defaults() -> None:
    result = apply_role_preset(_input())

    assert result.capability is ModelCapability.RESEARCH
    assert result.power is SubagentPower.LOW
    assert result.tools == (
        "web_fetch",
        "web_crawl",
    )


def test_explicit_fields_win_over_preset() -> None:
    result = apply_role_preset(
        _input(
            capability=ModelCapability.CODING,
            power=SubagentPower.HIGH,
            tools=("read_file",),
        )
    )

    assert result.capability is ModelCapability.CODING
    assert result.power is SubagentPower.HIGH
    assert result.tools == ("read_file",)


def test_unknown_role_untouched() -> None:
    original = _input(role="analyst")

    assert apply_role_preset(original) is original


def test_coder_preset() -> None:
    result = apply_role_preset(_input(role="coder"))

    assert result.capability is ModelCapability.CODING
    assert result.power is SubagentPower.MEDIUM
    assert "write_file" in result.tools
