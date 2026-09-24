from dataclasses import replace

import pytest

from core.entities.models.model_catalog import (
    ModelAttributes,
    ModelCapability,
    ModelCatalog,
    ModelProfile,
    ModelRequirements,
    ModelSpeed,
)
from core.entities.models.model_router import (
    DefaultModelRouter,
    ModelRoutingError,
)
from core.entities.models.subagent import (
    SubagentPower,
    SubagentTask,
    TaskComplexity,
    TaskProfile,
)


def _profile(
    name: str,
    quality: int,
    usage: int,
    speed: ModelSpeed = ModelSpeed.FAST,
    vram: float | None = None,
    coding: int = 3,
) -> ModelProfile:
    return ModelProfile(
        name=name,
        description="",
        capabilities={
            ModelCapability.GENERAL: 3,
            ModelCapability.CODING: coding,
        },
        attributes=ModelAttributes(
            speed=speed,
            quality=quality,
            resource_usage=usage,
        ),
        requirements=ModelRequirements(
            context_size=8192,
            vram_gb=vram,
        ),
    )


def _catalog() -> ModelCatalog:
    return ModelCatalog(
        models=(
            _profile("e2b", quality=2, usage=1, vram=3.0),
            _profile("e4b", quality=3, usage=3, vram=5.0),
            _profile(
                "big",
                quality=5,
                usage=4,
                speed=ModelSpeed.MEDIUM,
                vram=9.0,
            ),
        ),
    )


def _task(
    complexity: TaskComplexity = TaskComplexity.MEDIUM,
    power: SubagentPower = SubagentPower.AUTO,
) -> SubagentTask:
    return SubagentTask(
        role="researcher",
        objective="do work",
        profile=TaskProfile(
            capability=ModelCapability.GENERAL,
            complexity=complexity,
            power=power,
        ),
    )


def test_explicit_power_selects_tier() -> None:
    router = DefaultModelRouter(_catalog())

    assert (
        router.select(
            _task(power=SubagentPower.LOW),
        )
        == "e2b"
    )
    assert (
        router.select(
            _task(power=SubagentPower.HIGH),
        )
        == "big"
    )


def test_explicit_model_bypasses_router() -> None:
    router = DefaultModelRouter(_catalog())

    task = replace(
        _task(power=SubagentPower.LOW),
        model="big",
    )

    assert router.select(task) == "big"


def test_auto_complex_reaches_high() -> None:
    router = DefaultModelRouter(_catalog())

    assert (
        router.select(
            _task(complexity=TaskComplexity.COMPLEX),
        )
        == "big"
    )


def test_auto_non_complex_never_high() -> None:
    router = DefaultModelRouter(_catalog())

    assert (
        router.select(
            _task(complexity=TaskComplexity.SIMPLE),
        )
        == "e2b"
    )
    assert (
        router.select(
            _task(complexity=TaskComplexity.MEDIUM),
        )
        != "big"
    )


def test_vram_budget_filters() -> None:
    router = DefaultModelRouter(
        _catalog(),
        vram_budget_gb=6.0,
    )

    assert (
        router.select(
            _task(complexity=TaskComplexity.COMPLEX),
        )
        == "e4b"
    )

    # HIGH now means best-available: heavy "big" is over
    # budget, so the tier resolves to e4b without error.
    assert (
        router.select(
            _task(power=SubagentPower.HIGH),
        )
        == "e4b"
    )

    tiny = DefaultModelRouter(
        _catalog(),
        vram_budget_gb=2.0,
    )

    with pytest.raises(ModelRoutingError):
        tiny.select(
            _task(power=SubagentPower.HIGH),
        )


def test_resident_affinity_wins_ties() -> None:
    catalog = ModelCatalog(
        models=(
            _profile("first", quality=3, usage=2, vram=5.0),
            _profile("second", quality=3, usage=2, vram=5.0),
        ),
    )

    plain = DefaultModelRouter(catalog)
    assert plain.select(_task()) == "first"

    affinity = DefaultModelRouter(
        catalog,
        resident_provider=lambda: "second",
    )
    assert affinity.select(_task()) == "second"


def test_single_model_mode_pins_main() -> None:
    router = DefaultModelRouter(
        _catalog(),
        main_model="e2b",
        single_model_mode=True,
    )

    assert (
        router.select(
            _task(
                complexity=TaskComplexity.COMPLEX,
                power=SubagentPower.HIGH,
            ),
        )
        == "e2b"
    )


def test_power_for_known_and_unknown() -> None:
    router = DefaultModelRouter(_catalog())

    assert router.power_for("e2b") is SubagentPower.LOW
    assert router.power_for("e4b") is SubagentPower.MEDIUM
    assert router.power_for("big") is SubagentPower.HIGH
    assert router.power_for("missing") is None


def test_high_picks_best_available() -> None:
    catalog = ModelCatalog(
        models=(
            _profile(
                "main",
                quality=4,
                usage=3,
                speed=ModelSpeed.MEDIUM,
                vram=9.0,
            ),
            _profile(
                "coder",
                quality=4,
                usage=4,
                speed=ModelSpeed.MEDIUM,
                vram=9.0,
                coding=5,
            ),
        ),
    )
    router = DefaultModelRouter(catalog)

    general = SubagentTask(
        role="researcher",
        objective="hard analysis",
        profile=TaskProfile(
            capability=ModelCapability.GENERAL,
            complexity=TaskComplexity.COMPLEX,
            power=SubagentPower.HIGH,
        ),
    )

    assert router.select(general) == "main"

    coding = SubagentTask(
        role="coder",
        objective="hard refactor",
        profile=TaskProfile(
            capability=ModelCapability.CODING,
            complexity=TaskComplexity.COMPLEX,
            power=SubagentPower.HIGH,
        ),
    )

    assert router.select(coding) == "coder"


def test_empty_catalog_raises() -> None:
    router = DefaultModelRouter(ModelCatalog(models=()))

    with pytest.raises(ModelRoutingError):
        router.select(_task())
