from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

from .crossover_kaiser import (
    kaiser_attenuation_db_from_beta,
    kaiser_beta_from_attenuation_db,
    kaiser_crossover_preset,
    legacy_kaiser_beta,
)


CrossoverEngine = Literal["off", "linear_phase_lr2", "kaiser_fir", "iir"]
CrossoverIIRFamily = Literal["linkwitz_riley", "butterworth", "bessel"]
CrossoverOverlapConvention = Literal["multiway_fir_studio", "per_side"]
CrossoverImplementation = Literal["dsp", "external_passive", "external_active"]
DEFAULT_CROSSOVER_FREQUENCIES_HZ = {
    1: (),
    2: (1_800.0,),
    3: (300.0, 3_000.0),
    4: (100.0, 800.0, 5_000.0),
}


@dataclass(frozen=True, init=False)
class CrossoverFilterSpec:
    """One HP or LP side of a shared crossover boundary."""

    engine: CrossoverEngine
    cycles: float
    attenuation_db: float
    family: CrossoverIIRFamily
    order: int

    def __init__(
        self,
        engine: CrossoverEngine = "kaiser_fir",
        cycles: float = 4.0,
        attenuation_db: float | None = None,
        family: CrossoverIIRFamily = "linkwitz_riley",
        order: int = 4,
        *,
        beta: float | None = None,
    ) -> None:
        if attenuation_db is None:
            attenuation = (
                kaiser_crossover_preset("standard").attenuation_db
                if beta is None
                else kaiser_attenuation_db_from_beta(beta)
            )
        else:
            attenuation = float(attenuation_db)
            if beta is not None:
                legacy_attenuation = kaiser_attenuation_db_from_beta(beta)
                if abs(attenuation - legacy_attenuation) > 0.5:
                    raise ValueError("Kaiser attenuation and legacy beta differ by more than 0.5 dB")
        object.__setattr__(self, "engine", engine)
        object.__setattr__(self, "cycles", float(cycles))
        object.__setattr__(self, "attenuation_db", attenuation)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "order", int(order))

    @property
    def beta(self) -> float:
        """Exact derived beta used by the FIR designer."""
        return kaiser_beta_from_attenuation_db(self.attenuation_db)

    @property
    def legacy_beta(self) -> float:
        """One-decimal beta for import/export compatibility."""
        return legacy_kaiser_beta(self.attenuation_db)

    def validate(self) -> None:
        if self.engine not in {"off", "linear_phase_lr2", "kaiser_fir", "iir"}:
            raise ValueError(f"unsupported crossover engine: {self.engine}")
        if self.engine == "kaiser_fir":
            if not math.isfinite(self.cycles) or self.cycles <= 0.0:
                raise ValueError("Kaiser FIR cycles must be positive")
            if not math.isfinite(self.attenuation_db) or self.attenuation_db < 21.0:
                raise ValueError("Kaiser attenuation must be at least 21 dB")
        if self.engine == "iir":
            if self.family not in {"linkwitz_riley", "butterworth", "bessel"}:
                raise ValueError(f"unsupported crossover IIR family: {self.family}")
            allowed = {2, 4, 6, 8} if self.family == "linkwitz_riley" else {1, 2, 3, 4, 6, 8}
            if self.order not in allowed:
                raise ValueError(f"unsupported {self.family} crossover order: {self.order}")


@dataclass(frozen=True)
class CrossoverBoundary:
    """A shared frequency boundary between adjacent low-to-high main Ways."""

    id: str
    name: str
    lower_way_id: str
    upper_way_id: str
    frequency_hz: float
    lower_low_pass: CrossoverFilterSpec = field(default_factory=CrossoverFilterSpec)
    upper_high_pass: CrossoverFilterSpec = field(default_factory=CrossoverFilterSpec)
    overlap_hz: float = 0.0
    implementation: CrossoverImplementation = "dsp"

    def validate(self) -> None:
        if not self.id.strip() or not self.lower_way_id.strip() or not self.upper_way_id.strip():
            raise ValueError("crossover boundary ids must not be empty")
        if self.lower_way_id == self.upper_way_id:
            raise ValueError("crossover boundary must connect two different Ways")
        if self.frequency_hz <= 0.0:
            raise ValueError("crossover frequency must be positive")
        if self.overlap_hz < 0.0:
            raise ValueError("crossover overlap must not be negative")
        if self.implementation not in {"dsp", "external_passive", "external_active"}:
            raise ValueError(f"unsupported crossover implementation: {self.implementation}")
        if self.implementation == "dsp":
            self.lower_low_pass.validate()
            self.upper_high_pass.validate()


@dataclass(frozen=True)
class SubCrossover:
    """Independent low-pass crossover shared by one Sub output."""

    way_id: str
    frequency_hz: float = 80.0
    low_pass: CrossoverFilterSpec = field(default_factory=CrossoverFilterSpec)
    implementation: CrossoverImplementation = "dsp"

    def validate(self) -> None:
        if not self.way_id.strip():
            raise ValueError("Sub crossover way_id must not be empty")
        if self.frequency_hz <= 0.0:
            raise ValueError("Sub crossover frequency must be positive")
        if self.implementation not in {"dsp", "external_passive", "external_active"}:
            raise ValueError(f"unsupported crossover implementation: {self.implementation}")
        if self.implementation == "dsp":
            self.low_pass.validate()


@dataclass(frozen=True)
class CrossoverPlan:
    """Canonical crossover settings, independent of UI and DSP devices."""

    main_way_ids: tuple[str, ...]
    boundaries: tuple[CrossoverBoundary, ...]
    sub_crossovers: tuple[SubCrossover, ...] = ()
    revision: int = 1
    overlap_convention: CrossoverOverlapConvention = "multiway_fir_studio"
    main_way_groups: tuple[tuple[str, ...], ...] = ()

    @property
    def effective_main_way_groups(self) -> tuple[tuple[str, ...], ...]:
        """Return independent low-to-high Main channel groups.

        Older saved systems have no group field and remain one-channel plans.
        """
        return self.main_way_groups or (self.main_way_ids,)

    def validate(self) -> None:
        if not self.main_way_ids:
            raise ValueError("at least one main Way is required")
        if any(not way_id.strip() for way_id in self.main_way_ids):
            raise ValueError("main Way ids must not be empty")
        if len(self.main_way_ids) != len(set(self.main_way_ids)):
            raise ValueError("main Way ids must be unique")
        groups = self.effective_main_way_groups
        if any(not group for group in groups):
            raise ValueError("main Way groups must not be empty")
        grouped_ids = tuple(way_id for group in groups for way_id in group)
        if grouped_ids != self.main_way_ids:
            raise ValueError("main Way groups must contain every main Way in order")
        expected_pairs = tuple(
            (group[index], group[index + 1])
            for group in groups
            for index in range(len(group) - 1)
        )
        if len(self.boundaries) != len(expected_pairs):
            raise ValueError("boundary count must match independent main Way groups")
        actual_pairs = tuple(
            (boundary.lower_way_id, boundary.upper_way_id)
            for boundary in self.boundaries
        )
        if actual_pairs != expected_pairs:
            raise ValueError("crossover boundaries must connect adjacent Ways within each group")
        if self.revision < 1:
            raise ValueError("crossover revision must be positive")
        if self.overlap_convention not in {"multiway_fir_studio", "per_side"}:
            raise ValueError(f"unsupported crossover overlap convention: {self.overlap_convention}")
        for boundary in self.boundaries:
            boundary.validate()
            if boundary.frequency_hz - crossover_overlap_shift(self, boundary) <= 0.0:
                raise ValueError("crossover overlap moves the high-pass cutoff to zero or below")
        for sub in self.sub_crossovers:
            sub.validate()


def crossover_overlap_shift(plan: CrossoverPlan, boundary: CrossoverBoundary) -> float:
    """Return the per-side cutoff shift for the selected compatibility rule."""
    if boundary.implementation != "dsp":
        return 0.0
    overlap = max(0.0, float(boundary.overlap_hz))
    group_size = next(
        (len(group) for group in plan.effective_main_way_groups if boundary.lower_way_id in group),
        len(plan.main_way_ids),
    )
    if plan.overlap_convention == "multiway_fir_studio" and group_size == 2:
        return overlap / 2.0
    return overlap


def default_crossover_plan(
    main_way_ids: tuple[str, ...],
    *,
    frequencies_hz: tuple[float, ...] | None = None,
    engine: CrossoverEngine = "kaiser_fir",
    kaiser_preset: str = "standard",
    sub_way_ids: tuple[str, ...] = (),
    sub_frequency_hz: float = 80.0,
) -> CrossoverPlan:
    """Create a low-to-high plan with linked settings on both sides."""
    if frequencies_hz is None:
        if len(main_way_ids) not in DEFAULT_CROSSOVER_FREQUENCIES_HZ:
            raise ValueError("frequencies_hz is required outside 1 to 4 main Ways")
        frequencies_hz = DEFAULT_CROSSOVER_FREQUENCIES_HZ[len(main_way_ids)]
    if len(frequencies_hz) != max(0, len(main_way_ids) - 1):
        raise ValueError("frequency count must equal main Way count minus one")
    preset = kaiser_crossover_preset(kaiser_preset)
    spec = CrossoverFilterSpec(
        engine=engine,
        cycles=preset.cycles,
        attenuation_db=preset.attenuation_db,
    )
    boundaries = tuple(
        CrossoverBoundary(
            id=f"crossover-{index + 1}",
            name=f"{main_way_ids[index]} / {main_way_ids[index + 1]}",
            lower_way_id=main_way_ids[index],
            upper_way_id=main_way_ids[index + 1],
            frequency_hz=float(frequency),
            lower_low_pass=spec,
            upper_high_pass=spec,
        )
        for index, frequency in enumerate(frequencies_hz)
    )
    return CrossoverPlan(
        main_way_ids=tuple(main_way_ids),
        boundaries=boundaries,
        sub_crossovers=tuple(
            SubCrossover(way_id=way_id, frequency_hz=float(sub_frequency_hz), low_pass=spec)
            for way_id in sub_way_ids
        ),
        main_way_groups=(tuple(main_way_ids),),
    )
