from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
from scipy import signal

from ..config import IIRFilter
from ..iir import design_iir_sos, iir_frequency_response
from ..linear_fir import design_linear_phase_lr2_fir
from .crossover_cycles import (
    MAIN_MIN_CYCLES,
    adjust_kaiser_cycles_for_tap_budget,
    adjust_linked_kaiser_cycles_for_tap_budget,
)
from .crossover_fir import (
    align_crossover_firs,
    center_or_crop_crossover_fir,
    centered_delta,
    design_kaiser_crossover_fir,
)
from .crossover_models import (
    CrossoverBoundary,
    CrossoverFilterSpec,
    CrossoverPlan,
    crossover_overlap_shift,
)
from .crossover_validation import (
    CrossoverValidationIssue,
    raise_for_crossover_errors,
    validate_crossover_plan,
)


@dataclass(frozen=True)
class CrossoverWayResult:
    way_id: str
    fir: np.ndarray | None
    iir_filters: tuple[IIRFilter, ...]
    natural_fir_taps: int
    output_fir_taps: int
    biquad_sections: int
    description: str


@dataclass(frozen=True)
class CrossoverBankResult:
    sample_rate: int
    ways: tuple[CrossoverWayResult, ...]
    issues: tuple[CrossoverValidationIssue, ...]

    def way(self, way_id: str) -> CrossoverWayResult:
        for item in self.ways:
            if item.way_id == way_id:
                return item
        raise KeyError(way_id)


def _cycles_issue(
    code: str,
    message: str,
    *,
    boundary_id: str = "",
    way_id: str = "",
) -> CrossoverValidationIssue:
    return CrossoverValidationIssue(
        severity="warning",
        code=code,
        message=message,
        boundary_id=boundary_id,
        way_id=way_id,
    )


def _apply_cycles_to_tap_budget(
    plan: CrossoverPlan,
    sample_rate: int,
    tap_limits: Mapping[str, int],
) -> tuple[CrossoverPlan, tuple[CrossoverValidationIssue, ...]]:
    """Apply tap-budget Cycles changes before FIR generation and final crop."""

    adjustment_issues: list[CrossoverValidationIssue] = []
    boundaries: list[CrossoverBoundary] = []
    for boundary in plan.boundaries:
        if boundary.implementation != "dsp":
            boundaries.append(boundary)
            continue
        lower = boundary.lower_low_pass
        upper = boundary.upper_high_pass
        lower_limit = tap_limits.get(boundary.lower_way_id)
        upper_limit = tap_limits.get(boundary.upper_way_id)
        if lower.engine == "kaiser_fir" and upper.engine == "kaiser_fir" and (
            lower_limit is not None or upper_limit is not None
        ):
            # A linked boundary has one transition sharpness.  If legacy or
            # hand-edited data differs, start from the gentler requested side
            # and never increase either side automatically.
            requested = min(float(lower.cycles), float(upper.cycles))
            adjustment = adjust_linked_kaiser_cycles_for_tap_budget(
                sample_rate,
                boundary.frequency_hz,
                requested,
                lower_available_taps=lower_limit,
                upper_available_taps=upper_limit,
                attenuation_db=min(lower.attenuation_db, upper.attenuation_db),
            )
            effective = float(adjustment.effective_cycles)
            changed = (
                not np.isclose(lower.cycles, effective)
                or not np.isclose(upper.cycles, effective)
            )
            lower = replace(lower, cycles=effective)
            upper = replace(upper, cycles=effective)
            if changed:
                adjustment_issues.append(_cycles_issue(
                    "linked_cycles_adjusted",
                    (
                        f"{boundary.name or boundary.id}: LP／HPを共通 {effective:.3f} Cyclesへ調整しました"
                        f"（厳しい側のDevice {adjustment.available_taps} taps基準）。"
                    ),
                    boundary_id=boundary.id,
                ))
        else:
            sides = (
                ("LP", lower, lower_limit, boundary.lower_way_id),
                ("HP", upper, upper_limit, boundary.upper_way_id),
            )
            adjusted_sides: dict[str, CrossoverFilterSpec] = {"LP": lower, "HP": upper}
            for side, spec, limit, way_id in sides:
                if spec.engine != "kaiser_fir" or limit is None:
                    continue
                adjustment = adjust_kaiser_cycles_for_tap_budget(
                    sample_rate,
                    boundary.frequency_hz,
                    spec.cycles,
                    int(limit),
                    role="main",
                    attenuation_db=spec.attenuation_db,
                )
                adjusted_sides[side] = replace(spec, cycles=adjustment.effective_cycles)
                if adjustment.adjusted:
                    adjustment_issues.append(_cycles_issue(
                        "cycles_adjusted",
                        f"{boundary.name or boundary.id} {side}: {adjustment.effective_cycles:.3f} Cyclesへ調整しました。",
                        boundary_id=boundary.id,
                        way_id=way_id,
                    ))
            lower = adjusted_sides["LP"]
            upper = adjusted_sides["HP"]
        boundaries.append(replace(
            boundary,
            lower_low_pass=lower,
            upper_high_pass=upper,
        ))

    sub_crossovers = []
    for sub in plan.sub_crossovers:
        if sub.implementation != "dsp":
            sub_crossovers.append(sub)
            continue
        spec = sub.low_pass
        limit = tap_limits.get(sub.way_id)
        if spec.engine == "kaiser_fir" and limit is not None:
            adjustment = adjust_kaiser_cycles_for_tap_budget(
                sample_rate,
                sub.frequency_hz,
                spec.cycles,
                int(limit),
                role="sub",
                attenuation_db=spec.attenuation_db,
            )
            effective = float(adjustment.effective_cycles)
            if not adjustment.safe:
                effective = max(MAIN_MIN_CYCLES, effective)
                adjustment_issues.append(_cycles_issue(
                    "sub_cycles_safety_floor",
                    f"Sub {sub.way_id}: 減衰量の安全条件により3.000 Cycles未満へ自動調整しません。",
                    way_id=sub.way_id,
                ))
            elif adjustment.adjusted:
                adjustment_issues.append(_cycles_issue(
                    "cycles_adjusted",
                    f"Sub {sub.way_id}: {effective:.3f} Cyclesへ調整しました。",
                    way_id=sub.way_id,
                ))
            spec = replace(spec, cycles=effective)
        sub_crossovers.append(replace(sub, low_pass=spec))
    return (
        replace(plan, boundaries=tuple(boundaries), sub_crossovers=tuple(sub_crossovers)),
        tuple(adjustment_issues),
    )


def _iir_filter(mode: str, frequency_hz: float, spec: CrossoverFilterSpec) -> IIRFilter:
    if spec.engine != "iir":
        raise ValueError("IIR crossover design requires engine='iir'")
    return IIRFilter(
        kind="high_pass" if mode == "hp" else "low_pass",
        fc=float(frequency_hz),
        family=spec.family,
        order=int(spec.order),
        enabled=True,
        origin="manual",
    )


def _edge_label(mode: str, spec: CrossoverFilterSpec) -> str:
    prefix = mode.upper()
    if spec.engine == "off":
        return f"{prefix} Off"
    if spec.engine == "kaiser_fir":
        return f"{prefix} Kaiser FIR"
    if spec.engine == "linear_phase_lr2":
        return f"{prefix} Linear Phase LR2"
    family = {
        "linkwitz_riley": "Linkwitz-Riley",
        "butterworth": "Butterworth",
        "bessel": "Bessel",
    }[spec.family]
    return f"{prefix} {family} {spec.order}"


def _design_fir_edge(
    sample_rate: int,
    mode: str,
    frequency_hz: float,
    spec: CrossoverFilterSpec,
    *,
    tap_reference_frequency_hz: float | None = None,
    target_taps: int | None = None,
) -> np.ndarray:
    if spec.engine == "kaiser_fir":
        return design_kaiser_crossover_fir(
            sample_rate,
            mode,
            frequency_hz,
            spec,
            tap_reference_frequency_hz=tap_reference_frequency_hz,
        )
    if spec.engine == "linear_phase_lr2":
        if target_taps is None:
            raise ValueError("Linear Phase LR2 crossover requires a target tap count")
        taps = int(target_taps)
        design_taps = taps if taps % 2 else taps - 1
        if design_taps < 3:
            raise ValueError("Linear Phase LR2 crossover requires at least 3 taps")
        return design_linear_phase_lr2_fir(
            int(sample_rate),
            mode,
            float(frequency_hz),
            taps=design_taps,
        )
    raise ValueError("FIR crossover design requires a FIR engine")


def _design_complement_fir_band(
    sample_rate: int,
    lower_frequency_hz: float,
    upper_frequency_hz: float,
    lower_overlap_hz: float,
    upper_overlap_hz: float,
    high_pass_spec: CrossoverFilterSpec,
    low_pass_spec: CrossoverFilterSpec,
    target_taps: int | None,
) -> np.ndarray:
    lower_complement = _design_fir_edge(
        sample_rate,
        "lp",
        lower_frequency_hz - lower_overlap_hz,
        high_pass_spec,
        tap_reference_frequency_hz=lower_frequency_hz,
        target_taps=target_taps,
    )
    upper_complement = _design_fir_edge(
        sample_rate,
        "hp",
        upper_frequency_hz + upper_overlap_hz,
        low_pass_spec,
        tap_reference_frequency_hz=upper_frequency_hz,
        target_taps=target_taps,
    )
    target = max(lower_complement.size, upper_complement.size)
    lower_complement = center_or_crop_crossover_fir(lower_complement, target)
    upper_complement = center_or_crop_crossover_fir(upper_complement, target)
    return centered_delta(target) - lower_complement - upper_complement


def _main_way_result(
    plan: CrossoverPlan,
    way_id: str,
    sample_rate: int,
    target_taps: int | None,
) -> CrossoverWayResult:
    group = next(
        (item for item in plan.effective_main_way_groups if way_id in item),
        None,
    )
    if group is None:
        raise KeyError(way_id)
    index = group.index(way_id)
    boundaries_by_pair = {
        (item.lower_way_id, item.upper_way_id): item
        for item in plan.boundaries
    }
    lower_boundary: CrossoverBoundary | None = (
        boundaries_by_pair[(group[index - 1], way_id)] if index > 0 else None
    )
    upper_boundary: CrossoverBoundary | None = (
        boundaries_by_pair[(way_id, group[index + 1])]
        if index + 1 < len(group)
        else None
    )
    hp_spec = (
        lower_boundary.upper_high_pass
        if lower_boundary is not None and lower_boundary.implementation == "dsp"
        else None
    )
    lp_spec = (
        upper_boundary.lower_low_pass
        if upper_boundary is not None and upper_boundary.implementation == "dsp"
        else None
    )

    iir_filters: list[IIRFilter] = []
    if hp_spec is not None and hp_spec.engine == "iir":
        iir_filters.append(_iir_filter("hp", lower_boundary.frequency_hz, hp_spec))
    if lp_spec is not None and lp_spec.engine == "iir":
        iir_filters.append(_iir_filter("lp", upper_boundary.frequency_hz, lp_spec))

    fir_engines = {"kaiser_fir", "linear_phase_lr2"}
    hp_is_fir = hp_spec is not None and hp_spec.engine in fir_engines
    lp_is_fir = lp_spec is not None and lp_spec.engine in fir_engines
    hp_overlap = (
        crossover_overlap_shift(plan, lower_boundary)
        if lower_boundary is not None
        else 0.0
    )
    lp_overlap = (
        crossover_overlap_shift(plan, upper_boundary)
        if upper_boundary is not None
        else 0.0
    )
    fir: np.ndarray | None = None
    if hp_is_fir and lp_is_fir:
        assert lower_boundary is not None and upper_boundary is not None
        fir = _design_complement_fir_band(
            sample_rate,
            lower_boundary.frequency_hz,
            upper_boundary.frequency_hz,
            hp_overlap,
            lp_overlap,
            hp_spec,
            lp_spec,
            target_taps,
        )
    elif hp_is_fir:
        assert lower_boundary is not None and hp_spec is not None
        fir = _design_fir_edge(
            sample_rate,
            "hp",
            lower_boundary.frequency_hz - hp_overlap,
            hp_spec,
            tap_reference_frequency_hz=lower_boundary.frequency_hz,
            target_taps=target_taps,
        )
    elif lp_is_fir:
        assert upper_boundary is not None and lp_spec is not None
        fir = _design_fir_edge(
            sample_rate,
            "lp",
            upper_boundary.frequency_hz + lp_overlap,
            lp_spec,
            tap_reference_frequency_hz=upper_boundary.frequency_hz,
            target_taps=target_taps,
        )

    natural_taps = 0 if fir is None else int(fir.size)
    if fir is not None and target_taps is not None:
        fir = center_or_crop_crossover_fir(fir, target_taps)
    labels = []
    if hp_spec is not None:
        labels.append(_edge_label("hp", hp_spec))
    if lp_spec is not None:
        labels.append(_edge_label("lp", lp_spec))
    external_labels = {
        "external_passive": "External passive network",
        "external_active": "External active network",
    }
    if lower_boundary is not None and lower_boundary.implementation != "dsp":
        labels.append(external_labels[lower_boundary.implementation])
    if upper_boundary is not None and upper_boundary.implementation != "dsp":
        labels.append(external_labels[upper_boundary.implementation])
    sections = sum(design_iir_sos(item, sample_rate).shape[0] for item in iir_filters)
    return CrossoverWayResult(
        way_id=way_id,
        fir=fir,
        iir_filters=tuple(iir_filters),
        natural_fir_taps=natural_taps,
        output_fir_taps=0 if fir is None else int(fir.size),
        biquad_sections=int(sections),
        description=" / ".join(labels) or "Full range",
    )


def _sub_way_result(
    sub,
    sample_rate: int,
    target_taps: int | None,
) -> CrossoverWayResult:
    spec = sub.low_pass
    if sub.implementation != "dsp":
        label = (
            "External passive network"
            if sub.implementation == "external_passive"
            else "External active network"
        )
        return CrossoverWayResult(
            way_id=sub.way_id,
            fir=None,
            iir_filters=(),
            natural_fir_taps=0,
            output_fir_taps=0,
            biquad_sections=0,
            description=label,
        )
    iir_filters = (_iir_filter("lp", sub.frequency_hz, spec),) if spec.engine == "iir" else ()
    fir = (
        _design_fir_edge(
            int(sample_rate),
            "lp",
            sub.frequency_hz,
            spec,
            target_taps=target_taps,
        )
        if spec.engine in {"kaiser_fir", "linear_phase_lr2"}
        else None
    )
    natural_taps = 0 if fir is None else int(fir.size)
    if fir is not None and target_taps is not None:
        fir = center_or_crop_crossover_fir(fir, target_taps)
    sections = sum(design_iir_sos(item, int(sample_rate)).shape[0] for item in iir_filters)
    return CrossoverWayResult(
        way_id=sub.way_id,
        fir=fir,
        iir_filters=iir_filters,
        natural_fir_taps=natural_taps,
        output_fir_taps=0 if fir is None else int(fir.size),
        biquad_sections=int(sections),
        description=_edge_label("lp", spec),
    )


def generate_crossover_way(
    plan: CrossoverPlan,
    way_id: str,
    sample_rate: int,
    *,
    target_taps: int | None = None,
) -> CrossoverWayResult:
    """Generate one Way from an already budget-fitted canonical plan."""

    issues = validate_crossover_plan(plan, sample_rates=(int(sample_rate),))
    raise_for_crossover_errors(issues)
    if way_id in plan.main_way_ids:
        return _main_way_result(
            plan,
            way_id,
            int(sample_rate),
            target_taps,
        )
    sub = next((item for item in plan.sub_crossovers if item.way_id == way_id), None)
    if sub is None:
        raise KeyError(way_id)
    return _sub_way_result(sub, int(sample_rate), target_taps)


def generate_crossover_bank(
    plan: CrossoverPlan,
    sample_rate: int,
    *,
    target_taps_by_way: Mapping[str, int] | None = None,
) -> CrossoverBankResult:
    tap_limits = target_taps_by_way or {}
    effective_plan, adjustment_issues = _apply_cycles_to_tap_budget(
        plan,
        int(sample_rate),
        tap_limits,
    )
    issues = validate_crossover_plan(
        effective_plan,
        sample_rates=(int(sample_rate),),
        target_taps_by_way=tap_limits,
    )
    issues = adjustment_issues + issues
    raise_for_crossover_errors(issues)
    ways = [
        _main_way_result(effective_plan, way_id, int(sample_rate), tap_limits.get(way_id))
        for way_id in effective_plan.main_way_ids
    ]
    for sub in effective_plan.sub_crossovers:
        ways.append(_sub_way_result(sub, int(sample_rate), tap_limits.get(sub.way_id)))
    return CrossoverBankResult(int(sample_rate), tuple(ways), issues)


def crossover_way_response(
    result: CrossoverWayResult,
    sample_rate: int,
    frequency_hz: np.ndarray,
    *,
    fir_override: np.ndarray | None = None,
) -> np.ndarray:
    frequency = np.asarray(frequency_hz, dtype=float)
    fir = result.fir if fir_override is None else np.asarray(fir_override, dtype=float)
    if fir is None:
        fir_response = np.ones_like(frequency, dtype=complex)
    else:
        _axis, fir_response = signal.freqz(fir, worN=frequency, fs=float(sample_rate))
    return np.asarray(fir_response, dtype=complex) * iir_frequency_response(
        result.iir_filters,
        int(sample_rate),
        frequency,
    )


def electrical_sum_response(
    bank: CrossoverBankResult,
    frequency_hz: np.ndarray,
    *,
    way_ids: set[str] | None = None,
    align_fir: bool = True,
) -> np.ndarray:
    selected = [item for item in bank.ways if way_ids is None or item.way_id in way_ids]
    frequency = np.asarray(frequency_hz, dtype=float)
    if not selected:
        return np.zeros_like(frequency, dtype=complex)
    aligned = (
        align_crossover_firs({item.way_id: item.fir for item in selected})
        if align_fir
        else {item.way_id: item.fir for item in selected}
    )
    total = np.zeros_like(frequency, dtype=complex)
    for item in selected:
        total += crossover_way_response(
            item,
            bank.sample_rate,
            frequency,
            fir_override=aligned[item.way_id] if align_fir else None,
        )
    return total
