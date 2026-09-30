from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Sequence

import numpy as np

from ..config import LinearFIRFilter
from .crossover_bank import CrossoverWayResult, generate_crossover_way
from .crossover_cycles import MAIN_MIN_CYCLES, adjust_kaiser_cycles_for_tap_budget
from .crossover_models import (
    CrossoverBoundary,
    CrossoverFilterSpec,
    CrossoverPlan,
    SubCrossover,
    crossover_overlap_shift,
    default_crossover_plan,
)
from .crossover_validation import CrossoverValidationIssue, validate_crossover_plan
from .models import DSPSystem, DSPWay


@dataclass(frozen=True)
class DSPCrossoverPreparation:
    plan: CrossoverPlan
    ways: dict[str, CrossoverWayResult]
    issues: tuple[CrossoverValidationIssue, ...]


def is_sub_way(way: DSPWay) -> bool:
    name = way.name.strip().casefold()
    return (
        name == "sub"
        or name.startswith("sub ")
        or name.startswith("sub-")
        or name.endswith(" sub")
    )


def crossover_way_ids(ways: Sequence[DSPWay]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    main = [way.id for way in ways if not is_sub_way(way)]
    subs = [way.id for way in ways if is_sub_way(way)]
    # DSP output rows are Hi -> Low, while CrossoverPlan is Low -> High.
    return tuple(reversed(main)), tuple(subs)


def crossover_plan_for_ways(
    ways: Sequence[DSPWay],
    *,
    main_way_groups: tuple[tuple[str, ...], ...] | None = None,
) -> CrossoverPlan:
    by_id = {way.id: way for way in ways}
    inferred_main_ids, sub_ids = crossover_way_ids(ways)
    groups = main_way_groups or (inferred_main_ids,)
    main_ids = tuple(way_id for group in groups for way_id in group)
    if not main_ids:
        raise ValueError("at least one Main Way is required")
    if set(main_ids) != set(inferred_main_ids):
        raise ValueError("main Way groups must match non-Sub Ways")
    boundaries: list[CrossoverBoundary] = []
    for group_index, group in enumerate(groups):
        defaults = default_crossover_plan(group, engine="linear_phase_lr2")
        for boundary_index, default in enumerate(defaults.boundaries):
            lower_way = by_id[default.lower_way_id]
            upper_way = by_id[default.upper_way_id]
            boundary_id = (
                default.id
                if len(groups) == 1
                else f"crossover-{group_index + 1}-{boundary_index + 1}"
            )
            boundaries.append(replace(
                default,
                id=boundary_id,
                name=f"{lower_way.name} / {upper_way.name}",
            ))
    subs: list[SubCrossover] = []
    default_sub_spec = CrossoverFilterSpec(engine="linear_phase_lr2")
    for sub_id in sub_ids:
        subs.append(SubCrossover(
            way_id=sub_id,
            frequency_hz=80.0,
            low_pass=default_sub_spec,
        ))
    return CrossoverPlan(
        main_way_ids=main_ids,
        boundaries=tuple(boundaries),
        sub_crossovers=tuple(subs),
        main_way_groups=groups,
    )


def sync_crossover_plan(
    plan: CrossoverPlan | None,
    ways: Sequence[DSPWay],
) -> CrossoverPlan:
    current_main_ids, _sub_ids = crossover_way_ids(ways)
    preserved_groups = None
    if plan is not None:
        candidate_groups = plan.effective_main_way_groups
        candidate_ids = tuple(way_id for group in candidate_groups for way_id in group)
        if set(candidate_ids) == set(current_main_ids):
            preserved_groups = candidate_groups
    fallback = crossover_plan_for_ways(ways, main_way_groups=preserved_groups)
    if plan is None:
        return fallback
    existing_boundaries = {
        (item.lower_way_id, item.upper_way_id): item for item in plan.boundaries
    }
    boundaries = tuple(
        replace(
            existing_boundaries.get(
                (item.lower_way_id, item.upper_way_id),
                item,
            ),
            lower_way_id=item.lower_way_id,
            upper_way_id=item.upper_way_id,
        )
        for item in fallback.boundaries
    )
    existing_subs = {item.way_id: item for item in plan.sub_crossovers}
    subs = tuple(existing_subs.get(item.way_id, item) for item in fallback.sub_crossovers)
    return replace(
        fallback,
        boundaries=boundaries,
        sub_crossovers=subs,
        revision=max(1, int(plan.revision)),
        overlap_convention=plan.overlap_convention,
    )


def ensure_system_crossover_plan(system: DSPSystem) -> DSPSystem:
    plan = sync_crossover_plan(system.crossover_plan, system.ways)
    if plan == system.crossover_plan:
        return system
    return replace(system, crossover_plan=plan)


def _issue(
    code: str,
    message: str,
    *,
    boundary_id: str = "",
    way_id: str = "",
) -> CrossoverValidationIssue:
    return CrossoverValidationIssue("warning", code, message, boundary_id, way_id)


def _fit_plan_to_devices(system: DSPSystem, plan: CrossoverPlan) -> tuple[CrossoverPlan, tuple[CrossoverValidationIssue, ...]]:
    way_by_id = {way.id: way for way in system.ways}
    issues: list[CrossoverValidationIssue] = []
    boundaries: list[CrossoverBoundary] = []
    for boundary in plan.boundaries:
        if boundary.implementation != "dsp":
            boundaries.append(boundary)
            continue
        lower = boundary.lower_low_pass
        upper = boundary.upper_high_pass
        if lower.engine == "kaiser_fir" and upper.engine == "kaiser_fir":
            requested = min(float(lower.cycles), float(upper.cycles))
            side_adjustments = []
            for way_id, attenuation in (
                (boundary.lower_way_id, lower.attenuation_db),
                (boundary.upper_way_id, upper.attenuation_db),
            ):
                way = way_by_id[way_id]
                device = system.device(way.dsp_id)
                side_adjustments.append(adjust_kaiser_cycles_for_tap_budget(
                    device.sample_rate,
                    boundary.frequency_hz,
                    requested,
                    device.taps,
                    role="main",
                    attenuation_db=attenuation,
                ))
            effective = min(item.effective_cycles for item in side_adjustments)
            changed = not np.isclose(lower.cycles, effective) or not np.isclose(upper.cycles, effective)
            lower = replace(lower, cycles=effective)
            upper = replace(upper, cycles=effective)
            if changed:
                issues.append(_issue(
                    "linked_cycles_adjusted",
                    f"{boundary.name or boundary.id}: LP／HPを共通 {effective:.3f} Cyclesへ調整しました。",
                    boundary_id=boundary.id,
                ))
        else:
            next_specs = []
            for spec, way_id, side in (
                (lower, boundary.lower_way_id, "LP"),
                (upper, boundary.upper_way_id, "HP"),
            ):
                if spec.engine != "kaiser_fir":
                    next_specs.append(spec)
                    continue
                way = way_by_id[way_id]
                device = system.device(way.dsp_id)
                adjustment = adjust_kaiser_cycles_for_tap_budget(
                    device.sample_rate,
                    boundary.frequency_hz,
                    spec.cycles,
                    device.taps,
                    role="main",
                    attenuation_db=spec.attenuation_db,
                )
                next_specs.append(replace(spec, cycles=adjustment.effective_cycles))
                if adjustment.adjusted:
                    issues.append(_issue(
                        "cycles_adjusted",
                        f"{boundary.name or boundary.id} {side}: {adjustment.effective_cycles:.3f} Cyclesへ調整しました。",
                        boundary_id=boundary.id,
                        way_id=way_id,
                    ))
            lower, upper = next_specs
        boundaries.append(replace(boundary, lower_low_pass=lower, upper_high_pass=upper))

    subs: list[SubCrossover] = []
    for sub in plan.sub_crossovers:
        if sub.implementation != "dsp":
            subs.append(sub)
            continue
        spec = sub.low_pass
        if spec.engine == "kaiser_fir":
            way = way_by_id[sub.way_id]
            device = system.device(way.dsp_id)
            adjustment = adjust_kaiser_cycles_for_tap_budget(
                device.sample_rate,
                sub.frequency_hz,
                spec.cycles,
                device.taps,
                role="sub",
                attenuation_db=spec.attenuation_db,
            )
            effective = adjustment.effective_cycles if adjustment.safe else max(MAIN_MIN_CYCLES, adjustment.effective_cycles)
            if not np.isclose(spec.cycles, effective):
                issues.append(_issue(
                    "cycles_adjusted",
                    f"Sub {sub.way_id}: {effective:.3f} Cyclesへ調整しました。",
                    way_id=sub.way_id,
                ))
            spec = replace(spec, cycles=effective)
        subs.append(replace(sub, low_pass=spec))
    return replace(plan, boundaries=tuple(boundaries), sub_crossovers=tuple(subs)), tuple(issues)


def prepare_system_crossovers(system: DSPSystem) -> DSPCrossoverPreparation:
    normalized = ensure_system_crossover_plan(system)
    assert normalized.crossover_plan is not None
    plan, adjustment_issues = _fit_plan_to_devices(normalized, normalized.crossover_plan)
    rates = tuple(sorted({device.sample_rate for device in normalized.devices}))
    validation = validate_crossover_plan(plan, sample_rates=rates)
    errors = [item.message for item in validation if item.severity == "error"]
    if errors:
        raise ValueError("; ".join(errors))
    results: dict[str, CrossoverWayResult] = {}
    for way in normalized.ways:
        device = normalized.device(way.dsp_id)
        results[way.id] = generate_crossover_way(
            plan,
            way.id,
            device.sample_rate,
            target_taps=device.taps,
        )
    unique: list[CrossoverValidationIssue] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in adjustment_issues + validation:
        key = (item.code, item.message, item.boundary_id, item.way_id)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return DSPCrossoverPreparation(plan, results, tuple(unique))


def crossover_mask_filters(plan: CrossoverPlan, way_id: str) -> tuple[LinearFIRFilter, ...]:
    filters: list[LinearFIRFilter] = []
    for boundary in plan.boundaries:
        if boundary.implementation != "dsp":
            continue
        shift = crossover_overlap_shift(plan, boundary)
        if boundary.lower_way_id == way_id:
            spec = boundary.lower_low_pass
            mode = "lp"
            frequency = boundary.frequency_hz + shift
        elif boundary.upper_way_id == way_id:
            spec = boundary.upper_high_pass
            mode = "hp"
            frequency = boundary.frequency_hz - shift
        else:
            continue
        if spec.engine == "off":
            continue
        response = "kaiser" if spec.engine == "kaiser_fir" else "linear_phase_lr2"
        filters.append(LinearFIRFilter(
            mode=mode,
            response=response,
            fc=float(frequency),
            cycles=float(spec.cycles),
            beta=float(spec.beta),
        ))
    sub = next((item for item in plan.sub_crossovers if item.way_id == way_id), None)
    if sub is not None and sub.implementation == "dsp" and sub.low_pass.engine != "off":
        spec = sub.low_pass
        filters.append(LinearFIRFilter(
            mode="lp",
            response="kaiser" if spec.engine == "kaiser_fir" else "linear_phase_lr2",
            fc=float(sub.frequency_hz),
            cycles=float(spec.cycles),
            beta=float(spec.beta),
        ))
    return tuple(filters)


def crossover_description(plan: CrossoverPlan, way_id: str) -> str:
    labels: list[str] = []
    names = {
        "off": "Off",
        "linear_phase_lr2": "Linear phase LR2",
        "kaiser_fir": "Kaiser FIR",
        "iir": "IIR",
    }
    external_names = {
        "external_passive": "External passive",
        "external_active": "External active",
    }
    for boundary in plan.boundaries:
        if boundary.lower_way_id == way_id:
            if boundary.implementation == "dsp":
                labels.append(f"LP {boundary.frequency_hz:g} Hz {names[boundary.lower_low_pass.engine]}")
            else:
                labels.append(f"{external_names[boundary.implementation]} {boundary.frequency_hz:g} Hz · DSP full range")
        elif boundary.upper_way_id == way_id:
            if boundary.implementation == "dsp":
                labels.append(f"HP {boundary.frequency_hz:g} Hz {names[boundary.upper_high_pass.engine]}")
            else:
                labels.append(f"{external_names[boundary.implementation]} {boundary.frequency_hz:g} Hz · DSP full range")
    for sub in plan.sub_crossovers:
        if sub.way_id == way_id:
            if sub.implementation == "dsp":
                labels.append(f"LP {sub.frequency_hz:g} Hz {names[sub.low_pass.engine]}")
            else:
                labels.append(f"{external_names[sub.implementation]} {sub.frequency_hz:g} Hz · DSP full range")
    return " / ".join(labels) or "Full range"
