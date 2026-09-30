from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .crossover_fir import kaiser_crossover_taps
from .crossover_models import (
    CrossoverBoundary,
    CrossoverFilterSpec,
    CrossoverPlan,
    crossover_overlap_shift,
)


@dataclass(frozen=True)
class CrossoverValidationIssue:
    severity: str
    code: str
    message: str
    boundary_id: str = ""
    way_id: str = ""


def _issue(
    severity: str,
    code: str,
    message: str,
    *,
    boundary_id: str = "",
    way_id: str = "",
) -> CrossoverValidationIssue:
    return CrossoverValidationIssue(severity, code, message, boundary_id, way_id)


def _filter_issue(spec: CrossoverFilterSpec, *, boundary_id: str, way_id: str) -> CrossoverValidationIssue | None:
    try:
        spec.validate()
    except ValueError as exc:
        return _issue("error", "invalid_filter", str(exc), boundary_id=boundary_id, way_id=way_id)
    return None


def _boundary_pair_issues(boundary: CrossoverBoundary) -> list[CrossoverValidationIssue]:
    if boundary.implementation != "dsp":
        return []
    lower = boundary.lower_low_pass
    upper = boundary.upper_high_pass
    issues: list[CrossoverValidationIssue] = []
    if lower.engine != upper.engine:
        issues.append(_issue(
            "warning",
            "mixed_engine",
            f"{boundary.name or boundary.id}: LPとHPで異なる方式を使用するため、電気的な補完性は保証されません。",
            boundary_id=boundary.id,
        ))
    elif lower.engine == "iir" and (lower.family != upper.family or lower.order != upper.order):
        issues.append(_issue(
            "warning",
            "mismatched_iir",
            f"{boundary.name or boundary.id}: IIR FamilyまたはOrderが一致していません。",
            boundary_id=boundary.id,
        ))
    elif lower.engine == "kaiser_fir" and (
        lower.cycles != upper.cycles or lower.attenuation_db != upper.attenuation_db
    ):
        issues.append(_issue(
            "warning",
            "mismatched_kaiser",
            f"{boundary.name or boundary.id}: Kaiser FIRのCyclesまたは推定減衰量が一致していません。",
            boundary_id=boundary.id,
        ))
    if (lower.engine == "off") != (upper.engine == "off"):
        issues.append(_issue(
            "warning",
            "one_sided_boundary",
            f"{boundary.name or boundary.id}: クロス境界の片側だけがOffです。",
            boundary_id=boundary.id,
        ))
    return issues


def validate_crossover_plan(
    plan: CrossoverPlan,
    *,
    sample_rates: Sequence[int] = (),
    target_taps_by_way: Mapping[str, int] | None = None,
) -> tuple[CrossoverValidationIssue, ...]:
    issues: list[CrossoverValidationIssue] = []
    main_ids = tuple(plan.main_way_ids)
    if not main_ids:
        issues.append(_issue("error", "missing_main_way", "少なくとも1つのMain Wayが必要です。"))
    if len(main_ids) != len(set(main_ids)):
        issues.append(_issue("error", "duplicate_main_way", "Main Way IDが重複しています。"))
    groups = plan.effective_main_way_groups
    grouped_ids = tuple(way_id for group in groups for way_id in group)
    if grouped_ids != main_ids:
        issues.append(_issue(
            "error",
            "invalid_main_way_groups",
            "Main Way groupの並びがMain Way IDと一致していません。",
        ))
    expected_pairs = tuple(
        (group[index], group[index + 1])
        for group in groups
        for index in range(len(group) - 1)
    )
    expected_boundaries = len(expected_pairs)
    if len(plan.boundaries) != expected_boundaries:
        issues.append(_issue(
            "error",
            "boundary_count",
            "クロス境界数はMain Way数-1である必要があります。",
        ))
    boundary_ids = [boundary.id for boundary in plan.boundaries]
    if len(boundary_ids) != len(set(boundary_ids)):
        issues.append(_issue("error", "duplicate_boundary", "Crossover boundary IDが重複しています。"))

    previous_frequency_by_group: dict[int, float] = {}
    for index, boundary in enumerate(plan.boundaries):
        try:
            boundary.validate()
        except ValueError as exc:
            issues.append(_issue("error", "invalid_boundary", str(exc), boundary_id=boundary.id))
        if boundary.frequency_hz - crossover_overlap_shift(plan, boundary) <= 0.0:
            issues.append(_issue(
                "error",
                "overlap_below_zero",
                f"{boundary.name or boundary.id}: Overlap後のHP cutoffは0 Hzより上が必要です。",
                boundary_id=boundary.id,
            ))
        if index < len(expected_pairs):
            expected_pair = expected_pairs[index]
            if (boundary.lower_way_id, boundary.upper_way_id) != expected_pair:
                issues.append(_issue(
                    "error",
                    "boundary_adjacency",
                    f"{boundary.name or boundary.id}: Main Wayの並びと接続先が一致していません。",
                    boundary_id=boundary.id,
                ))
        group_index = next(
            (
                item_index
                for item_index, group in enumerate(groups)
                if boundary.lower_way_id in group
            ),
            -1,
        )
        previous_frequency = previous_frequency_by_group.get(group_index, 0.0)
        if boundary.frequency_hz <= previous_frequency:
            issues.append(_issue(
                "error",
                "frequency_order",
                "クロス周波数はLow側からHigh側へ昇順に設定してください。",
                boundary_id=boundary.id,
            ))
        previous_frequency_by_group[group_index] = max(
            previous_frequency,
            float(boundary.frequency_hz),
        )
        if boundary.implementation == "dsp":
            for spec, way_id in (
                (boundary.lower_low_pass, boundary.lower_way_id),
                (boundary.upper_high_pass, boundary.upper_way_id),
            ):
                filter_issue = _filter_issue(spec, boundary_id=boundary.id, way_id=way_id)
                if filter_issue is not None:
                    issues.append(filter_issue)
        issues.extend(_boundary_pair_issues(boundary))

    sub_ids = [sub.way_id for sub in plan.sub_crossovers]
    if len(sub_ids) != len(set(sub_ids)):
        issues.append(_issue("error", "duplicate_sub_way", "Sub Way IDが重複しています。"))
    if set(sub_ids) & set(main_ids):
        issues.append(_issue("error", "sub_main_overlap", "同じWayをMainとSubの両方には設定できません。"))
    for sub in plan.sub_crossovers:
        try:
            sub.validate()
        except ValueError as exc:
            issues.append(_issue("error", "invalid_sub", str(exc), way_id=sub.way_id))

    rates = tuple(int(rate) for rate in sample_rates)
    if any(rate <= 0 for rate in rates):
        issues.append(_issue("error", "invalid_sample_rate", "Sample Rateは正数である必要があります。"))
    for rate in rates:
        nyquist = rate / 2.0
        for boundary in plan.boundaries:
            if boundary.frequency_hz + crossover_overlap_shift(plan, boundary) >= nyquist:
                issues.append(_issue(
                    "error",
                    "frequency_above_nyquist",
                    f"{boundary.name or boundary.id}: {rate:g} HzのNyquist未満に設定してください。",
                    boundary_id=boundary.id,
                ))
        for sub in plan.sub_crossovers:
            if sub.frequency_hz >= nyquist:
                issues.append(_issue(
                    "error",
                    "sub_above_nyquist",
                    f"Sub {sub.way_id}: {rate:g} HzのNyquist未満に設定してください。",
                    way_id=sub.way_id,
                ))

    tap_limits = target_taps_by_way or {}
    for way_id, taps in tap_limits.items():
        if int(taps) < 1:
            issues.append(_issue("error", "invalid_target_taps", "Device tapsは1以上が必要です。", way_id=way_id))
    if rates and tap_limits:
        for rate in rates:
            for boundary in plan.boundaries:
                if boundary.implementation != "dsp":
                    continue
                for spec, way_id in (
                    (boundary.lower_low_pass, boundary.lower_way_id),
                    (boundary.upper_high_pass, boundary.upper_way_id),
                ):
                    limit = tap_limits.get(way_id)
                    if spec.engine != "kaiser_fir" or limit is None:
                        continue
                    required = kaiser_crossover_taps(rate, boundary.frequency_hz, spec.cycles)
                    if required > int(limit):
                        issues.append(_issue(
                            "warning",
                            "fir_taps_exceed_device",
                            f"{way_id}: Kaiser FIR {required} tapsをDevice {int(limit)} tapsへ中央cropします。",
                            boundary_id=boundary.id,
                            way_id=way_id,
                        ))
            for sub in plan.sub_crossovers:
                limit = tap_limits.get(sub.way_id)
                if sub.low_pass.engine != "kaiser_fir" or limit is None:
                    continue
                required = kaiser_crossover_taps(rate, sub.frequency_hz, sub.low_pass.cycles)
                if required > int(limit):
                    issues.append(_issue(
                        "warning",
                        "fir_taps_exceed_device",
                        f"{sub.way_id}: Kaiser FIR {required} tapsをDevice {int(limit)} tapsへ中央cropします。",
                        way_id=sub.way_id,
                    ))
    return tuple(issues)


def raise_for_crossover_errors(issues: Sequence[CrossoverValidationIssue]) -> None:
    errors = [issue.message for issue in issues if issue.severity == "error"]
    if errors:
        raise ValueError("; ".join(errors))
