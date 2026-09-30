from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
from octave_boundary_smoothing import (
    AutoPhaseBoundarySpec,
    AutoPhaseOuterBoundarySpec,
    IsolatedGainBoundarySpec,
    SharedBoundarySpec,
    apply_auto_gain_eq_boundaries,
    apply_auto_phase_eq_boundaries,
)

from .config import AutoEQSection


def auto_eq_range_label(f_min: float, f_max: float, *, nyquist_label: str = "Nyq") -> str:
    start = float(f_min)
    end = float(f_max)
    end_text = nyquist_label if end == 0.0 else f"{end:.0f}"
    return f"{start:.0f}-{end_text} Hz"


def auto_eq_card_title(
    page_label: str,
    index: int,
    *,
    f_min: float,
    f_max: float,
    enabled: bool = True,
    nyquist_label: str = "Nyq",
) -> str:
    label = f"{page_label}{int(index) + 1} | {auto_eq_range_label(f_min, f_max, nyquist_label=nyquist_label)}"
    return label if enabled else f"{label} / OFF"


def auto_eq_summary_range(f_min: float, f_max: float) -> str:
    return auto_eq_range_label(f_min, f_max, nyquist_label="Nyquist")


def smootherstep_weight(t: np.ndarray | float) -> np.ndarray:
    values = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return values * values * values * (values * (values * 6.0 - 15.0) + 10.0)


def nearest_phase_branch_offset(anchor: float, reference: float) -> float:
    raw_turn = (float(anchor) - float(reference)) / 360.0
    base_turn = int(np.floor(raw_turn))
    candidates = []
    for turn in range(base_turn - 2, base_turn + 4):
        adjusted = float(anchor) - 360.0 * turn
        candidates.append((abs(adjusted - float(reference)), abs(adjusted), turn))
    _distance, _abs_adjusted, best_turn = min(candidates, key=lambda item: (item[0], item[1]))
    return 360.0 * best_turn


def nearest_equivalent_phase(value: float, reference: float) -> float:
    return float(value) - nearest_phase_branch_offset(float(value), float(reference))


def auto_gain_eq_sections(auto_eq: object) -> list[AutoEQSection]:
    gain_sections = getattr(auto_eq, "gain_sections", [])
    if gain_sections:
        return [
            replace(section, gain_enabled=True, phase_enabled=False)
            for section in gain_sections
            if section.enabled
        ]
    return [section for section in auto_eq_sections(auto_eq) if section.enabled and section.gain_enabled]


def auto_phase_eq_sections(auto_eq: object) -> list[AutoEQSection]:
    phase_sections = getattr(auto_eq, "phase_sections", [])
    if phase_sections:
        return [
            replace(section, gain_enabled=False, phase_enabled=True)
            for section in phase_sections
            if section.enabled
        ]
    return [section for section in auto_eq_sections(auto_eq) if section.enabled and section.phase_enabled]


def auto_eq_sections(auto_eq: object) -> list[AutoEQSection]:
    sections = getattr(auto_eq, "sections", [])
    if sections:
        return list(sections)
    return [
        AutoEQSection(
            enabled=bool(getattr(auto_eq, "gain_enabled", False) or getattr(auto_eq, "phase_enabled", False)),
            gain_enabled=bool(getattr(auto_eq, "gain_enabled", False)),
            phase_enabled=bool(getattr(auto_eq, "phase_enabled", False)),
            f_min=float(getattr(auto_eq, "f_min", 200.0)),
            f_max=float(getattr(auto_eq, "f_max", 10_000.0)),
            smoothing_fraction=float(getattr(auto_eq, "smoothing_fraction", 3.0)),
            phase_strength=1.0,
            gain_strength=float(getattr(auto_eq, "gain_strength", 1.0)),
            max_boost_db=float(getattr(auto_eq, "max_boost_db", 6.0)),
            max_cut_db=float(getattr(auto_eq, "max_cut_db", 12.0)),
            limit_mode=str(getattr(auto_eq, "limit_mode", "peak_dip")),
            trend_smoothing_oct=0.0,
            post_limit_smoothing_fraction=0.0,
            candidate_min_delta_db=float(getattr(auto_eq, "candidate_min_delta_db", 0.0)),
            candidate_min_wavelet_weight=float(getattr(auto_eq, "candidate_min_wavelet_weight", 0.0)),
        )
    ]


def auto_effective_range(section: AutoEQSection, frequency: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(frequency, dtype=float)
    finite = finite[np.isfinite(finite)]
    nyquist = float(np.max(finite)) if finite.size else max(float(section.f_max), float(section.f_min))
    f_min = max(float(section.f_min), 0.0)
    f_max = nyquist if float(section.f_max) == 0.0 else min(max(float(section.f_max), f_min), nyquist)
    if f_max <= f_min:
        f_max = min(nyquist, f_min + 1e-9)
    return f_min, f_max


def auto_eq_ranges_overlap(left: AutoEQSection, right: AutoEQSection, nyquist: float) -> bool:
    """Return True only when two enabled section interiors overlap."""
    left_end = float(nyquist) if float(left.f_max) == 0.0 else min(float(left.f_max), float(nyquist))
    right_end = float(nyquist) if float(right.f_max) == 0.0 else min(float(right.f_max), float(nyquist))
    return max(float(left.f_min), float(right.f_min)) < min(left_end, right_end)


def overlapping_auto_eq_section_pairs(
    sections: list[AutoEQSection],
    nyquist: float,
) -> list[tuple[int, int]]:
    enabled = [(index, section) for index, section in enumerate(sections) if section.enabled]
    return [
        (left_index, right_index)
        for position, (left_index, left) in enumerate(enabled)
        for right_index, right in enabled[position + 1 :]
        if auto_eq_ranges_overlap(left, right, nyquist)
    ]


def disable_overlapping_auto_eq_sections(
    sections: list[AutoEQSection],
    nyquist: float,
) -> tuple[list[AutoEQSection], set[int]]:
    """Disable every participant in an overlap graph, preserving all ranges."""
    conflict_indices = {
        index
        for pair in overlapping_auto_eq_section_pairs(sections, nyquist)
        for index in pair
    }
    return [
        replace(section, enabled=False) if index in conflict_indices else section
        for index, section in enumerate(sections)
    ], conflict_indices


def auto_boundary_oct(left: AutoEQSection, right: AutoEQSection | None = None) -> float:
    def smoothing_oct(section: AutoEQSection) -> float:
        if section.smoothing_fraction <= 0:
            return 1.0 / 24.0
        return 1.0 / float(section.smoothing_fraction)

    value = smoothing_oct(left) if right is None else 0.5 * (smoothing_oct(left) + smoothing_oct(right))
    return float(np.clip(value, 1.0 / 24.0, 1.0))


def auto_section_edge_taper_oct(section: AutoEQSection) -> float:
    if section.smoothing_fraction <= 0:
        return 0.0
    return float(np.clip(1.0 / float(section.smoothing_fraction), 1.0 / 24.0, 1.0))


def boundary_smootherstep_weight(freq: np.ndarray, f_start: float, f_end: float) -> np.ndarray:
    log_start = np.log2(max(float(f_start), 1e-9))
    log_end = np.log2(max(float(f_end), max(float(f_start), 1e-9) * (1.0 + 1e-9)))
    log_freq = np.log2(np.maximum(freq, 1e-9))
    t = np.clip((log_freq - log_start) / max(log_end - log_start, 1e-12), 0.0, 1.0)
    return smootherstep_weight(t)


def extend_section_correction_to_boundary(
    frequency: np.ndarray,
    correction: np.ndarray,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    freq = np.asarray(frequency, dtype=float)
    values = np.asarray(correction, dtype=float).copy()
    finite_section = np.isfinite(freq) & np.isfinite(values) & (freq >= float(f_min)) & (freq <= float(f_max))
    if not np.any(finite_section):
        return values
    section_freq = freq[finite_section]
    section_values = values[finite_section]
    lower = section_boundary_value(section_freq, section_values, f_min)
    upper = section_boundary_value(section_freq, section_values, f_max)
    values[freq < float(f_min)] = lower
    values[freq > float(f_max)] = upper
    return values


def taper_section_correction_inside_boundary(
    frequency: np.ndarray,
    correction: np.ndarray,
    f_min: float,
    f_max: float,
    boundary_oct: float,
    *,
    lower_to_zero: bool,
    upper_to_zero: bool,
    phase_boundary_nearest: bool = False,
) -> np.ndarray:
    freq = np.asarray(frequency, dtype=float)
    values = np.asarray(correction, dtype=float).copy()
    edge_width = float(boundary_oct)
    if edge_width <= 0 or f_max <= f_min:
        return values

    if lower_to_zero and f_min > 0:
        fade_end = min(float(f_max), float(f_min) * 2.0**edge_width)
        fade_mask = (freq >= float(f_min)) & (freq <= fade_end)
        if np.any(fade_mask) and fade_end > f_min:
            weight = boundary_smootherstep_weight(freq[fade_mask], f_min, fade_end)
            baseline = (
                nearest_equivalent_phase(0.0, section_boundary_value(freq, values, f_min))
                if phase_boundary_nearest
                else 0.0
            )
            values[fade_mask] = baseline + (values[fade_mask] - baseline) * weight

    if upper_to_zero and f_max > 0:
        fade_start = max(float(f_min), float(f_max) / 2.0**edge_width)
        fade_mask = (freq >= fade_start) & (freq <= float(f_max))
        if np.any(fade_mask) and f_max > fade_start:
            weight = boundary_smootherstep_weight(freq[fade_mask], fade_start, f_max)
            baseline = (
                nearest_equivalent_phase(0.0, section_boundary_value(freq, values, f_max))
                if phase_boundary_nearest
                else 0.0
            )
            values[fade_mask] = values[fade_mask] * (1.0 - weight) + baseline * weight

    return values


def combine_auto_eq_sections(
    frequency: np.ndarray,
    sections: list[AutoEQSection],
    corrections: list[np.ndarray],
    *,
    phase_boundary_nearest: bool = False,
    taper_edges: bool = True,
    edge_taper_strength: float | None = None,
    boundary_smoothing_method: str = "smooth_connect",
    correction_mode: Literal["gain", "phase"] = "gain",
) -> np.ndarray:
    if not sections or not corrections:
        return np.zeros_like(frequency, dtype=float)
    finite_frequency = np.asarray(frequency, dtype=float)
    finite_frequency = finite_frequency[np.isfinite(finite_frequency)]
    nyquist = float(np.max(finite_frequency)) if finite_frequency.size else 0.0
    overlap_pairs = overlapping_auto_eq_section_pairs(sections, nyquist)
    if overlap_pairs:
        labels = ", ".join(f"#{left + 1}/#{right + 1}" for left, right in overlap_pairs)
        raise ValueError(f"Auto EQ enabled section ranges overlap: {labels}")
    if boundary_smoothing_method != "smooth_connect":
        raise ValueError(f"Unsupported Auto EQ boundary smoothing method: {boundary_smoothing_method}")
    if correction_mode not in {"gain", "phase"}:
        raise ValueError(f"Unsupported Auto EQ correction mode: {correction_mode}")
    ordered = sorted(
        zip(sections, corrections),
        key=lambda pair: auto_effective_range(pair[0], frequency)[0],
    )
    freq = np.asarray(frequency, dtype=float)
    output = np.zeros_like(freq, dtype=float)
    ranges = [auto_effective_range(section, freq) for section, _ in ordered]
    corrections_ordered = [np.asarray(correction, dtype=float).copy() for _section, correction in ordered]
    taper_strength = 1.0 if edge_taper_strength is None else float(np.clip(edge_taper_strength, 0.0, 1.0))
    if not taper_edges:
        taper_strength = 0.0

    group_start = 0
    while group_start < len(ordered):
        group_end = group_start + 1
        while group_end < len(ordered) and ranges[group_end][0] <= ranges[group_end - 1][1]:
            group_end += 1

        group_sections = [section for section, _correction in ordered[group_start:group_end]]
        group_ranges = ranges[group_start:group_end]
        group_corrections = [
            extend_section_correction_to_boundary(freq, corrections_ordered[index], *ranges[index])
            for index in range(group_start, group_end)
        ]

        if phase_boundary_nearest and len(group_corrections) > 1:
            for local_index in range(1, len(group_corrections)):
                boundary = group_ranges[local_index][0]
                boundary_index = int(np.argmin(np.abs(freq - boundary)))
                branch_offset = nearest_phase_branch_offset(
                    group_corrections[local_index][boundary_index],
                    group_corrections[local_index - 1][boundary_index],
                )
                group_corrections[local_index] = group_corrections[local_index] - branch_offset

        applied_outer_transition = False
        if correction_mode == "phase":
            if len(group_corrections) == 1:
                group_correction = group_corrections[0]
            else:
                group_correction = np.zeros_like(freq, dtype=float)
                for correction, (f_min, f_max) in zip(group_corrections, group_ranges):
                    section_mask = (freq >= f_min) & (freq <= f_max)
                    group_correction[section_mask] = correction[section_mask]

            shared_phase_boundaries = [
                AutoPhaseBoundarySpec(
                    name=f"Auto Phase EQ shared boundary #{boundary_index + 1}",
                    fb_hz=group_ranges[boundary_index + 1][0],
                    delta_oct_left=auto_section_edge_taper_oct(group_sections[boundary_index]),
                    delta_oct_right=auto_section_edge_taper_oct(group_sections[boundary_index + 1]),
                    curve="smootherstep",
                    phase_turn_policy="preserve_unwrapped",
                )
                for boundary_index in range(len(group_sections) - 1)
            ]
            group_low, group_high = group_ranges[0][0], group_ranges[-1][1]
            edge_oct_values = [auto_section_edge_taper_oct(section) for section in group_sections]
            edge_oct = min((value for value in edge_oct_values if value > 0.0), default=0.0)
            outer_phase_boundaries: list[AutoPhaseOuterBoundarySpec] = []
            if taper_strength > 0.0 and edge_oct > 0.0:
                if group_low > 0.0 and np.any(freq < group_low):
                    outer_phase_boundaries.append(
                        AutoPhaseOuterBoundarySpec(
                            group_low,
                            edge_oct,
                            "lower",
                            name="Auto Phase EQ Low",
                            phase_turn_policy=("nearest_equivalent" if phase_boundary_nearest else "preserve_unwrapped"),
                        )
                    )
                if group_high < nyquist and np.any(freq > group_high):
                    outer_phase_boundaries.append(
                        AutoPhaseOuterBoundarySpec(
                            group_high,
                            edge_oct,
                            "upper",
                            name="Auto Phase EQ Hi",
                            phase_turn_policy=("nearest_equivalent" if phase_boundary_nearest else "preserve_unwrapped"),
                        )
                    )
            phase_result = apply_auto_phase_eq_boundaries(
                freq,
                group_correction,
                shared_phase_boundaries,
                outer_boundaries=outer_phase_boundaries,
                outer_transition_strength=taper_strength,
                input_phase_mode="unwrapped",
                output_phase_mode="unwrapped",
                overlap_policy="error",
            )
            group_correction = phase_result.phase_unwrapped_deg
            applied_outer_transition = any(item.applied for item in phase_result.outer_items)
        else:
            group_correction = np.zeros_like(freq, dtype=float)
            for correction, (f_min, f_max) in zip(group_corrections, group_ranges):
                section_mask = (freq >= f_min) & (freq <= f_max)
                group_correction[section_mask] = correction[section_mask]

            # Resolve and apply every shared boundary as one batch. Reference
            # gains are read from the original stitched curve, so processing
            # one boundary cannot change the inputs of a later boundary.
            shared_boundaries = [
                SharedBoundarySpec(
                    name=f"Auto Gain EQ shared boundary #{boundary_index + 1}",
                    fb_hz=group_ranges[boundary_index + 1][0],
                    delta_oct_left=auto_section_edge_taper_oct(group_sections[boundary_index]),
                    delta_oct_right=auto_section_edge_taper_oct(group_sections[boundary_index + 1]),
                    curve="smootherstep",
                )
                for boundary_index in range(len(group_sections) - 1)
            ]
            group_low, group_high = group_ranges[0][0], group_ranges[-1][1]
            low_oct = auto_section_edge_taper_oct(group_sections[0])
            high_oct = auto_section_edge_taper_oct(group_sections[-1])
            isolated_boundaries: list[IsolatedGainBoundarySpec] = []
            if low_oct > 0.0 and group_low > 0.0:
                isolated_boundaries.append(
                    IsolatedGainBoundarySpec(
                        group_low,
                        low_oct,
                        "lower",
                        name="Auto Gain EQ Low",
                    )
                )
            if high_oct > 0.0 and group_high < nyquist:
                isolated_boundaries.append(
                    IsolatedGainBoundarySpec(
                        group_high,
                        high_oct,
                        "upper",
                        name="Auto Gain EQ Hi",
                    )
                )
            group_correction = apply_auto_gain_eq_boundaries(
                freq,
                group_correction,
                shared_boundaries,
                isolated_boundaries,
                overlap_policy="error",
            ).gain_db

        group_low = group_ranges[0][0]
        group_high = group_ranges[-1][1]
        group_mask = (freq >= group_low) & (freq <= group_high)
        if not applied_outer_transition:
            group_correction = np.where(group_mask, group_correction, 0.0)

        output += group_correction
        group_start = group_end

    return output


def section_boundary_value(freq: np.ndarray, values: np.ndarray, boundary_freq: float) -> float:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(freq) & np.isfinite(values)
    if not np.any(finite):
        return 0.0
    idx = int(np.argmin(np.abs(freq[finite] - float(boundary_freq))))
    return float(values[finite][idx])


def apply_auto_edge_smooth(
    correction: np.ndarray,
    frequency: np.ndarray,
    f_min: float,
    f_max: float,
    edge_smooth_percent: float,
) -> np.ndarray:
    smooth = float(edge_smooth_percent) / 100.0
    if smooth <= 0:
        return correction
    span = max(float(f_max) - float(f_min), 0.0)
    fade_width = span * min(smooth, 0.5)
    if fade_width <= 0:
        return correction

    weight = np.ones_like(correction, dtype=float)
    start_mask = (frequency >= f_min) & (frequency < f_min + fade_width)
    if np.any(start_mask):
        t = (frequency[start_mask] - f_min) / fade_width
        weight[start_mask] = smootherstep_weight(t)
    end_mask = (frequency > f_max - fade_width) & (frequency <= f_max)
    if np.any(end_mask):
        t = (f_max - frequency[end_mask]) / fade_width
        weight[end_mask] = smootherstep_weight(t)
    return correction * weight


def apply_auto_edge_smooth_octave(
    correction: np.ndarray,
    frequency: np.ndarray,
    f_min: float,
    f_max: float,
    edge_smooth_oct: float,
) -> np.ndarray:
    edge_width = float(edge_smooth_oct)
    if edge_width <= 0:
        return correction

    f_min = max(float(f_min), 1e-9)
    f_max = max(float(f_max), f_min)
    if f_max <= f_min:
        return correction

    freq = np.maximum(np.asarray(frequency, dtype=float), 1e-9)
    weight = np.ones_like(correction, dtype=float)
    has_lower_outside = bool(np.any(freq < f_min))
    has_upper_outside = bool(np.any(freq > f_max))

    if has_lower_outside:
        start_distance_oct = np.log2(freq / f_min)
        start_mask = (freq >= f_min) & (start_distance_oct < edge_width)
        if np.any(start_mask):
            t = np.clip(start_distance_oct[start_mask] / edge_width, 0.0, 1.0)
            weight[start_mask] = smootherstep_weight(t)

    if has_upper_outside:
        end_distance_oct = np.log2(f_max / freq)
        end_mask = (freq <= f_max) & (end_distance_oct < edge_width)
        if np.any(end_mask):
            t = np.clip(end_distance_oct[end_mask] / edge_width, 0.0, 1.0)
            weight[end_mask] = np.minimum(weight[end_mask], smootherstep_weight(t))

    return correction * weight
