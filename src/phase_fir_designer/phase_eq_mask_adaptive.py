from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from octave_boundary_smoothing import OneSidedBoundarySpec, apply_one_sided_boundary


@dataclass(frozen=True)
class PhaseMaskBParameters:
    base_slope_deg_per_oct: float
    max_slope_deg_per_oct: float
    landing_hz: float
    reference_oct: float

    @property
    def key(self) -> str:
        return (
            f"base={self.base_slope_deg_per_oct:g},max={self.max_slope_deg_per_oct:g},"
            f"landing={self.landing_hz:g},ref={self.reference_oct:.8g}"
        )


@dataclass(frozen=True)
class PhaseMaskBResult:
    phase_deg: np.ndarray
    neutral_phase_deg: float
    residual_phase_deg: float
    fade_slope_deg_per_oct: float
    method: str
    width_scale: float
    confidence: float
    runtime_ms: float


@dataclass(frozen=True)
class PhaseMaskSafetyMetrics:
    group_delay_peak_ms: float
    group_delay_roughness_ms: float
    group_delay_max_ms: float


@dataclass(frozen=True)
class AdaptivePhaseMaskResult:
    phase_deg: np.ndarray
    selected_max_slope_deg_per_oct: float
    required_slope_deg_per_oct: float
    compared_candidates: bool
    reason: str
    conservative: PhaseMaskBResult
    aggressive: PhaseMaskBResult | None
    conservative_metrics: PhaseMaskSafetyMetrics | None
    aggressive_metrics: PhaseMaskSafetyMetrics | None


def closest_zero_equivalent_phase(phase_deg: float) -> float:
    """Return the nearest 360-degree multiple, with ties away from zero."""

    turns = abs(float(phase_deg)) / 360.0
    rounded_turns = np.floor(turns + 0.5)
    return float(np.copysign(360.0 * rounded_turns, float(phase_deg)))


def phase_mask_required_slope(
    frequency: np.ndarray,
    phase_deg: np.ndarray,
    *,
    boundary_hz: float,
    landing_hz: float = 20.0,
) -> float:
    freq = np.asarray(frequency, dtype=float)
    source = _unwrap_phase(phase_deg)
    hold = float(np.interp(float(boundary_hz), freq, source))
    neutral = closest_zero_equivalent_phase(hold)
    residual = hold - neutral
    landing = _effective_landing_hz(freq, float(boundary_hz), float(landing_hz))
    available_oct = max(float(np.log2(float(boundary_hz) / landing)), 0.25)
    return abs(residual) / available_oct


def apply_phase_eq_mask_b_low(
    frequency: np.ndarray,
    phase_deg: np.ndarray,
    *,
    boundary_hz: float,
    smoothing_oct: float,
    parameters: PhaseMaskBParameters,
    upper_limit_hz: float | None = None,
) -> PhaseMaskBResult:
    """Build one branch-aware, adaptive-slope Phase EQ Mask Lo candidate."""

    started = perf_counter()
    freq = np.asarray(frequency, dtype=float)
    source = _unwrap_phase(phase_deg)
    boundary = float(boundary_hz)
    hold = float(np.interp(boundary, freq, source))
    neutral = closest_zero_equivalent_phase(hold)
    residual = hold - neutral
    landing = _effective_landing_hz(freq, boundary, float(parameters.landing_hz))
    available_oct = max(float(np.log2(boundary / landing)), 0.25)
    required_slope = abs(residual) / available_oct
    slope = min(
        max(float(parameters.base_slope_deg_per_oct), required_slope),
        float(parameters.max_slope_deg_per_oct),
    )
    if np.isclose(residual, 0.0, atol=1.0e-12):
        slope = 0.0

    output = source.copy()
    outside = freq < boundary
    if np.any(outside):
        output[outside] = _fade_residual_to_neutral(
            freq[outside],
            boundary_hz=boundary,
            neutral_phase_deg=neutral,
            residual_phase_deg=residual,
            slope_deg_per_oct=slope,
        )

    if smoothing_oct <= 0.0:
        return PhaseMaskBResult(
            output,
            neutral,
            residual,
            slope,
            "no_inner_smoothing",
            1.0,
            1.0,
            (perf_counter() - started) * 1000.0,
        )

    outer_slope = 0.0
    if slope > 0.0 and not np.isclose(residual, 0.0, atol=1.0e-12):
        outer_slope = np.copysign(slope, residual) / max(boundary * np.log(2.0), 1.0e-12)
    connection = apply_one_sided_boundary(
        freq,
        source,
        OneSidedBoundarySpec(
            fb_hz=boundary,
            delta_oct=float(smoothing_oct),
            side="lower",
            axis="linear",
            outer_value=hold,
            outer_slope=outer_slope,
            reference_oct=float(parameters.reference_oct),
            upper_limit_hz=upper_limit_hz,
            name="Phase EQ mask adaptive Lo",
        ),
    )
    item = connection.item
    if item.applied and item.f_start_hz is not None and item.f_end_hz is not None:
        transition = (freq >= float(item.f_start_hz)) & (freq <= float(item.f_end_hz))
        output[transition] = connection.values[transition]
    return PhaseMaskBResult(
        output,
        neutral,
        residual,
        slope,
        item.method,
        float(item.width_scale),
        float(item.inner_confidence),
        (perf_counter() - started) * 1000.0,
    )


def apply_adaptive_phase_eq_mask_low(
    frequency: np.ndarray,
    phase_deg: np.ndarray,
    *,
    boundary_hz: float,
    smoothing_oct: float,
    upper_limit_hz: float | None = None,
    base_slope_deg_per_oct: float = 10.0,
    conservative_max_slope_deg_per_oct: float = 30.0,
    aggressive_max_slope_deg_per_oct: float = 60.0,
    landing_hz: float = 20.0,
    reference_oct: float = 1.0 / 24.0,
) -> AdaptivePhaseMaskResult:
    """Select 60 deg/oct only when it is as smooth as the 30 deg/oct candidate."""

    freq = np.asarray(frequency, dtype=float)
    source = _unwrap_phase(phase_deg)
    required_slope = phase_mask_required_slope(
        freq,
        source,
        boundary_hz=boundary_hz,
        landing_hz=landing_hz,
    )
    conservative_parameters = PhaseMaskBParameters(
        base_slope_deg_per_oct,
        conservative_max_slope_deg_per_oct,
        landing_hz,
        reference_oct,
    )
    conservative = apply_phase_eq_mask_b_low(
        freq,
        source,
        boundary_hz=boundary_hz,
        smoothing_oct=smoothing_oct,
        parameters=conservative_parameters,
        upper_limit_hz=upper_limit_hz,
    )
    if required_slope <= conservative_max_slope_deg_per_oct + 1.0e-12:
        return AdaptivePhaseMaskResult(
            conservative.phase_deg,
            conservative_max_slope_deg_per_oct,
            required_slope,
            False,
            "required_slope_within_conservative_limit",
            conservative,
            None,
            None,
            None,
        )

    aggressive_parameters = PhaseMaskBParameters(
        base_slope_deg_per_oct,
        aggressive_max_slope_deg_per_oct,
        landing_hz,
        reference_oct,
    )
    aggressive = apply_phase_eq_mask_b_low(
        freq,
        source,
        boundary_hz=boundary_hz,
        smoothing_oct=smoothing_oct,
        parameters=aggressive_parameters,
        upper_limit_hz=upper_limit_hz,
    )
    conservative_metrics = _phase_safety_metrics(
        freq,
        conservative.phase_deg,
        boundary_hz=float(boundary_hz),
        smoothing_oct=float(smoothing_oct),
        landing_hz=float(landing_hz),
        upper_limit_hz=upper_limit_hz,
    )
    aggressive_metrics = _phase_safety_metrics(
        freq,
        aggressive.phase_deg,
        boundary_hz=float(boundary_hz),
        smoothing_oct=float(smoothing_oct),
        landing_hz=float(landing_hz),
        upper_limit_hz=upper_limit_hz,
    )
    if conservative_metrics is None or aggressive_metrics is None:
        safe = False
        reason = "insufficient_group_delay_samples"
    else:
        peak_tolerance = max(0.05, 0.10 * conservative_metrics.group_delay_peak_ms)
        roughness_tolerance = max(
            0.02,
            0.15 * conservative_metrics.group_delay_roughness_ms,
        )
        spike_tolerance = max(0.10, 0.25 * conservative_metrics.group_delay_max_ms)
        peak_ok = (
            aggressive_metrics.group_delay_peak_ms
            <= conservative_metrics.group_delay_peak_ms + peak_tolerance
        )
        roughness_ok = (
            aggressive_metrics.group_delay_roughness_ms
            <= conservative_metrics.group_delay_roughness_ms + roughness_tolerance
        )
        no_new_spike = (
            aggressive_metrics.group_delay_max_ms
            <= conservative_metrics.group_delay_max_ms + spike_tolerance
        )
        safe = bool(peak_ok and roughness_ok and no_new_spike)
        reason = "aggressive_candidate_within_safety_thresholds" if safe else "group_delay_guard"
    selected = aggressive if safe else conservative
    return AdaptivePhaseMaskResult(
        selected.phase_deg,
        (
            aggressive_max_slope_deg_per_oct
            if safe
            else conservative_max_slope_deg_per_oct
        ),
        required_slope,
        True,
        reason,
        conservative,
        aggressive,
        conservative_metrics,
        aggressive_metrics,
    )


def _phase_safety_metrics(
    frequency: np.ndarray,
    phase_deg: np.ndarray,
    *,
    boundary_hz: float,
    smoothing_oct: float,
    landing_hz: float,
    upper_limit_hz: float | None,
) -> PhaseMaskSafetyMetrics | None:
    lower = min(_effective_landing_hz(frequency, boundary_hz, landing_hz), boundary_hz / 2.0)
    upper = boundary_hz * (2.0 ** max(2.0 * smoothing_oct, 0.5))
    if upper_limit_hz is not None:
        upper = min(upper, float(upper_limit_hz))
    local = (frequency > 0.0) & (frequency >= lower) & (frequency <= upper)
    local_frequency = np.asarray(frequency[local], dtype=float)
    local_phase = np.asarray(phase_deg[local], dtype=float)
    if local_frequency.size < 4 or not np.all(np.isfinite(local_phase)):
        return None
    group_delay_ms = -np.gradient(local_phase, local_frequency, edge_order=1) / 360.0 * 1000.0
    if not np.all(np.isfinite(group_delay_ms)):
        return None
    return PhaseMaskSafetyMetrics(
        float(np.percentile(np.abs(group_delay_ms), 99.0)),
        float(np.percentile(np.abs(np.diff(group_delay_ms)), 95.0)),
        float(np.max(np.abs(group_delay_ms))),
    )


def _effective_landing_hz(frequency: np.ndarray, boundary_hz: float, landing_hz: float) -> float:
    freq = np.asarray(frequency, dtype=float)
    positive = freq[np.isfinite(freq) & (freq > 0.0)]
    first_positive = float(positive[0]) if positive.size else 1.0
    return min(max(float(landing_hz), first_positive), float(boundary_hz) / (2.0**0.25))


def _fade_residual_to_neutral(
    frequency: np.ndarray,
    *,
    boundary_hz: float,
    neutral_phase_deg: float,
    residual_phase_deg: float,
    slope_deg_per_oct: float,
) -> np.ndarray:
    freq = np.maximum(np.asarray(frequency, dtype=float), 1.0e-12)
    residual = float(residual_phase_deg)
    slope = abs(float(slope_deg_per_oct))
    if slope <= 0.0 or np.isclose(residual, 0.0, atol=1.0e-12):
        return np.full_like(freq, float(neutral_phase_deg) + residual)
    distance_oct = np.maximum(np.log2(float(boundary_hz) / freq), 0.0)
    total_oct = abs(residual) / slope
    sign = 1.0 if residual >= 0.0 else -1.0
    magnitude = np.maximum(abs(residual) - slope * distance_oct, 0.0)
    edge_oct = min(1.0 / 6.0, total_oct * 0.5)
    if edge_oct > 0.0:
        start = max(total_oct - edge_oct, 0.0)
        edge = (distance_oct > start) & (distance_oct < total_oct)
        if np.any(edge):
            t = (distance_oct[edge] - start) / edge_oct
            magnitude[edge] = _cubic_hermite_edge_to_zero(t, slope, edge_oct)
    return float(neutral_phase_deg) + sign * magnitude


def _cubic_hermite_edge_to_zero(t: np.ndarray, slope: float, width_oct: float) -> np.ndarray:
    values = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    y0 = float(slope) * float(width_oct)
    m0 = -float(slope)
    h00 = 2.0 * values**3 - 3.0 * values**2 + 1.0
    h10 = values**3 - 2.0 * values**2 + values
    return h00 * y0 + h10 * float(width_oct) * m0


def _unwrap_phase(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    output = source.copy()
    finite = np.isfinite(output)
    if np.count_nonzero(finite) >= 2:
        output[finite] = np.rad2deg(np.unwrap(np.deg2rad(output[finite])))
    return output
