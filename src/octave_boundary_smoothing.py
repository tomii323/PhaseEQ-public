"""Standalone octave-domain smoothing for isolated and shared EQ boundaries.

The functions in this module only depend on NumPy and do not use Streamlit or
application settings.  The module can therefore be imported as a small DSP
library or executed directly to run its built-in demonstration::

    python -m octave_boundary_smoothing
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from scipy.ndimage import median_filter
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
Curve = Literal["smootherstep"]
OverlapPolicy = Literal["error", "skip_later", "clip_midpoint"]
InputPhaseMode = Literal["wrapped", "unwrapped"]
OutputPhaseMode = Literal["wrapped", "unwrapped"]
PhaseTurnPolicy = Literal["preserve_unwrapped", "nearest_equivalent"]
IsolatedBoundarySide = Literal["lower", "upper"]
BoundaryAxis = Literal["log", "linear"]


@dataclass(frozen=True)
class SharedBoundarySpec:
    """Configuration for one shared boundary."""

    fb_hz: float
    delta_oct_left: float
    delta_oct_right: float
    curve: Curve = "smootherstep"
    enabled: bool = True
    name: str = ""


@dataclass(frozen=True)
class ResolvedSharedBoundary:
    """A shared boundary after validation and overlap resolution."""

    index: int
    name: str
    fb_hz: float
    f_start_hz: float
    f_end_hz: float
    curve: Curve
    clipped_left: bool = False
    clipped_right: bool = False


@dataclass(frozen=True)
class SharedBoundaryBatchItem:
    """Diagnostic result for one shared boundary."""

    index: int
    name: str
    fb_hz: float
    applied: bool
    reason: str
    f_start_hz: float | None
    f_end_hz: float | None
    processed_points: int
    max_change_db: float
    clipped_left: bool = False
    clipped_right: bool = False
    method: str = "none"
    left_slope: float | None = None
    right_slope: float | None = None
    left_confidence: float = 0.0
    right_confidence: float = 0.0
    width_scale: float = 1.0
    slope_relaxation: float = 0.0


@dataclass(frozen=True)
class SharedBoundaryBatchResult:
    """Result and diagnostics from shared-boundary batch processing."""

    gain_db: FloatArray
    items: tuple[SharedBoundaryBatchItem, ...]
    applied_count: int
    skipped_count: int
    total_max_change_db: float


@dataclass(frozen=True)
class IsolatedGainBoundarySpec:
    """Configuration for one finite outer Auto Gain EQ boundary."""

    fb_hz: float
    delta_oct: float
    side: IsolatedBoundarySide
    strength: float = 1.0
    enabled: bool = True
    name: str = ""


@dataclass(frozen=True)
class IsolatedGainBoundaryItem:
    """Diagnostic result for one isolated Auto Gain EQ boundary."""

    index: int
    name: str
    fb_hz: float
    side: IsolatedBoundarySide
    applied: bool
    reason: str
    f_start_hz: float | None
    f_end_hz: float | None
    processed_points: int
    max_change_db: float
    method: str = "none"
    inner_value_db: float | None = None
    outer_value_db: float | None = None
    inner_slope_db_per_oct: float | None = None
    outer_slope_db_per_oct: float | None = None
    inner_confidence: float = 0.0
    width_scale: float = 1.0
    slope_relaxation: float = 0.0


@dataclass(frozen=True)
class AutoGainBoundaryBatchResult:
    """Complete Auto Gain EQ boundary result."""

    gain_db: FloatArray
    shared_items: tuple[SharedBoundaryBatchItem, ...]
    isolated_items: tuple[IsolatedGainBoundaryItem, ...]
    applied_count: int
    skipped_count: int
    total_max_change_db: float


@dataclass(frozen=True)
class AutoPhaseBoundarySpec:
    """Configuration for one shared Auto Phase EQ boundary."""

    fb_hz: float
    delta_oct_left: float
    delta_oct_right: float
    curve: Curve = "smootherstep"
    enabled: bool = True
    name: str = ""
    reference_oct: float = 1.0 / 24.0
    phase_turn_policy: PhaseTurnPolicy = "preserve_unwrapped"
    max_phase_change_deg: float | None = None


@dataclass(frozen=True)
class AutoPhaseOuterBoundarySpec:
    """Configuration for one finite Auto Phase EQ outer boundary."""

    fb_hz: float
    delta_oct: float
    side: IsolatedBoundarySide
    enabled: bool = True
    name: str = ""
    phase_turn_policy: PhaseTurnPolicy = "nearest_equivalent"


@dataclass(frozen=True)
class ResolvedAutoPhaseBoundary:
    """An Auto Phase EQ boundary after validation and overlap resolution."""

    index: int
    name: str
    fb_hz: float
    f_start_hz: float
    f_end_hz: float
    curve: Curve
    reference_oct: float
    phase_turn_policy: PhaseTurnPolicy
    max_phase_change_deg: float | None
    clipped_left: bool = False
    clipped_right: bool = False


@dataclass(frozen=True)
class AutoPhaseBoundaryItem:
    """Diagnostic result for one shared Auto Phase EQ boundary."""

    index: int
    name: str
    fb_hz: float
    applied: bool
    reason: str
    f_start_hz: float | None
    f_end_hz: float | None
    processed_points: int
    left_reference_deg: float | None
    right_reference_deg: float | None
    max_phase_change_deg: float
    max_group_delay_change_ms: float
    clipped_left: bool = False
    clipped_right: bool = False
    method: str = "none"
    left_slope_deg_per_hz: float | None = None
    right_slope_deg_per_hz: float | None = None
    left_confidence: float = 0.0
    right_confidence: float = 0.0
    width_scale: float = 1.0
    slope_relaxation: float = 0.0


@dataclass(frozen=True)
class AutoPhaseOuterBoundaryItem:
    """Diagnostic result for one finite Auto Phase EQ outer boundary."""

    index: int
    name: str
    fb_hz: float
    side: IsolatedBoundarySide
    applied: bool
    reason: str
    processed_points: int
    base_phase_deg: float | None
    max_phase_change_deg: float
    max_group_delay_change_ms: float
    method: str = "none"
    inner_phase_deg: float | None = None
    inner_slope_deg_per_hz: float | None = None
    outer_slope_deg_per_hz: float | None = None
    inner_confidence: float = 0.0
    width_scale: float = 1.0
    slope_relaxation: float = 0.0


@dataclass(frozen=True)
class OneSidedBoundarySpec:
    """Slope-aware connection between one input curve and an outer model."""

    fb_hz: float
    delta_oct: float
    side: IsolatedBoundarySide
    axis: BoundaryAxis
    outer_value: float
    outer_slope: float = 0.0
    reference_oct: float = 1.0 / 6.0
    lower_limit_hz: float | None = None
    upper_limit_hz: float | None = None
    enabled: bool = True
    name: str = ""


@dataclass(frozen=True)
class OneSidedBoundaryItem:
    """Diagnostics for a slope-aware one-sided boundary connection."""

    name: str
    fb_hz: float
    side: IsolatedBoundarySide
    axis: BoundaryAxis
    applied: bool
    reason: str
    f_start_hz: float | None
    f_end_hz: float | None
    processed_points: int
    max_change: float
    method: str = "none"
    inner_value: float | None = None
    outer_value: float | None = None
    inner_slope: float | None = None
    outer_slope: float | None = None
    inner_confidence: float = 0.0
    width_scale: float = 1.0
    slope_relaxation: float = 0.0


@dataclass(frozen=True)
class OneSidedBoundaryResult:
    """Processed values and diagnostics for one one-sided boundary."""

    values: FloatArray
    item: OneSidedBoundaryItem


@dataclass(frozen=True)
class AutoPhaseBatchResult:
    """Processed phase curves and diagnostics for a boundary batch."""

    phase_deg: FloatArray
    phase_unwrapped_deg: FloatArray
    items: tuple[AutoPhaseBoundaryItem, ...]
    outer_items: tuple[AutoPhaseOuterBoundaryItem, ...]
    applied_count: int
    skipped_count: int
    total_max_phase_change_deg: float
    total_max_group_delay_change_ms: float


def _smootherstep(x: FloatArray) -> FloatArray:
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (x * (x * 6.0 - 15.0) + 10.0)


@dataclass(frozen=True)
class _RobustFit:
    slope: float
    confidence: float
    residual_scale: float


@dataclass(frozen=True)
class _ConnectionCandidate:
    values: FloatArray
    method: str
    left_value: float
    right_value: float
    left_slope: float
    right_slope: float
    left_confidence: float
    right_confidence: float
    width_scale: float
    slope_relaxation: float
    f_start_hz: float
    f_end_hz: float
    indices: NDArray[np.int64]


def _median_filter_1d(values: FloatArray, kernel_size: int = 5) -> FloatArray:
    """Median filter used only for slope estimation.

    ``mode="nearest"`` is equivalent to the previous edge-padded sliding
    implementation, while keeping the complete operation in compiled SciPy
    code for large Analysis FFT grids.
    """

    if values.size < 3:
        return values.copy()
    size = min(int(kernel_size), int(values.size) if values.size % 2 else int(values.size) - 1)
    size = max(size, 3)
    return np.asarray(
        median_filter(np.asarray(values, dtype=np.float64), size=size, mode="nearest"),
        dtype=np.float64,
    )


def _robust_linear_fit(x: FloatArray, y: FloatArray) -> _RobustFit:
    """Estimate a local slope and a 0..1 confidence without smoothing output data."""

    if x.size < 2 or y.size != x.size:
        return _RobustFit(0.0, 0.0, float("inf"))
    filtered = _median_filter_1d(np.asarray(y, dtype=np.float64))
    dx = np.diff(x)
    valid = np.isfinite(dx) & (np.abs(dx) > np.finfo(np.float64).eps)
    if not np.any(valid):
        return _RobustFit(0.0, 0.0, float("inf"))
    x_center = float(np.median(x))
    centered_x = x - x_center
    slope = float(np.median(np.diff(filtered)[valid] / dx[valid]))
    intercept = float(np.median(filtered - slope * centered_x))
    design = np.column_stack((centered_x, np.ones_like(centered_x)))
    weights = np.ones_like(x)
    for _iteration in range(6):
        weighted = design * np.sqrt(weights)[:, None]
        target = filtered * np.sqrt(weights)
        estimate, *_unused = np.linalg.lstsq(weighted, target, rcond=None)
        slope, intercept = float(estimate[0]), float(estimate[1])
        residual = filtered - (slope * centered_x + intercept)
        center = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - center)))
        if scale <= np.finfo(np.float64).eps:
            break
        normalized = np.abs(residual - center) / (1.345 * scale)
        weights = np.where(normalized <= 1.0, 1.0, 1.0 / np.maximum(normalized, 1.0))
    # Confidence is evaluated against the unfiltered reference.  The median
    # copy stabilizes the estimate, but must not hide poor source quality.
    residual = np.asarray(y, dtype=np.float64) - (slope * centered_x + intercept)
    center = float(np.median(residual))
    residual_scale = 1.4826 * float(np.median(np.abs(residual - center)))
    span = max(float(x[-1] - x[0]), np.finfo(np.float64).eps)
    trend_scale = max(abs(slope) * span, float(np.percentile(filtered, 90) - np.percentile(filtered, 10)), 1.0e-9)
    residual_confidence = 1.0 / (1.0 + residual_scale / trend_scale)
    midpoint = max(2, x.size // 2)
    split_slopes: list[float] = []
    for part_x, part_y in ((x[:midpoint], filtered[:midpoint]), (x[midpoint - 1 :], filtered[midpoint - 1 :])):
        if part_x.size >= 2 and part_x[-1] > part_x[0]:
            split_slopes.append(float(np.polyfit(part_x, part_y, 1)[0]))
    slope_spread = abs(split_slopes[0] - split_slopes[1]) if len(split_slopes) == 2 else 0.0
    stability_scale = max(abs(slope), trend_scale / span, 1.0e-12)
    stability_confidence = 1.0 / (1.0 + slope_spread / stability_scale)
    coverage_confidence = min(1.0, x.size / 8.0)
    if residual_scale <= np.finfo(np.float64).eps:
        inlier_confidence = 1.0
    else:
        inlier_confidence = float(np.mean(np.abs(residual - center) <= 3.0 * residual_scale))
    confidence = float(
        np.clip(
            coverage_confidence
            * np.sqrt(residual_confidence * stability_confidence)
            * (0.5 + 0.5 * inlier_confidence),
            0.0,
            1.0,
        )
    )
    return _RobustFit(slope, confidence, residual_scale)


def _cubic_hermite(
    x: FloatArray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    m0: float,
    m1: float,
) -> FloatArray:
    width = float(x1 - x0)
    t = np.clip((x - x0) / width, 0.0, 1.0)
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    return h00 * y0 + h10 * width * m0 + h01 * y1 + h11 * width * m1


def _monotone_slopes(y0: float, y1: float, m0: float, m1: float, width: float) -> tuple[float, float]:
    """Fritsch-Carlson compatible projection; only changes unsafe slopes."""

    secant = (y1 - y0) / width
    if np.isclose(secant, 0.0, rtol=0.0, atol=1.0e-12):
        return 0.0, 0.0
    if m0 * secant <= 0.0:
        m0 = 0.0
    if m1 * secant <= 0.0:
        m1 = 0.0
    alpha = m0 / secant
    beta = m1 / secant
    norm_sq = alpha**2 + beta**2
    if norm_sq > 9.0:
        scale = 3.0 / np.sqrt(norm_sq)
        m0 = scale * alpha * secant
        m1 = scale * beta * secant
    return float(m0), float(m1)


def _candidate_is_safe(
    x: FloatArray,
    values: FloatArray,
    *,
    domain_start: float,
    domain_end: float,
    y0: float,
    y1: float,
    m0: float,
    m1: float,
    residual_margin: float,
    allow_turn: bool,
) -> bool:
    if values.size < 3 or not np.all(np.isfinite(values)):
        return False
    width = float(domain_end - domain_start)
    if width <= 0.0:
        return False
    secant = (y1 - y0) / width
    left_line = y0 + m0 * (x - domain_start)
    right_line = y1 + m1 * (x - domain_end)
    lower = np.minimum.reduce((left_line, right_line, np.full_like(x, min(y0, y1)))) - residual_margin
    upper = np.maximum.reduce((left_line, right_line, np.full_like(x, max(y0, y1)))) + residual_margin
    if np.any(values < lower) or np.any(values > upper):
        return False
    derivative = np.gradient(values, x, edge_order=1)
    slope_scale = max(abs(secant), abs(m0), abs(m1), 1.0e-12)
    threshold = slope_scale * 1.0e-6
    signs = np.sign(np.where(np.abs(derivative) <= threshold, 0.0, derivative))
    signs = signs[signs != 0.0]
    turns = int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size > 1 else 0
    if turns > (1 if allow_turn else 0):
        return False
    if not allow_turn and secant != 0.0 and np.any(derivative * secant < -threshold * abs(secant)):
        return False
    slope_low = min(m0, m1, secant) - slope_scale
    slope_high = max(m0, m1, secant) + slope_scale
    return bool(np.all((derivative >= slope_low) & (derivative <= slope_high)))


def _adaptive_connection(
    frequencies: FloatArray,
    source: FloatArray,
    *,
    fb_hz: float,
    base_start_hz: float,
    base_end_hz: float,
    lower_limit_hz: float,
    upper_limit_hz: float,
    axis: Literal["log", "linear"],
    align_right_to_left: bool,
) -> _ConnectionCandidate | None:
    """Build a robust Hermite connection, expanding width only after rejection."""

    left_oct = max(np.log2(fb_hz / base_start_hz), 0.0)
    right_oct = max(np.log2(base_end_hz / fb_hz), 0.0)
    last_inputs: tuple | None = None
    for width_scale in (1.0, 1.25, 1.5, 2.0):
        f_start = max(fb_hz / (2.0 ** (left_oct * width_scale)), lower_limit_hz)
        f_end = min(fb_hz * (2.0 ** (right_oct * width_scale)), upper_limit_hz)
        indices = np.flatnonzero((frequencies >= f_start) & (frequencies <= f_end))
        left_indices = np.flatnonzero(frequencies < f_start)
        right_indices = np.flatnonzero(frequencies > f_end)
        if indices.size < 3 or left_indices.size < 1 or right_indices.size < 1:
            continue
        reference_left_oct = max(1.0 / 24.0, min(left_oct * width_scale, 1.0 / 3.0))
        reference_right_oct = max(1.0 / 24.0, min(right_oct * width_scale, 1.0 / 3.0))
        left_reference = left_indices[frequencies[left_indices] >= f_start / (2.0**reference_left_oct)]
        right_reference = right_indices[frequencies[right_indices] <= f_end * (2.0**reference_right_oct)]
        if left_reference.size < 2:
            left_reference = left_indices[-min(8, left_indices.size) :]
        if right_reference.size < 2:
            right_reference = right_indices[: min(8, right_indices.size)]
        x_all = np.log2(np.maximum(frequencies, 1.0e-12)) if axis == "log" else frequencies
        left_fit = _robust_linear_fit(x_all[left_reference], source[left_reference])
        right_fit = _robust_linear_fit(x_all[right_reference], source[right_reference])
        y0 = float(source[left_indices[-1]])
        y1 = float(source[right_indices[0]])
        if align_right_to_left:
            y1 = _align_phase_equivalent(y1, y0)
        x_band = x_all[indices]
        if axis == "log":
            x0, x1 = float(np.log2(f_start)), float(np.log2(f_end))
        else:
            x0, x1 = float(f_start), float(f_end)
        secant = (y1 - y0) / max(x1 - x0, np.finfo(np.float64).eps)
        m0 = left_fit.confidence * left_fit.slope + (1.0 - left_fit.confidence) * secant
        m1 = right_fit.confidence * right_fit.slope + (1.0 - right_fit.confidence) * secant
        residual_margin = 3.0 * max(left_fit.residual_scale, right_fit.residual_scale, 1.0e-9)
        allow_turn = bool(
            left_fit.confidence >= 0.8
            and right_fit.confidence >= 0.8
            and m0 * m1 < 0.0
        )
        candidate = _cubic_hermite(x_band, x0, x1, y0, y1, m0, m1)
        last_inputs = (
            x_band, x0, x1, y0, y1, m0, m1, residual_margin, allow_turn,
            left_fit, right_fit, width_scale, f_start, f_end, indices,
        )
        if _candidate_is_safe(
            x_band,
            candidate,
            domain_start=x0,
            domain_end=x1,
            y0=y0,
            y1=y1,
            m0=m0,
            m1=m1,
            residual_margin=residual_margin,
            allow_turn=allow_turn,
        ):
            return _ConnectionCandidate(
                candidate, "constrained_hermite", y0, y1, m0, m1,
                left_fit.confidence, right_fit.confidence, width_scale, 0.0,
                f_start, f_end, indices,
            )
    if last_inputs is None:
        return None
    (
        x_band, x0, x1, y0, y1, m0, m1, residual_margin, _allow_turn,
        left_fit, right_fit, width_scale, f_start, f_end, indices,
    ) = last_inputs
    safe_m0, safe_m1 = _monotone_slopes(y0, y1, m0, m1, x1 - x0)
    for retained in (0.75, 0.5, 0.25, 0.0):
        relaxed_m0 = safe_m0 * retained + ((y1 - y0) / (x1 - x0)) * (1.0 - retained)
        relaxed_m1 = safe_m1 * retained + ((y1 - y0) / (x1 - x0)) * (1.0 - retained)
        candidate = _cubic_hermite(x_band, x0, x1, y0, y1, relaxed_m0, relaxed_m1)
        if _candidate_is_safe(
            x_band,
            candidate,
            domain_start=x0,
            domain_end=x1,
            y0=y0,
            y1=y1,
            m0=relaxed_m0,
            m1=relaxed_m1,
            residual_margin=residual_margin,
            allow_turn=False,
        ):
            return _ConnectionCandidate(
                candidate, "constrained_hermite_relaxed", y0, y1, relaxed_m0, relaxed_m1,
                left_fit.confidence, right_fit.confidence, width_scale, 1.0 - retained,
                f_start, f_end, indices,
            )
    t = np.clip((x_band - x0) / (x1 - x0), 0.0, 1.0)
    weight = _smootherstep(t)
    fallback = (1.0 - weight) * y0 + weight * y1
    return _ConnectionCandidate(
        fallback, "fixed_endpoint_smootherstep_fallback", y0, y1, 0.0, 0.0,
        left_fit.confidence, right_fit.confidence, width_scale, 1.0,
        f_start, f_end, indices,
    )


def _one_sided_connection(
    frequencies: FloatArray,
    source: FloatArray,
    spec: OneSidedBoundarySpec,
) -> _ConnectionCandidate | None:
    """Build a one-sided Hermite connection while preserving the input slope."""

    boundary = _validated_boundary(spec.fb_hz)
    width_oct = _validated_octave_width(spec.delta_oct, "delta_oct", allow_zero=False)
    reference_oct = _validated_octave_width(spec.reference_oct, "reference_oct", allow_zero=False)
    if spec.side not in ("lower", "upper"):
        raise ValueError("side must be 'lower' or 'upper'")
    if spec.axis not in ("log", "linear"):
        raise ValueError("axis must be 'log' or 'linear'")
    outer_value = float(spec.outer_value)
    outer_slope = float(spec.outer_slope)
    if not np.isfinite(outer_value) or not np.isfinite(outer_slope):
        raise ValueError("outer_value and outer_slope must be finite")
    lower_limit = float(frequencies[0]) if spec.lower_limit_hz is None else float(spec.lower_limit_hz)
    upper_limit = float(frequencies[-1]) if spec.upper_limit_hz is None else float(spec.upper_limit_hz)
    if not np.isfinite(lower_limit) or not np.isfinite(upper_limit) or lower_limit >= upper_limit:
        raise ValueError("one-sided boundary limits must be finite and increasing")

    x_all = np.log2(np.maximum(frequencies, 1.0e-12)) if spec.axis == "log" else frequencies
    last_inputs: tuple | None = None
    for width_scale in (1.0, 1.25, 1.5, 2.0):
        if spec.side == "lower":
            f_start = max(boundary, lower_limit)
            f_end = min(boundary * (2.0 ** (width_oct * width_scale)), upper_limit)
            indices = np.flatnonzero((frequencies >= f_start) & (frequencies <= f_end))
            inside = np.flatnonzero(frequencies > f_end)
            reference = inside[frequencies[inside] <= f_end * (2.0**reference_oct)]
            if reference.size < 2:
                reference = inside[: min(8, inside.size)]
            if indices.size >= 3 and reference.size < 2:
                reference = indices[max(1, indices.size // 2) :]
            if indices.size < 3 or reference.size < 2:
                continue
            x_band = x_all[indices]
            x0, x1 = float(x_band[0]), float(x_band[-1])
            inner_value = float(source[indices[-1]])
            fit = _robust_linear_fit(x_all[reference], source[reference])
            secant = (inner_value - outer_value) / max(x1 - x0, np.finfo(np.float64).eps)
            inner_slope = (
                fit.slope
                if fit.confidence >= 0.9
                else fit.confidence * fit.slope + (1.0 - fit.confidence) * secant
            )
            y0, y1 = outer_value, inner_value
            m0, m1 = outer_slope, inner_slope
        else:
            f_start = max(boundary / (2.0 ** (width_oct * width_scale)), lower_limit)
            f_end = min(boundary, upper_limit)
            indices = np.flatnonzero((frequencies >= f_start) & (frequencies <= f_end))
            inside = np.flatnonzero(frequencies < f_start)
            reference = inside[frequencies[inside] >= f_start / (2.0**reference_oct)]
            if reference.size < 2:
                reference = inside[-min(8, inside.size) :]
            if indices.size >= 3 and reference.size < 2:
                reference = indices[: max(2, (indices.size + 1) // 2)]
            if indices.size < 3 or reference.size < 2:
                continue
            x_band = x_all[indices]
            x0, x1 = float(x_band[0]), float(x_band[-1])
            inner_value = float(source[indices[0]])
            fit = _robust_linear_fit(x_all[reference], source[reference])
            secant = (outer_value - inner_value) / max(x1 - x0, np.finfo(np.float64).eps)
            inner_slope = (
                fit.slope
                if fit.confidence >= 0.9
                else fit.confidence * fit.slope + (1.0 - fit.confidence) * secant
            )
            y0, y1 = inner_value, outer_value
            m0, m1 = inner_slope, outer_slope

        residual_margin = 3.0 * max(fit.residual_scale, 1.0e-9)
        candidate = _cubic_hermite(x_band, x0, x1, y0, y1, m0, m1)
        last_inputs = (
            x_band, x0, x1, y0, y1, m0, m1, residual_margin,
            fit, width_scale, f_start, f_end, indices,
        )
        if _candidate_is_safe(
            x_band,
            candidate,
            domain_start=x0,
            domain_end=x1,
            y0=y0,
            y1=y1,
            m0=m0,
            m1=m1,
            residual_margin=residual_margin,
            allow_turn=False,
        ):
            return _ConnectionCandidate(
                candidate,
                "one_sided_constrained_hermite",
                y0,
                y1,
                m0,
                m1,
                1.0 if spec.side == "lower" else fit.confidence,
                fit.confidence if spec.side == "lower" else 1.0,
                width_scale,
                0.0,
                f_start,
                f_end,
                indices,
            )

    if last_inputs is None:
        return None
    (
        x_band, x0, x1, y0, y1, m0, m1, residual_margin,
        fit, width_scale, f_start, f_end, indices,
    ) = last_inputs
    safe_m0, safe_m1 = _monotone_slopes(y0, y1, m0, m1, x1 - x0)
    inner_safe = safe_m1 if spec.side == "lower" else safe_m0
    outer_safe = safe_m0 if spec.side == "lower" else safe_m1
    for retained in (0.8, 0.6, 0.4, 0.0):
        relaxed_inner = inner_safe * retained
        if spec.side == "lower":
            relaxed_m0, relaxed_m1 = outer_safe, relaxed_inner
        else:
            relaxed_m0, relaxed_m1 = relaxed_inner, outer_safe
        candidate = _cubic_hermite(x_band, x0, x1, y0, y1, relaxed_m0, relaxed_m1)
        if _candidate_is_safe(
            x_band,
            candidate,
            domain_start=x0,
            domain_end=x1,
            y0=y0,
            y1=y1,
            m0=relaxed_m0,
            m1=relaxed_m1,
            residual_margin=residual_margin,
            allow_turn=False,
        ):
            return _ConnectionCandidate(
                candidate,
                "one_sided_constrained_hermite_relaxed",
                y0,
                y1,
                relaxed_m0,
                relaxed_m1,
                1.0 if spec.side == "lower" else fit.confidence,
                fit.confidence if spec.side == "lower" else 1.0,
                width_scale,
                1.0 - retained,
                f_start,
                f_end,
                indices,
            )

    t = np.clip((x_band - x0) / max(x1 - x0, np.finfo(np.float64).eps), 0.0, 1.0)
    fallback = (1.0 - _smootherstep(t)) * y0 + _smootherstep(t) * y1
    return _ConnectionCandidate(
        fallback,
        "one_sided_fixed_endpoint_smootherstep_fallback",
        y0,
        y1,
        0.0,
        0.0,
        1.0 if spec.side == "lower" else fit.confidence,
        fit.confidence if spec.side == "lower" else 1.0,
        width_scale,
        1.0,
        f_start,
        f_end,
        indices,
    )


def apply_one_sided_boundary(
    freqs_hz: ArrayLike,
    values: ArrayLike,
    spec: OneSidedBoundarySpec,
) -> OneSidedBoundaryResult:
    """Connect an input curve to an outer value/slope with adaptive Hermite."""

    if not isinstance(spec, OneSidedBoundarySpec):
        raise TypeError("spec must be OneSidedBoundarySpec")
    frequencies, source = _validate_batch_arrays(freqs_hz, values, minimum_samples=3)
    output = source.copy()
    if not spec.enabled:
        return OneSidedBoundaryResult(
            output,
            OneSidedBoundaryItem(
                spec.name, float(spec.fb_hz), spec.side, spec.axis, False, "disabled",
                None, None, 0, 0.0,
            ),
        )
    connection = _one_sided_connection(frequencies, source, spec)
    if connection is None:
        boundary = _validated_boundary(spec.fb_hz)
        width = _validated_octave_width(spec.delta_oct, "delta_oct", allow_zero=False)
        lower_limit = float(frequencies[0]) if spec.lower_limit_hz is None else float(spec.lower_limit_hz)
        upper_limit = float(frequencies[-1]) if spec.upper_limit_hz is None else float(spec.upper_limit_hz)
        if spec.side == "lower":
            f_start = max(boundary, lower_limit)
            f_end = min(boundary * (2.0**width), upper_limit)
        else:
            f_start = max(boundary / (2.0**width), lower_limit)
            f_end = min(boundary, upper_limit)
        if f_start >= f_end:
            indices = np.asarray([], dtype=np.int64)
        else:
            indices = np.flatnonzero((frequencies >= f_start) & (frequencies <= f_end))
        if indices.size:
            t = np.clip(
                np.log2(np.maximum(frequencies[indices], 1.0e-12) / f_start)
                / max(np.log2(f_end / f_start), np.finfo(np.float64).eps),
                0.0,
                1.0,
            )
            if spec.side == "lower":
                inner_value = float(source[indices[-1]])
                output[indices] = spec.outer_value + _smootherstep(t) * (inner_value - spec.outer_value)
            else:
                inner_value = float(source[indices[0]])
                output[indices] = inner_value + _smootherstep(t) * (spec.outer_value - inner_value)
            return OneSidedBoundaryResult(
                output,
                OneSidedBoundaryItem(
                    name=spec.name,
                    fb_hz=float(spec.fb_hz),
                    side=spec.side,
                    axis=spec.axis,
                    applied=True,
                    reason="applied with sparse fixed-endpoint fallback",
                    f_start_hz=f_start,
                    f_end_hz=f_end,
                    processed_points=int(indices.size),
                    max_change=float(np.max(np.abs(output[indices] - source[indices]))),
                    method="one_sided_sparse_smootherstep_fallback",
                    inner_value=inner_value,
                    outer_value=float(spec.outer_value),
                    inner_slope=0.0,
                    outer_slope=0.0,
                    inner_confidence=0.0,
                    width_scale=1.0,
                    slope_relaxation=1.0,
                ),
            )
        return OneSidedBoundaryResult(
            output,
            OneSidedBoundaryItem(
                spec.name, float(spec.fb_hz), spec.side, spec.axis, False,
                "insufficient robust reference samples", None, None, 0, 0.0,
            ),
        )
    original = output[connection.indices].copy()
    output[connection.indices] = connection.values
    if spec.side == "lower":
        inner_value = connection.right_value
        outer_value = connection.left_value
        inner_slope = connection.right_slope
        outer_slope = connection.left_slope
        inner_confidence = connection.right_confidence
    else:
        inner_value = connection.left_value
        outer_value = connection.right_value
        inner_slope = connection.left_slope
        outer_slope = connection.right_slope
        inner_confidence = connection.left_confidence
    return OneSidedBoundaryResult(
        output,
        OneSidedBoundaryItem(
            name=spec.name,
            fb_hz=float(spec.fb_hz),
            side=spec.side,
            axis=spec.axis,
            applied=True,
            reason="applied",
            f_start_hz=connection.f_start_hz,
            f_end_hz=connection.f_end_hz,
            processed_points=int(connection.indices.size),
            max_change=float(np.max(np.abs(connection.values - original))),
            method=connection.method,
            inner_value=inner_value,
            outer_value=outer_value,
            inner_slope=inner_slope,
            outer_slope=outer_slope,
            inner_confidence=inner_confidence,
            width_scale=connection.width_scale,
            slope_relaxation=connection.slope_relaxation,
        ),
    )


def _validated_boundary(boundary_hz: float) -> float:
    value = float(boundary_hz)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("boundary_hz must be finite and greater than 0 Hz.")
    return value


def _validated_octave_width(value: float, name: str, *, allow_zero: bool) -> float:
    width = float(value)
    minimum_ok = width >= 0.0 if allow_zero else width > 0.0
    if not np.isfinite(width) or not minimum_ok:
        comparison = "non-negative" if allow_zero else "greater than 0"
        raise ValueError(f"{name} must be finite and {comparison}.")
    return width


def _validate_batch_arrays(
    freqs_hz: ArrayLike,
    gain_db: ArrayLike,
    *,
    minimum_samples: int = 3,
) -> tuple[FloatArray, FloatArray]:
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    gain = np.asarray(gain_db, dtype=np.float64)
    if freqs.ndim != 1:
        raise ValueError("freqs_hz must be one-dimensional")
    if gain.ndim != 1:
        raise ValueError("gain_db must be one-dimensional")
    if freqs.shape != gain.shape:
        raise ValueError("freqs_hz and gain_db must have the same shape")
    if freqs.size < minimum_samples:
        raise ValueError(f"at least {minimum_samples} samples are required")
    if not np.all(np.isfinite(freqs)):
        raise ValueError("freqs_hz contains NaN or Inf")
    if not np.all(np.isfinite(gain)):
        raise ValueError("gain_db contains NaN or Inf")
    if np.any(freqs < 0.0):
        raise ValueError("negative frequencies are not allowed")
    if not np.all(np.diff(freqs) > 0.0):
        raise ValueError("freqs_hz must be strictly increasing")
    return freqs, gain


def _resolve_boundary_specs(
    specs: list[SharedBoundarySpec],
    *,
    freqs: FloatArray,
) -> list[ResolvedSharedBoundary]:
    resolved: list[ResolvedSharedBoundary] = []
    enabled_specs = [(index, spec) for index, spec in enumerate(specs) if spec.enabled]
    enabled_specs.sort(key=lambda item: item[1].fb_hz)
    previous_fb: float | None = None
    for original_index, spec in enabled_specs:
        fb = float(spec.fb_hz)
        left = float(spec.delta_oct_left)
        right = float(spec.delta_oct_right)
        if not np.isfinite(fb):
            raise ValueError(f"boundary[{original_index}].fb_hz must be finite")
        if fb <= freqs[0] or fb >= freqs[-1]:
            raise ValueError(f"boundary[{original_index}].fb_hz must be inside the frequency range")
        if previous_fb is not None and np.isclose(fb, previous_fb, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"duplicate shared boundary: {fb} Hz")
        if not np.isfinite(left):
            raise ValueError(f"boundary[{original_index}].delta_oct_left must be finite")
        if not np.isfinite(right):
            raise ValueError(f"boundary[{original_index}].delta_oct_right must be finite")
        if left < 0.0:
            raise ValueError(f"boundary[{original_index}].delta_oct_left must be >= 0")
        if right < 0.0:
            raise ValueError(f"boundary[{original_index}].delta_oct_right must be >= 0")
        if left == 0.0 and right == 0.0:
            continue
        if spec.curve != "smootherstep":
            raise ValueError(f"boundary[{original_index}].curve must be 'smootherstep'")
        f_start = fb / (2.0**left) if left > 0.0 else fb
        f_end = fb * (2.0**right) if right > 0.0 else fb
        f_start = max(f_start, float(freqs[0]))
        f_end = min(f_end, float(freqs[-1]))
        resolved.append(
            ResolvedSharedBoundary(
                index=original_index,
                name=spec.name,
                fb_hz=fb,
                f_start_hz=f_start,
                f_end_hz=f_end,
                curve=spec.curve,
            )
        )
        previous_fb = fb
    return resolved


def _resolve_overlaps(
    boundaries: list[ResolvedSharedBoundary],
    *,
    overlap_policy: OverlapPolicy,
) -> tuple[list[ResolvedSharedBoundary], set[int]]:
    if overlap_policy not in ("error", "skip_later", "clip_midpoint"):
        raise ValueError("overlap_policy must be 'error', 'skip_later', or 'clip_midpoint'")
    if len(boundaries) <= 1:
        return boundaries, set()
    output = list(boundaries)
    skipped_indexes: set[int] = set()
    for index in range(len(output) - 1):
        current = output[index]
        following = output[index + 1]
        if current.index in skipped_indexes or following.index in skipped_indexes:
            continue
        if current.f_end_hz <= following.f_start_hz:
            continue
        if overlap_policy == "error":
            raise ValueError(
                "shared boundary smoothing regions overlap: "
                f"{current.fb_hz:g} Hz [{current.f_start_hz:g}, {current.f_end_hz:g}] and "
                f"{following.fb_hz:g} Hz [{following.f_start_hz:g}, {following.f_end_hz:g}]"
            )
        if overlap_policy == "skip_later":
            skipped_indexes.add(following.index)
            continue
        split_hz = np.sqrt(current.fb_hz * following.fb_hz)
        split_hz = max(split_hz, current.fb_hz)
        split_hz = min(split_hz, following.fb_hz)
        if split_hz <= current.f_start_hz or split_hz >= following.f_end_hz:
            raise ValueError("overlap cannot be resolved by midpoint clipping")
        output[index] = ResolvedSharedBoundary(
            index=current.index,
            name=current.name,
            fb_hz=current.fb_hz,
            f_start_hz=current.f_start_hz,
            f_end_hz=split_hz,
            curve=current.curve,
            clipped_left=current.clipped_left,
            clipped_right=True,
        )
        output[index + 1] = ResolvedSharedBoundary(
            index=following.index,
            name=following.name,
            fb_hz=following.fb_hz,
            f_start_hz=split_hz,
            f_end_hz=following.f_end_hz,
            curve=following.curve,
            clipped_left=True,
            clipped_right=following.clipped_right,
        )
    return output, skipped_indexes


def _batch_item_not_applied(
    boundary: ResolvedSharedBoundary,
    reason: str,
) -> SharedBoundaryBatchItem:
    return SharedBoundaryBatchItem(
        index=boundary.index,
        name=boundary.name,
        fb_hz=boundary.fb_hz,
        applied=False,
        reason=reason,
        f_start_hz=boundary.f_start_hz,
        f_end_hz=boundary.f_end_hz,
        processed_points=0,
        max_change_db=0.0,
        clipped_left=boundary.clipped_left,
        clipped_right=boundary.clipped_right,
    )


def _apply_resolved_shared_boundary(
    freqs: FloatArray,
    gain_source: FloatArray,
    gain_output: FloatArray,
    boundary: ResolvedSharedBoundary,
    *,
    lower_limit_hz: float,
    upper_limit_hz: float,
) -> SharedBoundaryBatchItem:
    f_start = boundary.f_start_hz
    f_end = boundary.f_end_hz
    if f_start >= f_end:
        return _batch_item_not_applied(boundary, "empty smoothing region")
    connection = _adaptive_connection(
        freqs,
        gain_source,
        fb_hz=boundary.fb_hz,
        base_start_hz=f_start,
        base_end_hz=f_end,
        lower_limit_hz=lower_limit_hz,
        upper_limit_hz=upper_limit_hz,
        axis="log",
        align_right_to_left=False,
    )
    if connection is None:
        return _batch_item_not_applied(boundary, "insufficient robust reference samples")
    indices = connection.indices
    new_segment = connection.values
    previous_segment = gain_output[indices].copy()
    gain_output[indices] = new_segment
    max_change = float(np.max(np.abs(new_segment - previous_segment)))
    return SharedBoundaryBatchItem(
        index=boundary.index,
        name=boundary.name,
        fb_hz=boundary.fb_hz,
        applied=True,
        reason="applied",
        f_start_hz=connection.f_start_hz,
        f_end_hz=connection.f_end_hz,
        processed_points=int(indices.size),
        max_change_db=max_change,
        clipped_left=boundary.clipped_left,
        clipped_right=boundary.clipped_right,
        method=connection.method,
        left_slope=connection.left_slope,
        right_slope=connection.right_slope,
        left_confidence=connection.left_confidence,
        right_confidence=connection.right_confidence,
        width_scale=connection.width_scale,
        slope_relaxation=connection.slope_relaxation,
    )


def apply_shared_boundaries_batch(
    freqs_hz: ArrayLike,
    gain_db: ArrayLike,
    boundaries: list[SharedBoundarySpec],
    *,
    overlap_policy: OverlapPolicy = "error",
) -> SharedBoundaryBatchResult:
    """Apply multiple shared boundaries without processing-order dependence."""

    freqs, gain = _validate_batch_arrays(freqs_hz, gain_db)
    if not isinstance(boundaries, list):
        raise TypeError("boundaries must be a list")
    if len(boundaries) == 0:
        return SharedBoundaryBatchResult(
            gain_db=gain.copy(),
            items=(),
            applied_count=0,
            skipped_count=0,
            total_max_change_db=0.0,
        )
    for index, boundary in enumerate(boundaries):
        if not isinstance(boundary, SharedBoundarySpec):
            raise TypeError(f"boundaries[{index}] must be SharedBoundarySpec")
    resolved = _resolve_boundary_specs(boundaries, freqs=freqs)
    resolved, skipped_indexes = _resolve_overlaps(resolved, overlap_policy=overlap_policy)
    gain_source = gain.copy()
    gain_output = gain.copy()
    items: list[SharedBoundaryBatchItem] = []
    for resolved_index, boundary in enumerate(resolved):
        if boundary.index in skipped_indexes:
            items.append(
                _batch_item_not_applied(
                    boundary,
                    "skipped because smoothing region overlaps an earlier boundary",
                )
            )
            continue
        lower_limit = float(freqs[0])
        upper_limit = float(freqs[-1])
        if resolved_index > 0:
            lower_limit = resolved[resolved_index - 1].f_end_hz
        if resolved_index + 1 < len(resolved):
            upper_limit = resolved[resolved_index + 1].f_start_hz
        items.append(
            _apply_resolved_shared_boundary(
                freqs,
                gain_source,
                gain_output,
                boundary,
                lower_limit_hz=lower_limit,
                upper_limit_hz=upper_limit,
            )
        )
    if not np.all(np.isfinite(gain_output)):
        raise RuntimeError("batch boundary smoothing generated NaN or Inf")
    items.sort(key=lambda item: item.index)
    applied_count = sum(item.applied for item in items)
    skipped_count = len(items) - applied_count
    total_max_change = float(np.max(np.abs(gain_output - gain)))
    return SharedBoundaryBatchResult(
        gain_db=gain_output,
        items=tuple(items),
        applied_count=applied_count,
        skipped_count=skipped_count,
        total_max_change_db=total_max_change,
    )


def _apply_isolated_gain_boundary(
    frequencies: FloatArray,
    gain_output: FloatArray,
    spec: IsolatedGainBoundarySpec,
    index: int,
    *,
    lower_limit_hz: float | None = None,
    upper_limit_hz: float | None = None,
) -> IsolatedGainBoundaryItem:
    boundary = _validated_boundary(spec.fb_hz)
    width = _validated_octave_width(spec.delta_oct, "delta_oct", allow_zero=False)
    strength = float(spec.strength)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be finite and between 0 and 1")
    if spec.side not in ("lower", "upper"):
        raise ValueError("side must be 'lower' or 'upper'")
    outside = np.flatnonzero(frequencies > boundary) if spec.side == "upper" else np.flatnonzero(frequencies < boundary)
    region_start = boundary / (2.0**width) if spec.side == "upper" else boundary
    region_end = boundary if spec.side == "upper" else boundary * (2.0**width)
    if outside.size == 0:
        return IsolatedGainBoundaryItem(
            index, spec.name, boundary, spec.side, False, "missing outer reference",
            region_start, region_end, 0, 0.0,
        )
    outer_value = float(gain_output[outside[0] if spec.side == "upper" else outside[-1]])
    if strength == 0.0:
        return IsolatedGainBoundaryItem(
            index, spec.name, boundary, spec.side, False, "strength is zero", region_start, region_end, 0, 0.0
        )
    result = apply_one_sided_boundary(
        frequencies,
        gain_output,
        OneSidedBoundarySpec(
            fb_hz=boundary,
            delta_oct=width,
            side=spec.side,
            axis="log",
            outer_value=outer_value,
            outer_slope=0.0,
            reference_oct=1.0 / 6.0,
            lower_limit_hz=lower_limit_hz,
            upper_limit_hz=upper_limit_hz,
            name=spec.name,
        ),
    )
    item = result.item
    if not item.applied:
        return IsolatedGainBoundaryItem(
            index, spec.name, boundary, spec.side, False, item.reason,
            item.f_start_hz, item.f_end_hz, 0, 0.0,
        )
    updated = gain_output + strength * (result.values - gain_output)
    max_change = float(np.max(np.abs(updated - gain_output)))
    gain_output[:] = updated
    return IsolatedGainBoundaryItem(
        index=index,
        name=spec.name,
        fb_hz=boundary,
        side=spec.side,
        applied=True,
        reason="applied",
        f_start_hz=item.f_start_hz,
        f_end_hz=item.f_end_hz,
        processed_points=item.processed_points,
        max_change_db=max_change,
        method=item.method,
        inner_value_db=item.inner_value,
        outer_value_db=item.outer_value,
        inner_slope_db_per_oct=item.inner_slope,
        outer_slope_db_per_oct=item.outer_slope,
        inner_confidence=item.inner_confidence,
        width_scale=item.width_scale,
        slope_relaxation=item.slope_relaxation,
    )


def apply_auto_gain_eq_boundaries(
    freqs_hz: ArrayLike,
    gain_db: ArrayLike,
    shared_boundaries: Sequence[SharedBoundarySpec] = (),
    isolated_boundaries: Sequence[IsolatedGainBoundarySpec] = (),
    *,
    overlap_policy: OverlapPolicy = "error",
) -> AutoGainBoundaryBatchResult:
    """Apply all Auto Gain EQ shared and isolated boundaries in one call."""

    shared = list(shared_boundaries)
    isolated = list(isolated_boundaries)
    for index, spec in enumerate(shared):
        if not isinstance(spec, SharedBoundarySpec):
            raise TypeError(f"shared_boundaries[{index}] must be SharedBoundarySpec")
    for index, spec in enumerate(isolated):
        if not isinstance(spec, IsolatedGainBoundarySpec):
            raise TypeError(f"isolated_boundaries[{index}] must be IsolatedGainBoundarySpec")
    minimum_samples = 3 if any(spec.enabled for spec in shared) else 1
    frequencies, original = _validate_batch_arrays(
        freqs_hz,
        gain_db,
        minimum_samples=minimum_samples,
    )
    if any(spec.enabled for spec in shared):
        shared_result = apply_shared_boundaries_batch(
            frequencies,
            original,
            shared,
            overlap_policy=overlap_policy,
        )
    else:
        shared_result = SharedBoundaryBatchResult(
            gain_db=original.copy(),
            items=(),
            applied_count=0,
            skipped_count=0,
            total_max_change_db=0.0,
        )
    output = shared_result.gain_db.copy()
    isolated_items_list: list[IsolatedGainBoundaryItem] = []
    for index, spec in sorted(
        ((index, spec) for index, spec in enumerate(isolated) if spec.enabled),
        key=lambda item: item[1].fb_hz,
    ):
        lower_limit = None
        upper_limit = None
        if spec.side == "lower":
            starts = [
                item.f_start_hz
                for item in shared_result.items
                if item.applied and item.fb_hz > spec.fb_hz and item.f_start_hz is not None
            ]
            upper_limit = min(starts) if starts else None
        else:
            ends = [
                item.f_end_hz
                for item in shared_result.items
                if item.applied and item.fb_hz < spec.fb_hz and item.f_end_hz is not None
            ]
            lower_limit = max(ends) if ends else None
        isolated_items_list.append(
            _apply_isolated_gain_boundary(
                frequencies,
                output,
                spec,
                index,
                lower_limit_hz=lower_limit,
                upper_limit_hz=upper_limit,
            )
        )
    isolated_items = tuple(isolated_items_list)
    isolated_items = tuple(sorted(isolated_items, key=lambda item: item.index))
    applied_count = shared_result.applied_count + sum(item.applied for item in isolated_items)
    skipped_count = shared_result.skipped_count + sum(not item.applied for item in isolated_items)
    return AutoGainBoundaryBatchResult(
        gain_db=output,
        shared_items=shared_result.items,
        isolated_items=isolated_items,
        applied_count=applied_count,
        skipped_count=skipped_count,
        total_max_change_db=float(np.max(np.abs(output - original))),
    )


def wrap_phase_deg(phase_deg: ArrayLike) -> FloatArray:
    """Wrap phase to the half-open interval [-180, 180) degrees."""

    phase = np.asarray(phase_deg, dtype=np.float64)
    return (phase + 180.0) % 360.0 - 180.0


def unwrap_phase_deg(phase_deg: ArrayLike) -> FloatArray:
    """Unwrap a degree-phase array."""

    phase = np.asarray(phase_deg, dtype=np.float64)
    return np.rad2deg(np.unwrap(np.deg2rad(phase)))


def _prepare_phase(
    phase_deg: FloatArray,
    *,
    input_phase_mode: InputPhaseMode,
) -> FloatArray:
    if input_phase_mode == "wrapped":
        return unwrap_phase_deg(phase_deg)
    if input_phase_mode == "unwrapped":
        return phase_deg.copy()
    raise ValueError("input_phase_mode must be 'wrapped' or 'unwrapped'")


def _validate_phase_arrays(
    freqs_hz: ArrayLike,
    phase_deg: ArrayLike,
    *,
    minimum_samples: int = 4,
) -> tuple[FloatArray, FloatArray]:
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    phase = np.asarray(phase_deg, dtype=np.float64)
    if freqs.ndim != 1:
        raise ValueError("freqs_hz must be one-dimensional")
    if phase.ndim != 1:
        raise ValueError("phase_deg must be one-dimensional")
    if freqs.shape != phase.shape:
        raise ValueError("freqs_hz and phase_deg must have the same shape")
    if freqs.size < minimum_samples:
        raise ValueError(f"at least {minimum_samples} frequency samples are required")
    if not np.all(np.isfinite(freqs)):
        raise ValueError("freqs_hz contains NaN or Inf")
    if not np.all(np.isfinite(phase)):
        raise ValueError("phase_deg contains NaN or Inf")
    if np.any(freqs < 0.0):
        raise ValueError("negative frequencies are not allowed")
    if not np.all(np.diff(freqs) > 0.0):
        raise ValueError("freqs_hz must be strictly increasing")
    if not np.any(freqs > 0.0):
        raise ValueError("frequency axis must contain positive frequencies")
    return freqs, phase


def _robust_phase_reference(
    freqs: FloatArray,
    phase_unwrapped_deg: FloatArray,
    *,
    edge_hz: float,
    side: Literal["left", "right"],
    reference_oct: float,
) -> float | None:
    if reference_oct < 0.0:
        raise ValueError("reference_oct must be >= 0")
    if side == "left":
        if reference_oct == 0.0:
            indices = np.flatnonzero(freqs < edge_hz)
            return None if indices.size == 0 else float(phase_unwrapped_deg[indices[-1]])
        f_low = edge_hz / (2.0**reference_oct)
        mask = (freqs >= f_low) & (freqs < edge_hz)
    elif side == "right":
        if reference_oct == 0.0:
            indices = np.flatnonzero(freqs > edge_hz)
            return None if indices.size == 0 else float(phase_unwrapped_deg[indices[0]])
        f_high = edge_hz * (2.0**reference_oct)
        mask = (freqs > edge_hz) & (freqs <= f_high)
    else:
        raise ValueError("side must be 'left' or 'right'")
    values = phase_unwrapped_deg[mask]
    return None if values.size == 0 else float(np.median(values))


def _align_phase_equivalent(right_phase_deg: float, left_phase_deg: float) -> float:
    """Return the 360-degree equivalent right phase nearest the left phase."""

    turns = np.round((left_phase_deg - right_phase_deg) / 360.0)
    return float(right_phase_deg + 360.0 * turns)


def phase_to_group_delay_seconds(
    freqs_hz: FloatArray,
    phase_unwrapped_deg: FloatArray,
) -> FloatArray:
    """Calculate group delay from unwrapped phase."""

    phase_rad = np.deg2rad(phase_unwrapped_deg)
    derivative = np.gradient(phase_rad, freqs_hz, edge_order=1)
    return -derivative / (2.0 * np.pi)


def _resolve_auto_phase_specs(
    specs: Sequence[AutoPhaseBoundarySpec],
    *,
    freqs: FloatArray,
) -> list[ResolvedAutoPhaseBoundary]:
    resolved: list[ResolvedAutoPhaseBoundary] = []
    enabled_specs = [(index, spec) for index, spec in enumerate(specs) if spec.enabled]
    enabled_specs.sort(key=lambda item: item[1].fb_hz)
    previous_fb: float | None = None
    for original_index, spec in enabled_specs:
        fb = float(spec.fb_hz)
        left_oct = float(spec.delta_oct_left)
        right_oct = float(spec.delta_oct_right)
        reference_oct = float(spec.reference_oct)
        if not np.isfinite(fb):
            raise ValueError(f"boundary[{original_index}].fb_hz must be finite")
        if fb <= freqs[0] or fb >= freqs[-1]:
            raise ValueError(f"boundary[{original_index}].fb_hz must be inside the frequency range")
        if previous_fb is not None and np.isclose(fb, previous_fb, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"duplicate AutoPhaseEQ boundary: {fb:g} Hz")
        for name, value in (
            ("delta_oct_left", left_oct),
            ("delta_oct_right", right_oct),
            ("reference_oct", reference_oct),
        ):
            if not np.isfinite(value):
                raise ValueError(f"boundary[{original_index}].{name} must be finite")
            if value < 0.0:
                raise ValueError(f"boundary[{original_index}].{name} must be >= 0")
        if left_oct == 0.0 and right_oct == 0.0:
            previous_fb = fb
            continue
        if spec.curve != "smootherstep":
            raise ValueError(f"boundary[{original_index}].curve must be 'smootherstep'")
        if spec.phase_turn_policy not in ("preserve_unwrapped", "nearest_equivalent"):
            raise ValueError(f"boundary[{original_index}].phase_turn_policy is invalid")
        max_change = spec.max_phase_change_deg
        if max_change is not None:
            max_change = float(max_change)
            if not np.isfinite(max_change) or max_change <= 0.0:
                raise ValueError(f"boundary[{original_index}].max_phase_change_deg must be greater than 0")
        f_start = fb / (2.0**left_oct) if left_oct > 0.0 else fb
        f_end = fb * (2.0**right_oct) if right_oct > 0.0 else fb
        resolved.append(
            ResolvedAutoPhaseBoundary(
                index=original_index,
                name=spec.name,
                fb_hz=fb,
                f_start_hz=max(f_start, float(freqs[0])),
                f_end_hz=min(f_end, float(freqs[-1])),
                curve=spec.curve,
                reference_oct=reference_oct,
                phase_turn_policy=spec.phase_turn_policy,
                max_phase_change_deg=max_change,
            )
        )
        previous_fb = fb
    return resolved


def _resolve_auto_phase_overlaps(
    boundaries: list[ResolvedAutoPhaseBoundary],
    *,
    overlap_policy: OverlapPolicy,
) -> tuple[list[ResolvedAutoPhaseBoundary], set[int]]:
    if overlap_policy not in ("error", "skip_later", "clip_midpoint"):
        raise ValueError("overlap_policy must be 'error', 'skip_later', or 'clip_midpoint'")
    if len(boundaries) <= 1:
        return boundaries, set()
    output = list(boundaries)
    skipped_indexes: set[int] = set()
    for index in range(len(output) - 1):
        current = output[index]
        following = output[index + 1]
        if current.index in skipped_indexes or following.index in skipped_indexes:
            continue
        if current.f_end_hz <= following.f_start_hz:
            continue
        if overlap_policy == "error":
            raise ValueError(
                "AutoPhaseEQ smoothing regions overlap: "
                f"{current.fb_hz:g} Hz and {following.fb_hz:g} Hz"
            )
        if overlap_policy == "skip_later":
            skipped_indexes.add(following.index)
            continue
        split_hz = float(np.sqrt(current.fb_hz * following.fb_hz))
        split_hz = float(np.clip(split_hz, current.fb_hz, following.fb_hz))
        output[index] = ResolvedAutoPhaseBoundary(
            index=current.index,
            name=current.name,
            fb_hz=current.fb_hz,
            f_start_hz=current.f_start_hz,
            f_end_hz=split_hz,
            curve=current.curve,
            reference_oct=current.reference_oct,
            phase_turn_policy=current.phase_turn_policy,
            max_phase_change_deg=current.max_phase_change_deg,
            clipped_left=current.clipped_left,
            clipped_right=True,
        )
        output[index + 1] = ResolvedAutoPhaseBoundary(
            index=following.index,
            name=following.name,
            fb_hz=following.fb_hz,
            f_start_hz=split_hz,
            f_end_hz=following.f_end_hz,
            curve=following.curve,
            reference_oct=following.reference_oct,
            phase_turn_policy=following.phase_turn_policy,
            max_phase_change_deg=following.max_phase_change_deg,
            clipped_left=True,
            clipped_right=following.clipped_right,
        )
    return output, skipped_indexes


def _auto_phase_item_not_applied(
    boundary: ResolvedAutoPhaseBoundary,
    reason: str,
    *,
    left_reference_deg: float | None = None,
    right_reference_deg: float | None = None,
    max_phase_change_deg: float = 0.0,
) -> AutoPhaseBoundaryItem:
    return AutoPhaseBoundaryItem(
        index=boundary.index,
        name=boundary.name,
        fb_hz=boundary.fb_hz,
        applied=False,
        reason=reason,
        f_start_hz=boundary.f_start_hz,
        f_end_hz=boundary.f_end_hz,
        processed_points=0,
        left_reference_deg=left_reference_deg,
        right_reference_deg=right_reference_deg,
        max_phase_change_deg=max_phase_change_deg,
        max_group_delay_change_ms=0.0,
        clipped_left=boundary.clipped_left,
        clipped_right=boundary.clipped_right,
    )


def _apply_resolved_auto_phase_boundary(
    freqs: FloatArray,
    phase_source: FloatArray,
    phase_output: FloatArray,
    boundary: ResolvedAutoPhaseBoundary,
    *,
    lower_limit_hz: float,
    upper_limit_hz: float,
) -> AutoPhaseBoundaryItem:
    f_start = boundary.f_start_hz
    f_end = boundary.f_end_hz
    if f_start >= f_end:
        return _auto_phase_item_not_applied(boundary, "empty smoothing region")
    connection = _adaptive_connection(
        freqs,
        phase_source,
        fb_hz=boundary.fb_hz,
        base_start_hz=f_start,
        base_end_hz=f_end,
        lower_limit_hz=lower_limit_hz,
        upper_limit_hz=upper_limit_hz,
        axis="linear",
        align_right_to_left=boundary.phase_turn_policy == "nearest_equivalent",
    )
    if connection is None:
        return _auto_phase_item_not_applied(boundary, "insufficient robust reference samples")
    indices = connection.indices
    f_band = freqs[indices]
    left_phase = connection.left_value
    right_phase = connection.right_value
    new_segment = connection.values
    original_segment = phase_output[indices].copy()
    max_phase_change = float(np.max(np.abs(new_segment - original_segment)))
    if (
        boundary.max_phase_change_deg is not None
        and max_phase_change > boundary.max_phase_change_deg
    ):
        return _auto_phase_item_not_applied(
            boundary,
            "maximum phase change exceeds configured limit",
            left_reference_deg=left_phase,
            right_reference_deg=right_phase,
            max_phase_change_deg=max_phase_change,
        )
    group_delay_before = phase_to_group_delay_seconds(f_band, original_segment)
    group_delay_after = phase_to_group_delay_seconds(f_band, new_segment)
    max_group_delay_change_ms = float(np.max(np.abs(group_delay_after - group_delay_before)) * 1000.0)
    phase_output[indices] = new_segment
    return AutoPhaseBoundaryItem(
        index=boundary.index,
        name=boundary.name,
        fb_hz=boundary.fb_hz,
        applied=True,
        reason="applied",
        f_start_hz=connection.f_start_hz,
        f_end_hz=connection.f_end_hz,
        processed_points=int(indices.size),
        left_reference_deg=left_phase,
        right_reference_deg=right_phase,
        max_phase_change_deg=max_phase_change,
        max_group_delay_change_ms=max_group_delay_change_ms,
        clipped_left=boundary.clipped_left,
        clipped_right=boundary.clipped_right,
        method=connection.method,
        left_slope_deg_per_hz=connection.left_slope,
        right_slope_deg_per_hz=connection.right_slope,
        left_confidence=connection.left_confidence,
        right_confidence=connection.right_confidence,
        width_scale=connection.width_scale,
        slope_relaxation=connection.slope_relaxation,
    )


def _apply_auto_phase_outer_boundaries(
    freqs: FloatArray,
    phase_output: FloatArray,
    specs: Sequence[AutoPhaseOuterBoundarySpec],
    strength: float,
    shared_items: Sequence[AutoPhaseBoundaryItem] = (),
) -> tuple[FloatArray, tuple[AutoPhaseOuterBoundaryItem, ...]]:
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("outer_transition_strength must be finite and between 0 and 1")
    source = phase_output.copy()
    tapered = phase_output.copy()
    items: list[AutoPhaseOuterBoundaryItem] = []
    enabled_specs = sorted(
        ((index, spec) for index, spec in enumerate(specs) if spec.enabled),
        key=lambda item: item[1].fb_hz,
    )
    for index, spec in enabled_specs:
        if not isinstance(spec, AutoPhaseOuterBoundarySpec):
            raise TypeError(f"outer_boundaries[{index}] must be AutoPhaseOuterBoundarySpec")
        boundary = _validated_boundary(spec.fb_hz)
        width = _validated_octave_width(spec.delta_oct, "delta_oct", allow_zero=False)
        if spec.side not in ("lower", "upper"):
            raise ValueError("side must be 'lower' or 'upper'")
        if spec.phase_turn_policy not in ("preserve_unwrapped", "nearest_equivalent"):
            raise ValueError("phase_turn_policy must be 'preserve_unwrapped' or 'nearest_equivalent'")
        boundary_index = int(np.argmin(np.abs(freqs - boundary)))
        boundary_phase = float(source[boundary_index])
        base_phase = (
            _align_phase_equivalent(0.0, boundary_phase)
            if spec.phase_turn_policy == "nearest_equivalent"
            else 0.0
        )
        before = tapered.copy()
        lower_limit = None
        upper_limit = None
        if spec.side == "lower":
            starts = [
                item.f_start_hz
                for item in shared_items
                if item.applied and item.fb_hz > boundary and item.f_start_hz is not None
            ]
            upper_limit = min(starts) if starts else None
        else:
            ends = [
                item.f_end_hz
                for item in shared_items
                if item.applied and item.fb_hz < boundary and item.f_end_hz is not None
            ]
            lower_limit = max(ends) if ends else None
        connection = apply_one_sided_boundary(
            freqs,
            before,
            OneSidedBoundarySpec(
                fb_hz=boundary,
                delta_oct=width,
                side=spec.side,
                axis="linear",
                outer_value=base_phase,
                outer_slope=0.0,
                reference_oct=1.0 / 24.0,
                lower_limit_hz=lower_limit,
                upper_limit_hz=upper_limit,
                name=spec.name,
            ),
        )
        connection_item = connection.item
        if connection_item.applied:
            tapered = connection.values
            outside_mask = freqs < boundary if spec.side == "lower" else freqs > boundary
            tapered[outside_mask] = base_phase
        else:
            tapered = before.copy()
        actual_change = strength * (tapered - before)
        group_delay_before = phase_to_group_delay_seconds(freqs, before)
        group_delay_after = phase_to_group_delay_seconds(freqs, before + actual_change)
        items.append(
            AutoPhaseOuterBoundaryItem(
                index=index,
                name=spec.name,
                fb_hz=boundary,
                side=spec.side,
                applied=connection_item.applied and strength > 0.0,
                reason=(
                    connection_item.reason
                    if not connection_item.applied
                    else ("applied" if strength > 0.0 else "strength is zero")
                ),
                processed_points=connection_item.processed_points,
                base_phase_deg=base_phase,
                max_phase_change_deg=float(np.max(np.abs(actual_change))),
                max_group_delay_change_ms=float(
                    np.max(np.abs(group_delay_after - group_delay_before)) * 1000.0
                ),
                method=connection_item.method,
                inner_phase_deg=connection_item.inner_value,
                inner_slope_deg_per_hz=connection_item.inner_slope,
                outer_slope_deg_per_hz=connection_item.outer_slope,
                inner_confidence=connection_item.inner_confidence,
                width_scale=connection_item.width_scale,
                slope_relaxation=connection_item.slope_relaxation,
            )
        )
    output = source * (1.0 - strength) + tapered * strength
    return output, tuple(sorted(items, key=lambda item: item.index))


def apply_auto_phase_eq_boundaries(
    freqs_hz: ArrayLike,
    phase_deg: ArrayLike,
    boundaries: Sequence[AutoPhaseBoundarySpec],
    *,
    outer_boundaries: Sequence[AutoPhaseOuterBoundarySpec] = (),
    outer_transition_strength: float = 1.0,
    input_phase_mode: InputPhaseMode = "wrapped",
    output_phase_mode: OutputPhaseMode = "wrapped",
    overlap_policy: OverlapPolicy = "error",
) -> AutoPhaseBatchResult:
    """Apply multiple shared Auto Phase EQ boundaries as one batch."""

    if input_phase_mode not in ("wrapped", "unwrapped"):
        raise ValueError("input_phase_mode must be 'wrapped' or 'unwrapped'")
    if output_phase_mode not in ("wrapped", "unwrapped"):
        raise ValueError("output_phase_mode must be 'wrapped' or 'unwrapped'")
    if not isinstance(boundaries, Sequence):
        raise TypeError("boundaries must be a sequence")
    for index, boundary in enumerate(boundaries):
        if not isinstance(boundary, AutoPhaseBoundarySpec):
            raise TypeError(f"boundaries[{index}] must be AutoPhaseBoundarySpec")
    for index, boundary in enumerate(outer_boundaries):
        if not isinstance(boundary, AutoPhaseOuterBoundarySpec):
            raise TypeError(f"outer_boundaries[{index}] must be AutoPhaseOuterBoundarySpec")
    minimum_samples = 4 if any(boundary.enabled for boundary in boundaries) else 1
    if any(boundary.enabled for boundary in outer_boundaries):
        minimum_samples = max(minimum_samples, 2)
    freqs, phase_input = _validate_phase_arrays(
        freqs_hz,
        phase_deg,
        minimum_samples=minimum_samples,
    )
    phase_source = _prepare_phase(phase_input, input_phase_mode=input_phase_mode)
    phase_output = phase_source.copy()
    if len(boundaries) == 0 and len(outer_boundaries) == 0:
        final_phase = wrap_phase_deg(phase_output) if output_phase_mode == "wrapped" else phase_output.copy()
        return AutoPhaseBatchResult(
            phase_deg=final_phase,
            phase_unwrapped_deg=phase_output,
            items=(),
            outer_items=(),
            applied_count=0,
            skipped_count=0,
            total_max_phase_change_deg=0.0,
            total_max_group_delay_change_ms=0.0,
        )
    resolved = _resolve_auto_phase_specs(boundaries, freqs=freqs)
    resolved, skipped_indexes = _resolve_auto_phase_overlaps(resolved, overlap_policy=overlap_policy)
    items: list[AutoPhaseBoundaryItem] = []
    for resolved_index, boundary in enumerate(resolved):
        if boundary.index in skipped_indexes:
            items.append(
                _auto_phase_item_not_applied(
                    boundary,
                    "skipped because smoothing region overlaps an earlier boundary",
                )
            )
            continue
        items.append(
            _apply_resolved_auto_phase_boundary(
                freqs,
                phase_source,
                phase_output,
                boundary,
                lower_limit_hz=(resolved[resolved_index - 1].f_end_hz if resolved_index > 0 else float(freqs[0])),
                upper_limit_hz=(
                    resolved[resolved_index + 1].f_start_hz
                    if resolved_index + 1 < len(resolved)
                    else float(freqs[-1])
                ),
            )
        )
    phase_output, outer_items = _apply_auto_phase_outer_boundaries(
        freqs,
        phase_output,
        outer_boundaries,
        outer_transition_strength,
        items,
    )
    if not np.all(np.isfinite(phase_output)):
        raise RuntimeError("AutoPhaseEQ generated NaN or Inf")
    total_phase_change = float(np.max(np.abs(phase_output - phase_source)))
    group_delay_source = phase_to_group_delay_seconds(freqs, phase_source)
    group_delay_output = phase_to_group_delay_seconds(freqs, phase_output)
    total_group_delay_change_ms = float(
        np.max(np.abs(group_delay_output - group_delay_source)) * 1000.0
    )
    items.sort(key=lambda item: item.index)
    applied_count = sum(item.applied for item in items) + sum(item.applied for item in outer_items)
    skipped_count = len(items) + len(outer_items) - applied_count
    final_phase = wrap_phase_deg(phase_output) if output_phase_mode == "wrapped" else phase_output.copy()
    return AutoPhaseBatchResult(
        phase_deg=final_phase,
        phase_unwrapped_deg=phase_output,
        items=tuple(items),
        outer_items=outer_items,
        applied_count=applied_count,
        skipped_count=skipped_count,
        total_max_phase_change_deg=total_phase_change,
        total_max_group_delay_change_ms=total_group_delay_change_ms,
    )


def _demo_curves(sample_rate_hz: float, fft_size: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be finite and greater than 0.")
    if fft_size < 8 or fft_size % 2:
        raise ValueError("fft_size must be an even integer of at least 8.")
    frequencies = np.linspace(0.0, sample_rate_hz / 2.0, fft_size // 2 + 1)
    original = np.where(frequencies < 1000.0, -24.0, 0.0)
    smoothed = apply_shared_boundaries_batch(
        frequencies,
        original,
        [SharedBoundarySpec(1000.0, 0.5, 1.0 / 6.0)],
    ).gain_db
    return frequencies, original, smoothed


def _write_demo_csv(
    path: Path,
    frequencies_hz: FloatArray,
    original_db: FloatArray,
    smoothed_db: FloatArray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_hz", "original_db", "smoothed_db"])
        writer.writerows(zip(frequencies_hz, original_db, smoothed_db))


def main(argv: list[str] | None = None) -> int:
    """Run a self-contained asymmetric-boundary demonstration."""

    parser = argparse.ArgumentParser(description="Run the octave boundary smoothing demonstration.")
    parser.add_argument("--sample-rate", type=float, default=48_000.0, help="Sample rate in Hz.")
    parser.add_argument("--fft-size", type=int, default=4096, help="Even FFT size.")
    parser.add_argument("--csv", type=Path, help="Optional output CSV path.")
    args = parser.parse_args(argv)

    frequencies, original, smoothed = _demo_curves(args.sample_rate, args.fft_size)
    changed = np.flatnonzero(~np.isclose(original, smoothed, atol=1e-12, rtol=0.0))
    print("Octave boundary smoothing demo")
    print(f"frequency bins: {frequencies.size}")
    print(f"changed bins: {changed.size}")
    print(f"gain range: {np.min(smoothed):.3f} to {np.max(smoothed):.3f} dB")
    if args.csv is not None:
        _write_demo_csv(args.csv, frequencies, original, smoothed)
        print(f"CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
