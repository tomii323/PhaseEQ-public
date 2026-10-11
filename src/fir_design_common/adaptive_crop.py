"""Coefficient-driven high-precision FIR cropping.

The module is UI independent.  It analyses both ends with the same routine,
keeps the input immutable, and returns an explicit timing representation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
from numbers import Integral
from typing import Literal

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import find_peaks



THRESHOLD_DB = -132.0
ACTIVE_FLOOR_DB = -80.0
MAX_GAIN_ERROR_DB = 0.05
MAX_PHASE_ERROR_DEG = 0.5
MAX_COMPLEX_ERROR = 0.01
MAX_REMOVED_ENERGY_RATIO = 1e-6
MAX_DIP_ENDPOINT_DB = -40.0
ALGORITHM_VERSION = "adaptive-fir-crop-v6"


@dataclass(frozen=True)
class SideCandidate:
    cut_samples: int
    source: Literal["none", "zero", "threshold", "dip"]
    endpoint_db: float
    outer_peak_db: float
    removed_energy_ratio: float


@dataclass(frozen=True)
class AdaptiveCropMetrics:
    gain_error_db: float
    phase_error_deg: float
    complex_error: float
    removed_energy_ratio: float


@dataclass(frozen=True)
class AdaptiveCropPlan:
    """Immutable range decision; coefficients and time origin are untouched."""

    input_taps: int
    start: int
    stop: int
    adopted: bool
    metrics: AdaptiveCropMetrics
    source_left: str = "none"
    source_right: str = "none"
    candidate_count: int = 0
    input_digest: str = ""
    requested_minimum_taps: object = 1
    effective_minimum_taps: int = 1
    minimum_expanded: bool = False

    @property
    def taps(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class AdaptiveCropResult:
    coefficients: np.ndarray
    adopted: bool
    reason: str
    original_taps: int
    direct_taps: int
    final_taps: int
    left_removed: int
    right_removed: int
    padding_left: int
    padding_right: int
    timing_mode: Literal["embedded_zero_padding", "external_delay", "none"]
    crop_delay_samples: float
    source_left: str
    source_right: str
    metrics: AdaptiveCropMetrics
    original_digest: str = ""
    final_digest: str = ""
    candidate_count: int = 0
    target_kind: str = "original_fir_complex_response"
    window_mode: str = "none"
    algorithm_version: str = ALGORITHM_VERSION
    base_candidate_taps: int = 0
    base_left_removed: int = 0
    base_right_removed: int = 0
    protection_length_ratio: float = 1.0
    requested_minimum_taps: object = 1
    effective_minimum_taps: int = 1
    minimum_expanded: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm_version": self.algorithm_version,
            "adopted": self.adopted,
            "reason": self.reason,
            "original_taps": self.original_taps,
            "direct_taps": self.direct_taps,
            "final_taps": self.final_taps,
            "left_removed": self.left_removed,
            "right_removed": self.right_removed,
            "padding_left": self.padding_left,
            "padding_right": self.padding_right,
            "timing_mode": self.timing_mode,
            "crop_delay_samples": self.crop_delay_samples,
            "source_left": self.source_left,
            "source_right": self.source_right,
            "gain_error_db": self.metrics.gain_error_db,
            "phase_error_deg": self.metrics.phase_error_deg,
            "complex_error": self.metrics.complex_error,
            "removed_energy_ratio": self.metrics.removed_energy_ratio,
            "original_digest": self.original_digest,
            "final_digest": self.final_digest,
            "candidate_count": self.candidate_count,
            "target_kind": self.target_kind,
            "window_mode": self.window_mode,
            "base_candidate_taps": self.base_candidate_taps,
            "base_left_removed": self.base_left_removed,
            "base_right_removed": self.base_right_removed,
            "protection_length_ratio": self.protection_length_ratio,
            "requested_minimum_taps": self.requested_minimum_taps,
            "effective_minimum_taps": self.effective_minimum_taps,
            "minimum_expanded": self.minimum_expanded,
        }


def _validated(values: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError("FIR coefficients must be real")
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError("FIR coefficients must be a nonempty finite vector")
    return result.copy()


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _side_candidates(values: np.ndarray) -> tuple[SideCandidate, ...]:
    count = values.size
    half = max(1, count // 2)
    peak = float(np.max(np.abs(values)))
    if peak == 0.0:
        return (SideCandidate(0, "none", -np.inf, -np.inf, 0.0),)
    level = 20.0 * np.log10(np.maximum(np.abs(values) / peak, 1e-15))
    span = max(3, min(65, 2 * max(1, count // 400) + 1))
    envelope = maximum_filter1d(level, size=span, mode="nearest")
    smooth = uniform_filter1d(envelope, size=span, mode="nearest")
    energy = float(np.dot(values, values))

    positions: dict[int, str] = {0: "none"}
    nonzero = np.flatnonzero(values)
    zero_run = int(nonzero[0]) if nonzero.size else 0
    if 0 < zero_run < half:
        # The exact Unity-EQ support boundary must be evaluated even when
        # smoothing moves the threshold candidate into significant samples.
        positions[zero_run] = "zero"
    threshold_run = 0
    for index in range(half):
        if envelope[index] > THRESHOLD_DB:
            break
        threshold_run = index + 1
    if threshold_run:
        positions[threshold_run] = "threshold"

    distance = max(2, count // 200)
    dips, _ = find_peaks(-smooth[:half], prominence=6.0, distance=distance)
    for index in dips[:12]:
        cut = int(index)
        if cut > 0:
            positions.setdefault(cut, "dip")

    result = []
    for cut, source in sorted(positions.items()):
        endpoint = min(cut, count - 1)
        removed = values[:cut]
        removed_energy = float(np.dot(removed, removed) / energy) if energy else 0.0
        result.append(SideCandidate(
            cut, source, float(level[endpoint]),
            float(np.max(level[:cut])) if cut else -np.inf,
            removed_energy,
        ))
    return tuple(result)


def _centered_response(values: np.ndarray, fft_size: int) -> np.ndarray:
    frequency = np.fft.rfftfreq(fft_size)
    center = (values.size - 1) / 2.0
    return np.fft.rfft(values, fft_size) * np.exp(2j * np.pi * frequency * center)


def _metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    centered_shift_samples: float = 0.0,
) -> AdaptiveCropMetrics:
    fft_size = 1 << max(4, (max(reference.size, candidate.size) * 8 - 1).bit_length())
    ref = _centered_response(reference, fft_size)
    got = _centered_response(candidate, fft_size)
    if centered_shift_samples:
        frequency = np.fft.rfftfreq(fft_size)
        got *= np.exp(2j * np.pi * frequency * centered_shift_samples)
    ref_abs = np.abs(ref)
    peak = max(float(np.max(ref_abs)), 1e-15)
    mask = ref_abs >= peak * 10.0 ** (ACTIVE_FLOOR_DB / 20.0)
    if not np.any(mask):
        return AdaptiveCropMetrics(np.inf, np.inf, np.inf, np.inf)
    gain = np.max(np.abs(
        20.0 * np.log10(np.maximum(np.abs(got[mask]), 1e-15))
        - 20.0 * np.log10(np.maximum(ref_abs[mask], 1e-15))
    ))
    phase = np.unwrap(np.angle(got[mask] / ref[mask]))
    phase_error = np.max(np.abs(np.rad2deg(phase)))
    complex_error = np.max(np.abs(got[mask] - ref[mask])) / peak
    return AdaptiveCropMetrics(float(gain), float(phase_error), float(complex_error), 0.0)


def _passes(metrics: AdaptiveCropMetrics) -> bool:
    return (
        metrics.gain_error_db <= MAX_GAIN_ERROR_DB
        and metrics.phase_error_deg <= MAX_PHASE_ERROR_DEG
        and metrics.complex_error <= MAX_COMPLEX_ERROR
        and metrics.removed_energy_ratio <= MAX_REMOVED_ENERGY_RATIO
    )


def normalize_minimum_taps(value: object, input_taps: int, maximum_taps: int) -> int:
    """Normalize only integer inputs; never truncate or coerce user values."""
    is_positive_integer = (isinstance(value, Integral)
                           and not isinstance(value, (bool, np.bool_)) and value > 0)
    minimum = int(value) if is_positive_integer else 1
    if minimum % 2 == 0:
        minimum += 1
    if minimum > input_taps or minimum > maximum_taps:
        raise ValueError("minimum taps exceed input length or maximum taps")
    return minimum


def minimum_candidate_ranges(start: int, stop: int, count: int,
                             minimum: int, symmetric: bool) -> list[tuple[int, int]]:
    """Return source-only ranges containing the candidate, without cropping."""
    if stop - start >= minimum:
        return [(start, stop)]
    extra = minimum - (stop - start)
    if symmetric:
        side = extra // 2
        if extra % 2 or start < side or count - stop < side:
            return []
        return [(start - side, stop + side)]
    # Evaluate balanced expansion and both feasible boundary placements.
    lower = max(0, stop - minimum)
    upper = min(start, count - minimum)
    if lower > upper:
        return []
    balanced = min(upper, max(lower, start - extra // 2))
    return [(left, left + minimum) for left in sorted({lower, balanced, upper})]


def plan_adaptive_crop(
    coefficients: np.ndarray,
    *,
    minimum_taps: object = 1,
    maximum_taps: int | None = None,
    linear_phase: bool | None = None,
) -> AdaptiveCropPlan:
    """Select the shortest passing unwindowed support using existing criteria."""
    original = _validated(coefficients)
    count = original.size
    maximum = count if maximum_taps is None else int(maximum_taps)
    if maximum < 1:
        raise ValueError("maximum taps must be positive")
    minimum = normalize_minimum_taps(minimum_taps, count, maximum)
    if linear_phase is None:
        scale = max(float(np.max(np.abs(original))), 1.0)
        linear_phase = bool(np.max(np.abs(original - original[::-1])) <= scale * 1e-10)
    left = _side_candidates(original)
    right = _side_candidates(original[::-1])
    if linear_phase:
        by_left = {item.cut_samples: item for item in left}
        by_right = {item.cut_samples: item for item in right}
        pairs = [(by_left[k], by_right[k]) for k in sorted(set(by_left) & set(by_right))]
    else:
        pairs = list(product(left, right))
    accepted = []
    total_energy = max(float(np.dot(original, original)), 1e-300)
    for l_item, r_item in pairs:
        start, stop = l_item.cut_samples, count - r_item.cut_samples
        if (stop - start) % 2 == 0:
            if stop < count:
                stop += 1
            elif start > 0:
                start -= 1
            elif abs(original[0]) <= abs(original[-1]):
                start += 1
            else:
                stop -= 1
        if not 0 < stop - start <= maximum or (start == 0 and stop == count):
            continue
        original_length = stop - start
        for start, stop in minimum_candidate_ranges(start, stop, count, minimum, linear_phase):
            minimum_expanded = stop - start > original_length
            # A view is used only for evaluation. No output FIR is constructed here.
            metrics = _metrics(original, original[start:stop],
                               centered_shift_samples=(count - stop - start) / 2.0)
            removed = (float(np.dot(original[:start], original[:start]))
                       + float(np.dot(original[stop:], original[stop:]))) / total_energy
            metrics = AdaptiveCropMetrics(metrics.gain_error_db, metrics.phase_error_deg,
                                          metrics.complex_error, removed)
            safe = all(item.source != "dip" or item.endpoint_db <= MAX_DIP_ENDPOINT_DB
                       for item in (l_item, r_item))
            if safe and _passes(metrics):
                accepted.append(AdaptiveCropPlan(count, start, stop, True, metrics,
                                                l_item.source, r_item.source,
                                                len(pairs), _digest(original), minimum_taps, minimum,
                                                minimum_expanded))
    if accepted:
        return min(accepted, key=lambda plan: (plan.taps, plan.metrics.gain_error_db,
                                              plan.metrics.complex_error))
    return AdaptiveCropPlan(count, 0, count, False, _metrics(original, original),
                            candidate_count=len(pairs), input_digest=_digest(original),
                            requested_minimum_taps=minimum_taps, effective_minimum_taps=minimum)


def execute_crop(coefficients: np.ndarray, plan: AdaptiveCropPlan,
                 *, start: int | None = None, stop: int | None = None) -> np.ndarray:
    """Materialize a range once from its verified original real coefficients."""
    source = _validated(coefficients)
    first = plan.start if start is None else int(start)
    last = plan.stop if stop is None else int(stop)
    if source.size != plan.input_taps or _digest(source) != plan.input_digest:
        raise ValueError("crop plan does not match its input FIR")
    if not 0 <= first <= plan.start < plan.stop <= last <= source.size:
        raise ValueError("crop range must contain the Adaptive support")
    return source[first:last].copy()


def crop_result(coefficients: np.ndarray, plan: AdaptiveCropPlan,
                start: int, stop: int, output: np.ndarray,
                *, window_enabled: bool = False) -> AdaptiveCropResult:
    """Record final coefficients and their external absolute time origin."""
    source = _validated(coefficients)
    metrics = _metrics(source, output,
                       centered_shift_samples=(source.size - stop - start) / 2.0)
    removed = (float(np.dot(source[:start], source[:start]))
               + float(np.dot(source[stop:], source[stop:]))) / max(float(np.dot(source, source)), 1e-300)
    metrics = AdaptiveCropMetrics(metrics.gain_error_db, metrics.phase_error_deg,
                                  metrics.complex_error, removed)
    return AdaptiveCropResult(
        output, plan.adopted, "accepted" if plan.adopted else "no_candidate_passed",
        source.size, plan.taps, output.size, start, source.size - stop,
        0, 0, "external_delay" if start else "none", float(start),
        plan.source_left, plan.source_right, metrics,
        original_digest=plan.input_digest, final_digest=_digest(output),
        candidate_count=plan.candidate_count,
        window_mode="cosine_taper" if window_enabled else "none",
        base_candidate_taps=plan.taps, base_left_removed=plan.start,
        base_right_removed=source.size-plan.stop,
        protection_length_ratio=output.size / plan.taps,
        requested_minimum_taps=(int(plan.requested_minimum_taps)
                                if isinstance(plan.requested_minimum_taps, Integral)
                                else repr(plan.requested_minimum_taps)),
        effective_minimum_taps=plan.effective_minimum_taps,
        minimum_expanded=plan.minimum_expanded,
    )


def adaptive_crop_fir(
    coefficients: np.ndarray,
    *,
    minimum_taps: object = 1,
    maximum_taps: int | None = None,
    linear_phase: bool | None = None,
    timing_mode: Literal["embedded_zero_padding", "external_delay"] = "external_delay",
    output_window_enabled: bool = False,
) -> AdaptiveCropResult:
    """Compatibility facade for the common range/execute/taper pipeline.

    Legacy embedded timing requests now receive explicit external delay;
    padding is never used to compensate a removed time origin.
    """
    if timing_mode not in {"embedded_zero_padding", "external_delay"}:
        raise ValueError("unsupported timing mode")
    from .output_tap_policy import finalize_adaptive_fir
    return finalize_adaptive_fir(
        coefficients, minimum_taps=minimum_taps, maximum_taps=maximum_taps, linear_phase=linear_phase,
        output_window_enabled=output_window_enabled,
    ).adaptive
