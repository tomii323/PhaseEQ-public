"""Coefficient-driven high-precision FIR cropping.

The module is UI independent.  It analyses both ends with the same routine,
keeps the input immutable, and returns an explicit timing representation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
from typing import Literal

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import find_peaks

from fir_output_window import apply_output_window


THRESHOLD_DB = -120.0
ACTIVE_FLOOR_DB = -80.0
MAX_GAIN_ERROR_DB = 0.05
MAX_PHASE_ERROR_DEG = 0.5
MAX_COMPLEX_ERROR = 0.01
MAX_REMOVED_ENERGY_RATIO = 1e-6
MAX_DIP_ENDPOINT_DB = -40.0
ALGORITHM_VERSION = "adaptive-fir-crop-v1"


@dataclass(frozen=True)
class SideCandidate:
    cut_samples: int
    source: Literal["none", "threshold", "dip"]
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


def adaptive_crop_fir(
    coefficients: np.ndarray,
    *,
    maximum_taps: int | None = None,
    linear_phase: bool | None = None,
    timing_mode: Literal["embedded_zero_padding", "external_delay"] = "embedded_zero_padding",
    output_window_enabled: bool = False,
) -> AdaptiveCropResult:
    """Return the shortest passing high-precision candidate.

    ``embedded_zero_padding`` pads the side with more removed samples so the
    original tap centre remains the centre of the shortened FIR.
    """
    original = _validated(coefficients)
    count = original.size
    maximum = count if maximum_taps is None else max(1, int(maximum_taps))
    if linear_phase is None:
        scale = max(float(np.max(np.abs(original))), 1.0)
        linear_phase = bool(np.max(np.abs(original - original[::-1])) <= scale * 1e-10)

    left = _side_candidates(original)
    right = _side_candidates(original[::-1])
    pairs = []
    if linear_phase:
        by_left = {item.cut_samples: item for item in left}
        by_right = {item.cut_samples: item for item in right}
        pairs = [(by_left[k], by_right[k]) for k in sorted(set(by_left) & set(by_right))]
    else:
        pairs = list(product(left, right))

    reference = apply_output_window(original, output_window_enabled)
    accepted: list[tuple[int, float, float, SideCandidate, SideCandidate, np.ndarray, AdaptiveCropMetrics, int, int]] = []
    total_energy = max(float(np.dot(original, original)), 1e-300)
    for left_item, right_item in pairs:
        l_cut, r_cut = left_item.cut_samples, right_item.cut_samples
        if l_cut + r_cut <= 0 or l_cut + r_cut >= count:
            continue
        direct = original[l_cut:count-r_cut]
        pad_left = pad_right = 0
        if timing_mode == "embedded_zero_padding":
            pad_left = max(l_cut - r_cut, 0)
            pad_right = max(r_cut - l_cut, 0)
        candidate = np.pad(direct, (pad_left, pad_right))
        if candidate.size > maximum:
            continue
        candidate = apply_output_window(candidate, output_window_enabled)
        metrics = _metrics(
            reference,
            candidate,
            centered_shift_samples=(
                (r_cut - l_cut) / 2.0 if timing_mode == "external_delay" else 0.0
            ),
        )
        removed_energy = (
            float(np.dot(original[:l_cut], original[:l_cut]))
            + float(np.dot(original[count-r_cut:], original[count-r_cut:]))
        ) / total_energy
        metrics = AdaptiveCropMetrics(
            metrics.gain_error_db, metrics.phase_error_deg,
            metrics.complex_error, removed_energy,
        )
        endpoints_are_safe = all(
            item.source != "dip" or item.endpoint_db <= MAX_DIP_ENDPOINT_DB
            for item in (left_item, right_item)
        )
        if endpoints_are_safe and _passes(metrics):
            accepted.append((
                candidate.size, metrics.gain_error_db, metrics.complex_error,
                left_item, right_item, candidate, metrics, pad_left, pad_right,
            ))

    if not accepted:
        unchanged = apply_output_window(original, output_window_enabled)
        metrics = _metrics(reference, unchanged)
        return AdaptiveCropResult(
            unchanged, False, "no_candidate_passed", count, count, count,
            0, 0, 0, 0, "none", 0.0, "none", "none", metrics,
            original_digest=_digest(original), final_digest=_digest(unchanged),
            candidate_count=len(pairs),
            window_mode="cosine_taper" if output_window_enabled else "none",
        )

    selected = min(accepted, key=lambda item: (item[0], item[1], item[2]))
    final_taps, _gain, _complex, left_item, right_item, candidate, metrics, pad_left, pad_right = selected
    crop_delay = float(left_item.cut_samples) if timing_mode == "external_delay" else 0.0
    return AdaptiveCropResult(
        candidate, True, "accepted", count,
        count - left_item.cut_samples - right_item.cut_samples,
        final_taps, left_item.cut_samples, right_item.cut_samples,
        pad_left, pad_right, timing_mode, crop_delay,
        left_item.source, right_item.source, metrics,
        original_digest=_digest(original), final_digest=_digest(candidate),
        candidate_count=len(pairs),
        window_mode="cosine_taper" if output_window_enabled else "none",
    )
