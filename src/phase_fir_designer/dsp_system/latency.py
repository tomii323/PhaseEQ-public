from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal

from ..config import IIRFilter
from ..iir import iir_frequency_response


LatencyConfidence = Literal["high", "medium", "low"]
LatencyProvenance = Literal[
    "exact_linear_phase",
    "coefficient_group_delay",
    "assumed_common_path",
    "unknown",
]


@dataclass(frozen=True)
class FIRLatencyEstimate:
    scalar_delay_ms: float
    group_delay_ms: np.ndarray
    symmetric: bool
    effective_support_taps: int
    provenance: LatencyProvenance
    confidence: LatencyConfidence


@dataclass(frozen=True)
class IIRLatencyEstimate:
    scalar_delay_ms: float
    group_delay_ms: np.ndarray
    provenance: LatencyProvenance
    confidence: LatencyConfidence


@dataclass(frozen=True)
class WayLatencyEstimate:
    fir: FIRLatencyEstimate
    iir: IIRLatencyEstimate
    fixed_processing_delay_ms: float
    filter_delay_ms: float
    total_delay_ms: float
    active_frequency_hz: tuple[float, float] | None
    common_processing_cancels: bool


def effective_fir_support_taps(coefficients: np.ndarray, *, energy_fraction: float = 0.999) -> int:
    """Return the shortest contiguous interval containing the requested FIR energy."""

    values = np.asarray(coefficients, dtype=float).ravel()
    if values.size == 0:
        return 0
    energy = np.square(np.where(np.isfinite(values), values, 0.0))
    total = float(np.sum(energy))
    if total <= 1e-30:
        return 1
    fraction = float(np.clip(energy_fraction, 0.0, 1.0))
    target = total * fraction
    cumulative = np.concatenate(([0.0], np.cumsum(energy)))
    best = values.size
    right = 0
    for left in range(values.size):
        right = max(right, left + 1)
        while right <= values.size and cumulative[right] - cumulative[left] < target:
            right += 1
        if right > values.size:
            break
        best = min(best, right - left)
    return max(1, int(best))


def _group_delay_from_response(response: np.ndarray, frequency_hz: np.ndarray) -> np.ndarray:
    phase = np.unwrap(np.angle(np.asarray(response, dtype=complex)))
    frequency = np.asarray(frequency_hz, dtype=float)
    if phase.size < 2:
        return np.zeros_like(frequency)
    angular_frequency = 2.0 * np.pi * frequency
    with np.errstate(divide="ignore", invalid="ignore"):
        delay_seconds = -np.gradient(phase, angular_frequency, edge_order=1)
    return np.nan_to_num(delay_seconds * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)


def _active_mask(response: np.ndarray, *, floor_db: float = -40.0) -> np.ndarray:
    magnitude = np.abs(np.asarray(response, dtype=complex))
    if magnitude.size == 0 or not np.any(np.isfinite(magnitude)):
        return np.zeros(magnitude.shape, dtype=bool)
    peak = max(float(np.nanmax(magnitude)), 1e-15)
    return np.isfinite(magnitude) & (magnitude >= peak * 10.0 ** (float(floor_db) / 20.0))


def estimate_fir_latency(
    coefficients: np.ndarray,
    sample_rate: int,
    frequency_hz: np.ndarray,
    *,
    symmetry_tolerance: float = 1e-7,
) -> FIRLatencyEstimate:
    values = np.asarray(coefficients, dtype=float).ravel()
    frequency = np.asarray(frequency_hz, dtype=float)
    rate = int(sample_rate)
    if rate <= 0 or values.size == 0:
        raise ValueError("FIR latency requires coefficients and a positive sample rate")
    peak = max(float(np.max(np.abs(values))), 1e-15)
    symmetric_error = float(np.max(np.abs(values - values[::-1]))) / peak
    antisymmetric_error = float(np.max(np.abs(values + values[::-1]))) / peak
    symmetric = min(symmetric_error, antisymmetric_error) <= float(symmetry_tolerance)
    if symmetric:
        scalar = (values.size - 1) * 500.0 / float(rate)
        group_delay = np.full_like(frequency, scalar, dtype=float)
        provenance: LatencyProvenance = "exact_linear_phase"
        confidence: LatencyConfidence = "high"
    else:
        clipped = np.clip(frequency, 0.0, rate / 2.0)
        _w, response = signal.freqz(values, worN=clipped, fs=rate)
        group_delay = _group_delay_from_response(response, clipped)
        mask = _active_mask(response)
        scalar = float(np.nanmedian(group_delay[mask])) if np.any(mask) else 0.0
        provenance = "coefficient_group_delay"
        confidence = "medium"
    return FIRLatencyEstimate(
        scalar_delay_ms=float(scalar),
        group_delay_ms=group_delay,
        symmetric=bool(symmetric),
        effective_support_taps=effective_fir_support_taps(values),
        provenance=provenance,
        confidence=confidence,
    )


def estimate_iir_latency(
    filters: tuple[IIRFilter, ...] | list[IIRFilter],
    sample_rate: int,
    frequency_hz: np.ndarray,
    *,
    active_response: np.ndarray | None = None,
    floor_db: float = -40.0,
) -> IIRLatencyEstimate:
    frequency = np.asarray(frequency_hz, dtype=float)
    enabled = tuple(item for item in filters if item.enabled)
    if not enabled:
        return IIRLatencyEstimate(
            scalar_delay_ms=0.0,
            group_delay_ms=np.zeros_like(frequency),
            provenance="assumed_common_path",
            confidence="high",
        )
    response = iir_frequency_response(enabled, int(sample_rate), frequency)
    group_delay = _group_delay_from_response(response, frequency)
    mask = _active_mask(response if active_response is None else active_response, floor_db=floor_db)
    scalar = float(np.nanmedian(group_delay[mask])) if np.any(mask) else 0.0
    return IIRLatencyEstimate(
        scalar_delay_ms=scalar,
        group_delay_ms=group_delay,
        provenance="coefficient_group_delay",
        confidence="medium",
    )


def estimate_way_latency(
    coefficients: np.ndarray,
    iir_filters: tuple[IIRFilter, ...] | list[IIRFilter],
    sample_rate: int,
    frequency_hz: np.ndarray,
    *,
    active_response: np.ndarray | None = None,
    fixed_processing_delay_ms: float = 0.0,
    common_processing_cancels: bool = True,
    floor_db: float = -40.0,
) -> WayLatencyEstimate:
    frequency = np.asarray(frequency_hz, dtype=float)
    fir = estimate_fir_latency(coefficients, sample_rate, frequency)
    iir = estimate_iir_latency(
        iir_filters,
        sample_rate,
        frequency,
        active_response=active_response,
        floor_db=floor_db,
    )
    if active_response is None:
        mask = np.ones_like(frequency, dtype=bool)
    else:
        mask = _active_mask(active_response, floor_db=floor_db)
    if np.any(mask):
        combined = fir.group_delay_ms[mask] + iir.group_delay_ms[mask]
        filter_delay = float(np.nanmedian(combined))
        active_range = (float(frequency[mask][0]), float(frequency[mask][-1]))
    else:
        filter_delay = float(fir.scalar_delay_ms + iir.scalar_delay_ms)
        active_range = None
    fixed = float(fixed_processing_delay_ms)
    return WayLatencyEstimate(
        fir=fir,
        iir=iir,
        fixed_processing_delay_ms=fixed,
        filter_delay_ms=filter_delay,
        total_delay_ms=filter_delay + fixed,
        active_frequency_hz=active_range,
        common_processing_cancels=bool(common_processing_cancels),
    )
