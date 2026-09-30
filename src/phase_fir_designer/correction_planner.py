from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from wavelet_analysis import WaveletMap
from .config import CandidateWaveletProfile


@dataclass(frozen=True)
class WaveletFeature:
    frequency_hz: np.ndarray
    confidence: np.ndarray
    reflection_index: np.ndarray
    spread_ms: np.ndarray
    weight: np.ndarray


@dataclass(frozen=True)
class GainCorrectionPlan:
    correction_db: np.ndarray
    candidate_mask: np.ndarray
    candidate_weight: np.ndarray
    rejected_mask: np.ndarray
    rejected_reason: np.ndarray
    f_min: float
    f_max: float
    accepted_count: int
    rejected_count: int
    max_abs_correction_db: float
    groups: tuple["CorrectionCandidateGroup", ...] = ()


@dataclass(frozen=True)
class CorrectionCandidateGroup:
    f_min: float
    f_max: float
    kind: str
    peak_frequency_hz: float
    peak_value: float
    point_count: int
    suggested_q: float


@dataclass(frozen=True)
class PhaseCorrectionPlan:
    correction_deg: np.ndarray
    candidate_mask: np.ndarray
    rejected_mask: np.ndarray
    rejected_reason: np.ndarray
    f_min: float
    f_max: float
    accepted_count: int
    rejected_count: int
    max_abs_correction_deg: float
    groups: tuple[CorrectionCandidateGroup, ...] = ()


def wavelet_feature_from_map(wavelet_map: WaveletMap) -> WaveletFeature:
    frequency = np.asarray(wavelet_map.frequency_hz, dtype=float)
    confidence = _metric_or_default(wavelet_map.confidence, frequency.size, 1.0)
    reflection = _metric_or_default(wavelet_map.reflection_index, frequency.size, 0.0)
    spread = _metric_or_default(wavelet_map.spread_ms, frequency.size, 0.0)
    weight = (
        np.clip(np.nan_to_num(confidence, nan=0.0), 0.0, 1.0)
        * (1.0 / (1.0 + np.maximum(np.nan_to_num(reflection, nan=1.0), 0.0)))
        * (1.0 / (1.0 + np.maximum(np.nan_to_num(spread, nan=10.0), 0.0) / 8.0))
    )
    return WaveletFeature(
        frequency_hz=frequency,
        confidence=confidence,
        reflection_index=reflection,
        spread_ms=spread,
        weight=np.clip(weight, 0.0, 1.0),
    )


def wavelet_feature_from_profile(profile: CandidateWaveletProfile | None) -> WaveletFeature | None:
    if profile is None:
        return None
    frequency = np.asarray(profile.frequency_hz, dtype=float)
    weight = np.asarray(profile.weight, dtype=float)
    valid = np.isfinite(frequency) & np.isfinite(weight)
    if np.count_nonzero(valid) < 2:
        return None
    frequency = frequency[valid]
    weight = np.clip(weight[valid], 0.0, 1.0)
    return WaveletFeature(
        frequency_hz=frequency,
        confidence=weight,
        reflection_index=np.zeros_like(weight),
        spread_ms=np.zeros_like(weight),
        weight=weight,
    )


def plan_auto_gain_correction(
    frequency: np.ndarray,
    correction_db: np.ndarray,
    *,
    f_min: float,
    f_max: float,
    min_delta_db: float = 0.0,
    wavelet_feature: WaveletFeature | None = None,
    min_wavelet_weight: float = 0.0,
) -> GainCorrectionPlan:
    # Recipe v1 permanently closes the former Wavelet candidate-gating path.
    # Keep the arguments readable for old Project/Recipe compatibility, but
    # never let either value affect Auto Gain candidate selection or strength.
    del wavelet_feature, min_wavelet_weight
    freq = np.asarray(frequency, dtype=float)
    correction = np.asarray(correction_db, dtype=float)
    if freq.shape != correction.shape:
        raise ValueError("frequency and correction_db must have the same shape")

    finite = np.isfinite(freq) & np.isfinite(correction)
    range_mask = finite & (freq >= float(f_min)) & (freq <= float(f_max))
    delta_threshold = max(float(min_delta_db), 0.0)
    amplitude_mask = np.abs(correction) >= delta_threshold
    weight = np.ones_like(correction, dtype=float)
    wavelet_mask = np.ones_like(correction, dtype=bool)
    candidate = range_mask & amplitude_mask & wavelet_mask
    rejected = range_mask & ~candidate
    reason = np.full(correction.shape, "", dtype=object)
    reason[range_mask & ~amplitude_mask] = "below delta"
    reason[range_mask & amplitude_mask & ~wavelet_mask] = "low wavelet"
    apply_weight = np.ones_like(correction)
    planned = np.zeros_like(correction, dtype=float)
    planned[candidate] = correction[candidate] * apply_weight[candidate]
    accepted = int(np.count_nonzero(candidate))
    rejected_count = int(np.count_nonzero(rejected))
    max_abs = float(np.nanmax(np.abs(planned[candidate]))) if accepted else 0.0
    return GainCorrectionPlan(
        correction_db=planned,
        candidate_mask=candidate,
        candidate_weight=apply_weight,
        rejected_mask=rejected,
        rejected_reason=reason,
        f_min=float(f_min),
        f_max=float(f_max),
        accepted_count=accepted,
        rejected_count=rejected_count,
        max_abs_correction_db=max_abs,
        groups=_candidate_groups(freq, planned, candidate),
    )


def replace_gain_correction_values(
    plan: GainCorrectionPlan,
    frequency: np.ndarray,
    correction_db: np.ndarray,
) -> GainCorrectionPlan:
    freq = np.asarray(frequency, dtype=float)
    correction = np.asarray(correction_db, dtype=float)
    if freq.shape != correction.shape:
        raise ValueError("frequency and correction_db must have the same shape")
    candidate = np.asarray(plan.candidate_mask, dtype=bool)
    max_abs = float(np.nanmax(np.abs(correction[candidate]))) if np.any(candidate) else 0.0
    return replace(
        plan,
        correction_db=correction,
        max_abs_correction_db=max_abs,
        groups=_candidate_groups(freq, correction, candidate),
    )


def plan_auto_phase_correction(
    frequency: np.ndarray,
    correction_deg: np.ndarray,
    *,
    f_min: float,
    f_max: float,
    min_delta_deg: float = 0.0,
) -> PhaseCorrectionPlan:
    freq = np.asarray(frequency, dtype=float)
    correction = np.asarray(correction_deg, dtype=float)
    if freq.shape != correction.shape:
        raise ValueError("frequency and correction_deg must have the same shape")
    finite = np.isfinite(freq) & np.isfinite(correction)
    range_mask = finite & (freq >= float(f_min)) & (freq <= float(f_max))
    amplitude_mask = np.abs(correction) >= max(float(min_delta_deg), 0.0)
    candidate = range_mask & amplitude_mask
    rejected = range_mask & ~candidate
    reason = np.full(correction.shape, "", dtype=object)
    reason[rejected] = "below delta"
    planned = np.zeros_like(correction, dtype=float)
    planned[candidate] = correction[candidate]
    accepted = int(np.count_nonzero(candidate))
    rejected_count = int(np.count_nonzero(rejected))
    max_abs = float(np.nanmax(np.abs(planned[candidate]))) if accepted else 0.0
    return PhaseCorrectionPlan(
        correction_deg=planned,
        candidate_mask=candidate,
        rejected_mask=rejected,
        rejected_reason=reason,
        f_min=float(f_min),
        f_max=float(f_max),
        accepted_count=accepted,
        rejected_count=rejected_count,
        max_abs_correction_deg=max_abs,
        groups=_candidate_groups(freq, planned, candidate),
    )


def replace_phase_correction_values(
    plan: PhaseCorrectionPlan,
    frequency: np.ndarray,
    correction_deg: np.ndarray,
) -> PhaseCorrectionPlan:
    freq = np.asarray(frequency, dtype=float)
    correction = np.asarray(correction_deg, dtype=float)
    if freq.shape != correction.shape:
        raise ValueError("frequency and correction_deg must have the same shape")
    candidate = np.asarray(plan.candidate_mask, dtype=bool)
    max_abs = float(np.nanmax(np.abs(correction[candidate]))) if np.any(candidate) else 0.0
    return replace(
        plan,
        correction_deg=correction,
        max_abs_correction_deg=max_abs,
        groups=_candidate_groups(freq, correction, candidate),
    )


def _candidate_groups(
    frequency: np.ndarray,
    correction: np.ndarray,
    candidate_mask: np.ndarray,
) -> tuple[CorrectionCandidateGroup, ...]:
    indices = np.flatnonzero(candidate_mask)
    if indices.size == 0:
        return ()
    groups: list[CorrectionCandidateGroup] = []
    split_points = np.flatnonzero(np.diff(indices) > 1) + 1
    contiguous_parts = np.split(indices, split_points)
    parts: list[np.ndarray] = []
    for contiguous in contiguous_parts:
        if contiguous.size <= 1:
            parts.append(contiguous)
            continue
        values = correction[contiguous]
        signs = np.sign(values)
        sign_splits = np.flatnonzero((signs[1:] != signs[:-1]) & (signs[1:] != 0) & (signs[:-1] != 0)) + 1
        parts.extend(np.split(contiguous, sign_splits))
    for part in parts:
        if part.size == 0:
            continue
        values = correction[part]
        peak_local = int(np.nanargmax(np.abs(values)))
        peak_idx = int(part[peak_local])
        peak_value = float(correction[peak_idx])
        if np.nanmedian(values) > 0:
            kind = "boost"
        elif np.nanmedian(values) < 0:
            kind = "cut"
        else:
            kind = "mixed"
        groups.append(
            CorrectionCandidateGroup(
                f_min=float(frequency[int(part[0])]),
                f_max=float(frequency[int(part[-1])]),
                kind=kind,
                peak_frequency_hz=float(frequency[peak_idx]),
                peak_value=peak_value,
                point_count=int(part.size),
                suggested_q=_suggested_q(float(frequency[int(part[0])]), float(frequency[int(part[-1])]), float(frequency[peak_idx])),
            )
        )
    return tuple(groups)


def _suggested_q(f_min: float, f_max: float, peak_frequency: float) -> float:
    bandwidth = max(float(f_max) - float(f_min), 1e-9)
    if f_max <= f_min:
        return 4.0
    return float(np.clip(float(peak_frequency) / bandwidth, 0.2, 20.0))


def _metric_or_default(values: np.ndarray | None, size: int, default: float) -> np.ndarray:
    if values is None:
        return np.full(size, float(default), dtype=float)
    metric = np.asarray(values, dtype=float)
    if metric.shape != (size,):
        return np.full(size, float(default), dtype=float)
    return metric


def _wavelet_weight_on_axis(frequency: np.ndarray, feature: WaveletFeature) -> np.ndarray:
    source_frequency = np.asarray(feature.frequency_hz, dtype=float)
    source_weight = np.asarray(feature.weight, dtype=float)
    valid = np.isfinite(source_frequency) & np.isfinite(source_weight) & (source_frequency > 0)
    if np.count_nonzero(valid) < 2:
        return np.ones_like(frequency, dtype=float)
    order = np.argsort(source_frequency[valid])
    source_frequency = source_frequency[valid][order]
    source_weight = np.clip(source_weight[valid][order], 0.0, 1.0)
    return np.interp(
        np.asarray(frequency, dtype=float),
        source_frequency,
        source_weight,
        left=float(source_weight[0]),
        right=float(source_weight[-1]),
    )


def _relative_wavelet_threshold(weight: np.ndarray, reference_mask: np.ndarray, threshold: float) -> float:
    threshold = float(np.clip(threshold, 0.0, 1.0))
    if threshold <= 0.0:
        return 0.0
    reference = np.asarray(weight, dtype=float)[np.asarray(reference_mask, dtype=bool)]
    reference = reference[np.isfinite(reference)]
    if reference.size == 0:
        return threshold
    max_weight = float(np.max(reference))
    if max_weight <= 0.0:
        return 1.0
    return max_weight * threshold


def _reference_wavelet_max(weight: np.ndarray, reference_mask: np.ndarray) -> float:
    reference = np.asarray(weight, dtype=float)[np.asarray(reference_mask, dtype=bool)]
    reference = reference[np.isfinite(reference)]
    if reference.size == 0:
        return 0.0
    return float(np.max(reference))


def _soft_wavelet_apply_weight(weight: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    max_weight = _reference_wavelet_max(weight, reference_mask)
    if max_weight < 0.05:
        return np.ones_like(weight, dtype=float)
    normalized = np.clip(np.asarray(weight, dtype=float) / max_weight, 0.0, 1.0)
    return 0.60 + 0.40 * normalized
