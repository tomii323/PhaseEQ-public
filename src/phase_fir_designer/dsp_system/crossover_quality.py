from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DistortionProfile:
    frequency_hz: tuple[float, ...]
    h2_db: tuple[float, ...] = ()
    h3_db: tuple[float, ...] = ()
    thd_percent: tuple[float, ...] = ()
    measurement_level_db_spl: float | None = None
    source: str = "Imported"

    def validate(self) -> None:
        frequency = np.asarray(self.frequency_hz, dtype=float)
        if frequency.size < 2 or np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
            raise ValueError("distortion profile requires at least two positive frequencies")
        if np.any(np.diff(frequency) <= 0.0):
            raise ValueError("distortion profile frequencies must be strictly increasing")
        for name, values in (("h2_db", self.h2_db), ("h3_db", self.h3_db), ("thd_percent", self.thd_percent)):
            if values and len(values) != frequency.size:
                raise ValueError(f"{name} length must match frequency_hz")
            if values and np.any(~np.isfinite(np.asarray(values, dtype=float))):
                raise ValueError(f"{name} must contain finite values")
        if not self.h2_db and not self.h3_db and not self.thd_percent:
            raise ValueError("distortion profile requires H2, H3, or THD data")


@dataclass(frozen=True)
class DistortionEvaluation:
    evaluated: bool
    peak_harmonic_db: float | None
    p95_harmonic_db: float | None
    area_db_oct: float | None
    peak_thd_percent: float | None
    source: str | None


@dataclass(frozen=True)
class WaveletPeakEvaluation:
    evaluated: bool
    peak_db: float | None
    p95_db: float | None
    positive_area_db_oct: float | None
    confidence: float
    source_type: str | None


@dataclass(frozen=True)
class CrossoverQualityEvaluation:
    boundary_id: str
    frequency_hz: tuple[float, float] | None
    wavelet: WaveletPeakEvaluation
    lower_distortion: DistortionEvaluation
    upper_distortion: DistortionEvaluation
    warnings: tuple[str, ...] = ()


def _log_area(values: np.ndarray, frequency_hz: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.trapezoid(values, x=np.log2(frequency_hz)))


def _interp_optional(values: tuple[float, ...], source_frequency: np.ndarray, frequency: np.ndarray) -> np.ndarray | None:
    if not values:
        return None
    source = np.asarray(values, dtype=float)
    output = np.full_like(frequency, np.nan, dtype=float)
    positive = np.isfinite(frequency) & (frequency > 0.0)
    output[positive] = np.interp(
        np.log2(frequency[positive]), np.log2(source_frequency), source,
    )
    return output


def evaluate_distortion_profile(
    profile: DistortionProfile | None,
    frequency_hz: np.ndarray,
    fundamental_level_db: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> DistortionEvaluation:
    if profile is None:
        return DistortionEvaluation(False, None, None, None, None, None)
    profile.validate()
    frequency = np.asarray(frequency_hz, dtype=float)
    fundamental = np.asarray(fundamental_level_db, dtype=float)
    if frequency.shape != fundamental.shape:
        raise ValueError("frequency and fundamental level must have the same shape")
    active = np.ones_like(frequency, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    active &= np.isfinite(frequency) & (frequency > 0.0) & np.isfinite(fundamental)
    if np.count_nonzero(active) < 2:
        return DistortionEvaluation(False, None, None, None, None, profile.source)
    source_frequency = np.asarray(profile.frequency_hz, dtype=float)
    harmonic_levels: list[np.ndarray] = []
    for values in (profile.h2_db, profile.h3_db):
        relative = _interp_optional(values, source_frequency, frequency)
        if relative is not None:
            harmonic_levels.append(fundamental + relative)
    if harmonic_levels:
        energy = np.sum([10.0 ** (level / 10.0) for level in harmonic_levels], axis=0)
        combined = 10.0 * np.log10(np.maximum(energy, 1e-30))
        selected = combined[active]
        peak = float(np.max(selected))
        p95 = float(np.percentile(selected, 95.0))
        relative_to_peak = np.maximum(selected - float(np.max(fundamental[active])), 0.0)
        area = _log_area(relative_to_peak, frequency[active])
    else:
        peak = p95 = area = None
    thd = _interp_optional(profile.thd_percent, source_frequency, frequency)
    peak_thd = float(np.max(thd[active])) if thd is not None else None
    return DistortionEvaluation(True, peak, p95, area, peak_thd, profile.source)


def evaluate_wavelet_peak_profile(
    profile: Any | None,
    frequency_hz: np.ndarray,
    *,
    reference_db: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> WaveletPeakEvaluation:
    if profile is None:
        return WaveletPeakEvaluation(False, None, None, None, 0.0, None)
    source_frequency = np.asarray(profile.frequency_hz, dtype=float)
    source_level = np.asarray(profile.peak_level_db, dtype=float)
    frequency = np.asarray(frequency_hz, dtype=float)
    if source_frequency.size < 2 or source_frequency.size != source_level.size:
        return WaveletPeakEvaluation(False, None, None, None, 0.0, getattr(profile, "source_type", None))
    level = np.full_like(frequency, np.nan, dtype=float)
    positive = np.isfinite(frequency) & (frequency > 0.0)
    level[positive] = np.interp(
        np.log2(frequency[positive]), np.log2(source_frequency), source_level,
    )
    active = np.ones_like(frequency, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    active &= np.isfinite(level) & np.isfinite(frequency) & (frequency > 0.0)
    if np.count_nonzero(active) < 2:
        return WaveletPeakEvaluation(False, None, None, None, 0.0, getattr(profile, "source_type", None))
    selected = level[active]
    if reference_db is None:
        baseline = float(np.percentile(selected, 50.0))
        excess = np.maximum(selected - baseline, 0.0)
    else:
        reference = np.asarray(reference_db, dtype=float)
        if reference.shape != frequency.shape:
            raise ValueError("Wavelet reference must match the frequency axis")
        excess = np.maximum(selected - reference[active], 0.0)
    confidence = float(np.clip(getattr(profile, "reconstruction_confidence", 1.0), 0.0, 1.0))
    return WaveletPeakEvaluation(
        evaluated=True,
        peak_db=float(np.max(selected)),
        p95_db=float(np.percentile(selected, 95.0)),
        positive_area_db_oct=_log_area(excess, frequency[active]),
        confidence=confidence,
        source_type=str(getattr(profile, "source_type", "wavelet_1_3_oct")),
    )


def distortion_profile_from_columns(columns: dict[str, list[float] | tuple[float, ...]], *, source: str = "Imported") -> DistortionProfile:
    normalized = {str(key).strip().casefold(): value for key, value in columns.items()}
    frequency = normalized.get("frequency_hz", normalized.get("frequency"))
    if frequency is None:
        raise ValueError("distortion data requires frequency_hz or frequency")
    profile = DistortionProfile(
        frequency_hz=tuple(float(value) for value in frequency),
        h2_db=tuple(float(value) for value in normalized.get("h2_db", ())),
        h3_db=tuple(float(value) for value in normalized.get("h3_db", ())),
        thd_percent=tuple(float(value) for value in normalized.get("thd_percent", normalized.get("thd", ()))),
        source=str(source or "Imported"),
    )
    profile.validate()
    return profile
