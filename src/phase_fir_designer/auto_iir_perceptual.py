from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import signal

from wavelet_analysis import WaveletSettings, complex_morlet_scalogram
from .config import SpeakerResponse


@dataclass(frozen=True)
class AutoIIRPerceptualProfile:
    """Frequency-dependent peak levels used by the Auto IIR perceptual score.

    Levels may be absolute or relative dB. Auto IIR converts them to a level
    relative to the configured reference percentile before applying its soft
    audibility gate.
    """

    frequency_hz: tuple[float, ...]
    peak_level_db: tuple[float, ...]
    bandwidth_oct: float = 1.0 / 6.0
    source_type: str = "wavelet_1_6_oct"
    reconstruction_confidence: float = 1.0
    phase_available: bool = True
    source_warning: str | None = None


def generate_auto_iir_perceptual_profile(
    speaker: SpeakerResponse,
    sample_rate: int,
    *,
    original_impulse: np.ndarray | None = None,
    original_impulse_sample_rate: int | None = None,
    f_min: float = 20.0,
    f_max: float | None = None,
    bandwidth_oct: float = 1.0 / 6.0,
    frequency_bins: int = 64,
    time_bins: int = 192,
) -> AutoIIRPerceptualProfile:
    """Generate Auto IIR Wavelet peaks from measured IR or reconstructed FRD.

    Phase-bearing FRD is reconstructed from its complex response. Gain-only
    FRD uses the existing minimum-phase reconstruction and is explicitly
    labelled with lower confidence.
    """

    rate = int(sample_rate)
    impulse = None if original_impulse is None else np.asarray(original_impulse, dtype=float).ravel()
    source_type = "original_ir"
    confidence = 1.0
    phase_available = True
    warning = None
    if impulse is not None and impulse.size >= 64 and np.any(np.isfinite(impulse)):
        source_rate = int(original_impulse_sample_rate or rate)
        if source_rate <= 0:
            raise ValueError("original impulse sample rate must be positive")
        if source_rate != rate:
            divisor = int(np.gcd(source_rate, rate))
            impulse = signal.resample_poly(impulse, rate // divisor, source_rate // divisor)
    else:
        # Local import avoids the speaker -> phase_curves -> iir -> perceptual
        # dependency loop while keeping FRD reconstruction in its canonical
        # independent speaker module.
        from .speaker import speaker_response_wavelet_impulse_source

        reconstructed = speaker_response_wavelet_impulse_source(speaker, rate)
        if reconstructed is None:
            raise ValueError("Speaker response could not be reconstructed as an impulse")
        impulse = np.asarray(reconstructed.impulse, dtype=float)
        source_type = reconstructed.source_type
        confidence = float(reconstructed.reconstruction_confidence)
        phase_available = bool(reconstructed.phase_available)
        if source_type == "reconstructed_from_smoothed_response":
            source_type = (
                "reconstructed_from_smoothed_gain_phase"
                if phase_available
                else "reconstructed_from_smoothed_minimum_phase"
            )
        warning = reconstructed.source_warning
    profile = generate_wavelet_peak_profile(
        impulse,
        rate,
        f_min=f_min,
        f_max=f_max,
        bandwidth_oct=bandwidth_oct,
        frequency_bins=frequency_bins,
        time_bins=time_bins,
        source_type=source_type,
    )
    return replace(
        profile,
        reconstruction_confidence=confidence,
        phase_available=phase_available,
        source_warning=warning,
    )


def generate_wavelet_peak_profile(
    impulse: np.ndarray,
    sample_rate: int,
    *,
    f_min: float = 20.0,
    f_max: float | None = None,
    bandwidth_oct: float = 1.0 / 6.0,
    frequency_bins: int = 64,
    time_bins: int = 192,
    dynamic_range_db: float = 60.0,
    mask_floor_db: float = -80.0,
    source_type: str = "original_ir",
) -> AutoIIRPerceptualProfile:
    """Generate a Wavelet peak profile directly from an impulse response.

    ``f_min`` is intended to receive the effective Auto PEQ evaluation lower
    bound. It is not clamped to the former 50 Hz Wavelet display default.
    """

    source = np.asarray(impulse, dtype=float).ravel()
    rate = int(sample_rate)
    if rate <= 0:
        raise ValueError("sample rate must be positive")
    if source.size < 64 or not np.any(np.isfinite(source)):
        raise ValueError("Wavelet peak profile requires at least 64 finite impulse samples")
    lower = float(f_min)
    upper = min(float(f_max if f_max is not None else 20_000.0), rate / 2.0)
    bandwidth = float(bandwidth_oct)
    if not np.isfinite(lower) or lower <= 0.0:
        raise ValueError("f_min must be finite and positive")
    if not np.isfinite(upper) or upper <= lower:
        raise ValueError("f_max must be above f_min and no higher than Nyquist")
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth_oct must be finite and positive")
    duration_ms = source.size * 1000.0 / float(rate)
    effective_time_bins = max(3, int(time_bins))
    if effective_time_bins % 2 == 0:
        effective_time_bins += 1
    settings = WaveletSettings(
        f_min=lower,
        f_max=upper,
        t_min_ms=-duration_ms / 2.0,
        t_max_ms=duration_ms / 2.0,
        bandwidth_oct=bandwidth,
        dynamic_range_db=float(dynamic_range_db),
        frequency_bins=max(2, int(frequency_bins)),
        time_bins=effective_time_bins,
        mask_floor_db=float(mask_floor_db),
    )
    wavelet_map = complex_morlet_scalogram(
        np.where(np.isfinite(source), source, 0.0),
        rate,
        settings,
        label="Auto IIR Peak Profile",
    )
    profile = perceptual_profile_from_wavelet_map(wavelet_map)
    return replace(
        profile,
        bandwidth_oct=bandwidth,
        source_type=str(source_type or "original_ir"),
    )


def perceptual_profile_from_wavelet_map(wavelet_map: Any) -> AutoIIRPerceptualProfile:
    """Build an Auto IIR peak profile from the row maxima of a WaveletMap."""

    frequency = np.asarray(wavelet_map.frequency_hz, dtype=float).ravel()
    level = np.asarray(wavelet_map.level_db, dtype=float)
    if level.ndim != 2 or level.shape[0] != frequency.size:
        raise ValueError("Wavelet level rows must match the frequency axis")
    finite_level = np.where(np.isfinite(level), level, -np.inf)
    peak_level = np.max(finite_level, axis=1)
    peak_level[~np.isfinite(peak_level)] = np.nan
    valid = np.isfinite(frequency) & (frequency > 0.0) & np.isfinite(peak_level)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Wavelet map must contain at least two finite frequency rows")
    order = np.argsort(frequency[valid])
    frequency = frequency[valid][order]
    peak_level = peak_level[valid][order]
    frequency, unique = np.unique(frequency, return_index=True)
    peak_level = peak_level[unique]
    bandwidth = float(getattr(wavelet_map, "bandwidth_oct", 1.0 / 6.0))
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        bandwidth = 1.0 / 3.0
    return AutoIIRPerceptualProfile(
        frequency_hz=tuple(float(value) for value in frequency),
        peak_level_db=tuple(float(value) for value in peak_level),
        bandwidth_oct=bandwidth,
        source_type=str(getattr(wavelet_map, "source_type", "wavelet_1_6_oct")),
    )


def perceptual_level_on_axis(
    profile: AutoIIRPerceptualProfile,
    frequency_hz: np.ndarray,
) -> np.ndarray:
    """Interpolate a peak-level profile on a positive frequency axis."""

    source_frequency = np.asarray(profile.frequency_hz, dtype=float)
    source_level = np.asarray(profile.peak_level_db, dtype=float)
    requested = np.asarray(frequency_hz, dtype=float)
    valid = (
        np.isfinite(source_frequency)
        & (source_frequency > 0.0)
        & np.isfinite(source_level)
    )
    if np.count_nonzero(valid) < 2 or requested.ndim != 1 or np.any(requested <= 0.0):
        raise ValueError("Perceptual profile requires finite positive frequency data")
    order = np.argsort(source_frequency[valid])
    source_frequency = source_frequency[valid][order]
    source_level = source_level[valid][order]
    source_frequency, unique = np.unique(source_frequency, return_index=True)
    source_level = source_level[unique]
    return np.interp(
        np.log2(requested),
        np.log2(source_frequency),
        source_level,
        left=float(source_level[0]),
        right=float(source_level[-1]),
    )


def relative_audibility_weights(
    level_db: np.ndarray,
    *,
    threshold_db: float = -40.0,
    transition_db: float = 20.0,
    minimum_weight: float = 0.20,
    reference_percentile: float = 100.0,
) -> np.ndarray:
    """Return a smooth relative-level priority multiplier.

    ``threshold_db`` is relative to the profile peak by default, not an SPL
    hearing threshold. Values above it keep full priority; lower values fade
    through a smooth knee and never become a hard exclusion.
    """

    level = np.asarray(level_db, dtype=float)
    finite = level[np.isfinite(level)]
    if finite.size == 0:
        return np.ones_like(level, dtype=float)
    percentile = float(np.clip(reference_percentile, 50.0, 100.0))
    reference = float(np.percentile(finite, percentile))
    relative = np.nan_to_num(level - reference, nan=-300.0)
    transition = max(float(transition_db), 1e-6)
    lower = float(threshold_db) - transition
    position = np.clip((relative - lower) / transition, 0.0, 1.0)
    smooth = position * position * (3.0 - 2.0 * position)
    minimum = float(np.clip(minimum_weight, 0.0, 1.0))
    return minimum + (1.0 - minimum) * smooth


def weighted_peak_error(
    error_db: np.ndarray,
    weights: np.ndarray,
    *,
    power: float = 8.0,
) -> float:
    """Return a stable weighted peak metric between RMS and absolute maximum."""

    error = np.abs(np.asarray(error_db, dtype=float))
    weight = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if error.shape != weight.shape:
        raise ValueError("error and weights must have the same shape")
    exponent = max(float(power), 2.0)
    denominator = max(float(np.sum(weight)), 1e-12)
    scale = max(float(np.max(error)), 1e-12)
    normalized = np.clip(error / scale, 0.0, 1.0)
    return float(scale * (np.sum(weight * normalized**exponent) / denominator) ** (1.0 / exponent))
