from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from scipy import signal

from ..config import SpeakerResponse
from ..phase_curves import fractional_octave_smooth


@dataclass(frozen=True)
class SystemTargetShaperSettings:
    """Device-independent settings for deriving a relative System Target shape."""

    smoothing_fraction: float = 3.0
    reference_mode: str = "band_median"
    reference_low_hz: float = 500.0
    reference_high_hz: float = 2_000.0
    reference_frequency_hz: float = 1_000.0
    coverage_fade_oct: float = 1.0 / 3.0


@dataclass(frozen=True)
class SystemTargetShaperResult:
    frequency: np.ndarray
    system_gain_db: np.ndarray
    smoothed_system_gain_db: np.ndarray
    target_gain_db: np.ndarray
    normalized_system_gain_db: np.ndarray
    normalized_target_gain_db: np.ndarray
    correction_gain_db: np.ndarray
    correction_response: np.ndarray
    coverage_weight: np.ndarray
    system_reference_db: float
    target_reference_db: float


@dataclass(frozen=True)
class SystemTargetFIRResult:
    fir: np.ndarray
    frequency: np.ndarray
    desired_gain_db: np.ndarray
    realized_gain_db: np.ndarray
    error_db: np.ndarray
    rms_error_db: float
    max_abs_error_db: float
    delay_samples: float


def power_average_gain_db(responses: Sequence[np.ndarray]) -> np.ndarray:
    """Return the RMS/power-average magnitude of linked channel responses."""

    if not responses:
        raise ValueError("at least one system response is required")
    arrays = [np.asarray(item, dtype=complex) for item in responses]
    shape = arrays[0].shape
    if arrays[0].ndim != 1 or any(item.ndim != 1 or item.shape != shape for item in arrays):
        raise ValueError("linked system responses must be one-dimensional and share one axis")
    if any(not np.all(np.isfinite(item)) for item in arrays):
        raise ValueError("linked system responses must contain only finite values")
    mean_power = np.mean([np.abs(item) ** 2 for item in arrays], axis=0)
    return 10.0 * np.log10(np.maximum(mean_power, 1e-24))


def build_system_target_shaper(
    frequency: np.ndarray,
    system_response: np.ndarray,
    target_response: SpeakerResponse,
    settings: SystemTargetShaperSettings | None = None,
) -> SystemTargetShaperResult:
    """Derive a magnitude-only relative correction from System Sum to Target.

    This function deliberately does not impose boost, cut, clipping, or DSP
    headroom limits.  Those properties belong to the selected output device.
    """

    config = settings or SystemTargetShaperSettings()
    axis = _validated_axis(frequency)
    response = np.asarray(system_response, dtype=complex)
    if response.ndim != 1 or response.shape != axis.shape:
        raise ValueError("system_response must be one-dimensional and match frequency")
    if not np.all(np.isfinite(response)):
        raise ValueError("system_response must contain only finite values")
    _validate_settings(config)

    system_gain = 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))
    smoothed_system = fractional_octave_smooth(
        axis,
        system_gain,
        float(config.smoothing_fraction),
    )
    target_gain, coverage = _target_on_axis(
        target_response,
        axis,
        fade_width_oct=float(config.coverage_fade_oct),
    )
    reference_mask = coverage > 0.0
    system_reference = _reference_level(smoothed_system, axis, reference_mask, config)
    target_reference = _reference_level(target_gain, axis, reference_mask, config)
    normalized_system = smoothed_system - system_reference
    normalized_target = target_gain - target_reference
    correction_gain = coverage * (normalized_target - normalized_system)
    correction_response = np.power(10.0, correction_gain / 20.0).astype(complex)
    if not np.all(np.isfinite(correction_response)):
        raise ValueError("System Target correction is outside the numeric range")

    return SystemTargetShaperResult(
        frequency=axis.copy(),
        system_gain_db=system_gain,
        smoothed_system_gain_db=smoothed_system,
        target_gain_db=target_gain,
        normalized_system_gain_db=normalized_system,
        normalized_target_gain_db=normalized_target,
        correction_gain_db=correction_gain,
        correction_response=correction_response,
        coverage_weight=coverage,
        system_reference_db=float(system_reference),
        target_reference_db=float(target_reference),
    )


def design_system_target_fir(
    shaper: SystemTargetShaperResult,
    sample_rate: int,
    taps: int,
    *,
    window: str | tuple[str, float] | None = None,
) -> SystemTargetFIRResult:
    """Realize a System Target magnitude correction as a linear-phase FIR."""

    rate = int(sample_rate)
    output_taps = int(taps)
    if rate <= 0:
        raise ValueError("sample_rate must be positive")
    if output_taps < 3 or output_taps % 2 == 0:
        raise ValueError("System Target FIR taps must be an odd integer of at least 3")
    frequency = _validated_axis(shaper.frequency)
    desired_gain = np.asarray(shaper.correction_gain_db, dtype=float)
    if desired_gain.shape != frequency.shape or not np.all(np.isfinite(desired_gain)):
        raise ValueError("shaper correction must be finite and match its frequency axis")
    nyquist = rate / 2.0
    if frequency[-1] > nyquist * (1.0 + 1e-12):
        raise ValueError("shaper frequency exceeds Nyquist")

    grid_intervals = 1
    while grid_intervals < max(output_taps, 16_384):
        grid_intervals *= 2
    grid_frequency = np.linspace(0.0, nyquist, grid_intervals + 1)
    grid_gain = np.interp(
        grid_frequency,
        frequency,
        desired_gain,
        left=float(desired_gain[0]),
        right=float(desired_gain[-1]),
    )
    grid_magnitude = np.power(10.0, grid_gain / 20.0)
    if not np.all(np.isfinite(grid_magnitude)):
        raise ValueError("System Target FIR magnitude is outside the numeric range")
    fir = signal.firwin2(
        output_taps,
        grid_frequency,
        grid_magnitude,
        nfreqs=grid_intervals + 1,
        window=window,
        fs=float(rate),
    ).astype(np.float64, copy=False)
    fir = 0.5 * (fir + fir[::-1])

    angular_frequency = 2.0 * np.pi * frequency / float(rate)
    _, realized = signal.freqz(fir, worN=angular_frequency)
    realized_gain = 20.0 * np.log10(np.maximum(np.abs(realized), 1e-12))
    error = realized_gain - desired_gain
    evaluation_mask = np.asarray(shaper.coverage_weight, dtype=float) > 0.5
    if not np.any(evaluation_mask):
        evaluation_mask = np.ones_like(frequency, dtype=bool)
    return SystemTargetFIRResult(
        fir=fir,
        frequency=frequency.copy(),
        desired_gain_db=desired_gain.copy(),
        realized_gain_db=realized_gain,
        error_db=error,
        rms_error_db=float(np.sqrt(np.mean(error[evaluation_mask] ** 2))),
        max_abs_error_db=float(np.max(np.abs(error[evaluation_mask]))),
        delay_samples=(output_taps - 1) / 2.0,
    )


def apply_system_target_shaper(
    response: np.ndarray,
    shaper: SystemTargetShaperResult,
) -> np.ndarray:
    """Apply the analytical target transfer once to a complex response."""

    source = np.asarray(response, dtype=complex)
    if source.shape != shaper.correction_response.shape:
        raise ValueError("response must match the System Target frequency axis")
    return source * shaper.correction_response


def _validated_axis(frequency: np.ndarray) -> np.ndarray:
    axis = np.asarray(frequency, dtype=float)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError("frequency must contain at least two one-dimensional points")
    if not np.all(np.isfinite(axis)) or np.any(axis < 0.0) or np.any(np.diff(axis) <= 0.0):
        raise ValueError("frequency must be finite, non-negative, and strictly increasing")
    return axis


def _validate_settings(settings: SystemTargetShaperSettings) -> None:
    if float(settings.smoothing_fraction) < 0.0:
        raise ValueError("smoothing_fraction must not be negative")
    if float(settings.coverage_fade_oct) < 0.0:
        raise ValueError("coverage_fade_oct must not be negative")
    if float(settings.reference_low_hz) <= 0.0:
        raise ValueError("reference_low_hz must be positive")
    if float(settings.reference_high_hz) <= float(settings.reference_low_hz):
        raise ValueError("reference_high_hz must exceed reference_low_hz")
    if float(settings.reference_frequency_hz) <= 0.0:
        raise ValueError("reference_frequency_hz must be positive")
    if settings.reference_mode not in {"band_median", "frequency", "none"}:
        raise ValueError("reference_mode must be band_median, frequency, or none")


def _target_on_axis(
    response: SpeakerResponse,
    frequency: np.ndarray,
    *,
    fade_width_oct: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_frequency = np.asarray(response.frequency, dtype=float)
    source_gain = np.asarray(response.gain_db, dtype=float)
    if source_frequency.shape != source_gain.shape:
        raise ValueError("Target frequency and gain must have the same length")
    valid = np.isfinite(source_frequency) & np.isfinite(source_gain) & (source_frequency > 0.0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("System Target requires at least two finite positive-frequency points")
    source_frequency = source_frequency[valid]
    source_gain = source_gain[valid]
    order = np.argsort(source_frequency)
    source_frequency = source_frequency[order]
    source_gain = source_gain[order]
    unique_frequency, unique_indices = np.unique(source_frequency, return_index=True)
    source_frequency = unique_frequency
    source_gain = source_gain[unique_indices]
    if source_frequency.size < 2:
        raise ValueError("System Target requires at least two unique frequency points")

    target = np.full_like(frequency, float(source_gain[0]))
    positive = frequency > 0.0
    target[positive] = np.interp(
        np.log2(frequency[positive]),
        np.log2(source_frequency),
        source_gain,
        left=float(source_gain[0]),
        right=float(source_gain[-1]),
    )
    coverage = _coverage_weight(
        frequency,
        float(source_frequency[0]),
        float(source_frequency[-1]),
        fade_width_oct,
    )
    return target, coverage


def _coverage_weight(
    frequency: np.ndarray,
    minimum_hz: float,
    maximum_hz: float,
    fade_width_oct: float,
) -> np.ndarray:
    weight = np.zeros_like(frequency, dtype=float)
    positive = frequency > 0.0
    inside = positive & (frequency >= minimum_hz) & (frequency <= maximum_hz)
    weight[inside] = 1.0
    if fade_width_oct <= 0.0:
        return weight
    low_start = minimum_hz * 2.0 ** (-fade_width_oct)
    low = positive & (frequency >= low_start) & (frequency < minimum_hz)
    if np.any(low):
        position = np.log2(frequency[low] / low_start) / fade_width_oct
        weight[low] = _smootherstep(position)
    high_stop = maximum_hz * 2.0 ** fade_width_oct
    high = (frequency > maximum_hz) & (frequency <= high_stop)
    if np.any(high):
        position = np.log2(high_stop / frequency[high]) / fade_width_oct
        weight[high] = _smootherstep(position)
    return weight


def _reference_level(
    values: np.ndarray,
    frequency: np.ndarray,
    support: np.ndarray,
    settings: SystemTargetShaperSettings,
) -> float:
    if settings.reference_mode == "none":
        return 0.0
    if settings.reference_mode == "band_median":
        band = (
            support
            & (frequency >= float(settings.reference_low_hz))
            & (frequency <= float(settings.reference_high_hz))
        )
        if np.any(band):
            return float(np.median(values[band]))
    candidates = np.flatnonzero(support)
    if candidates.size == 0:
        raise ValueError("System Target does not overlap the requested frequency axis")
    positive_candidates = candidates[frequency[candidates] > 0.0]
    if positive_candidates.size:
        candidates = positive_candidates
    target = float(settings.reference_frequency_hz)
    distances = np.abs(np.log2(np.maximum(frequency[candidates], 1e-12) / target))
    return float(values[candidates[int(np.argmin(distances))]])


def _smootherstep(value: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return x**3 * (x * (x * 6.0 - 15.0) + 10.0)
