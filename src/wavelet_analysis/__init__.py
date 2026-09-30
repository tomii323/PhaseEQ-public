from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import signal


WAVELET_BANDWIDTH_OPTIONS = ("1/1 oct", "1/2 oct", "1/3 oct", "1/6 oct")
WAVELET_MODE_OPTIONS = ("Source", "Target", "Source - Target")
WAVELET_PRESETS = {
    "Overview": {"f_min": 50.0, "f_max": 20_000.0, "t_min_ms": -20.0, "t_max_ms": 60.0, "bandwidth_oct": 1.0 / 3.0},
    "Hi": {"f_min": 500.0, "f_max": 24_000.0, "t_min_ms": -5.0, "t_max_ms": 15.0, "bandwidth_oct": 1.0 / 6.0},
    "Mid": {"f_min": 100.0, "f_max": 10_000.0, "t_min_ms": -10.0, "t_max_ms": 30.0, "bandwidth_oct": 1.0 / 3.0},
    "Low": {"f_min": 20.0, "f_max": 500.0, "t_min_ms": -50.0, "t_max_ms": 150.0, "bandwidth_oct": 1.0 / 2.0},
    "Detail": {"f_min": 50.0, "f_max": 20_000.0, "t_min_ms": -10.0, "t_max_ms": 30.0, "bandwidth_oct": 1.0 / 6.0},
}


@dataclass(frozen=True)
class WaveletSettings:
    f_min: float = 50.0
    f_max: float = 20_000.0
    t_min_ms: float = -20.0
    t_max_ms: float = 50.0
    bandwidth_oct: float = 1.0 / 3.0
    dynamic_range_db: float = 40.0
    frequency_bins: int = 128
    time_bins: int = 420
    mask_floor_db: float = -60.0


@dataclass(frozen=True)
class WaveletMap:
    time_ms: np.ndarray
    frequency_hz: np.ndarray
    level_db: np.ndarray
    label: str
    reference_db: float = 0.0
    peak_time_ms: np.ndarray | None = None
    centroid_time_ms: np.ndarray | None = None
    spread_ms: np.ndarray | None = None
    confidence: np.ndarray | None = None
    reflection_index: np.ndarray | None = None
    source_type: str = "original_ir"
    reconstruction_confidence: float = 1.0
    phase_available: bool = True
    smoothing_detected: bool = False
    estimated_smoothing_oct: float | None = None
    smoothing_likelihood: float = 0.0
    source_warning: str | None = None
    alignment_center_sample: float | None = None


@dataclass(frozen=True)
class AdaptiveWindowResult:
    impulse: np.ndarray
    window: np.ndarray
    center_ms: float
    width_ms: float
    confidence: float


@dataclass(frozen=True)
class WaveletCenterResult:
    center_time_ms: float
    confidence: float
    method: str
    peak_center_ms: float
    centroid_center_ms: float
    valid_bins: int
    warning: str | None = None


def settings_from_preset(name: str, base: WaveletSettings | None = None) -> WaveletSettings:
    settings = base or WaveletSettings()
    preset = WAVELET_PRESETS.get(name)
    if not preset:
        return settings
    return replace(settings, **preset)


def bandwidth_label_to_oct(label: str) -> float:
    normalized = str(label).strip()
    mapping = {
        "1/1 oct": 1.0,
        "1/2 oct": 1.0 / 2.0,
        "1/3 oct": 1.0 / 3.0,
        "1/6 oct": 1.0 / 6.0,
    }
    return mapping.get(normalized, 1.0 / 3.0)


def bandwidth_oct_to_label(value: float) -> str:
    pairs = [(1.0, "1/1 oct"), (0.5, "1/2 oct"), (1.0 / 3.0, "1/3 oct"), (1.0 / 6.0, "1/6 oct")]
    return min(pairs, key=lambda item: abs(float(value) - item[0]))[1]


def complex_morlet_scalogram(
    impulse: np.ndarray,
    sample_rate: int,
    settings: WaveletSettings,
    *,
    label: str = "Source",
) -> WaveletMap:
    x = np.asarray(impulse, dtype=float)
    if x.size == 0:
        return _empty_map(settings, label)
    if not np.any(np.isfinite(x)):
        return _empty_map(settings, label)
    x = np.where(np.isfinite(x), x, 0.0)
    f_min = max(float(settings.f_min), 1.0)
    f_max = min(float(settings.f_max), float(sample_rate) / 2.0)
    if f_max <= f_min:
        f_max = min(float(sample_rate) / 2.0, f_min * 2.0)
    freqs = np.geomspace(f_min, f_max, max(2, int(settings.frequency_bins)))
    time_ms = np.linspace(float(settings.t_min_ms), float(settings.t_max_ms), max(2, int(settings.time_bins)))
    sample_positions = time_ms / 1000.0 * float(sample_rate) + (x.size - 1) / 2.0
    level = np.empty((freqs.size, time_ms.size), dtype=float)

    for row, frequency in enumerate(freqs):
        level[row, :] = _morlet_envelope_at_samples(
            x,
            sample_rate,
            float(frequency),
            float(settings.bandwidth_oct),
            sample_positions,
        )

    peak_time_ms, centroid_time_ms, spread_ms, confidence, reflection_index = _wavelet_metrics_from_energy(
        time_ms,
        freqs,
        level * level,
    )
    finite_level = level[np.isfinite(level)]
    max_level = float(np.max(finite_level)) if finite_level.size else 0.0
    reference = max(max_level, 1e-15)
    raw_level_db = 20.0 * np.log10(np.maximum(level / reference, 1e-15))
    mask = raw_level_db <= float(settings.mask_floor_db)
    level_db = raw_level_db
    floor = -abs(float(settings.dynamic_range_db))
    level_db = np.clip(level_db, floor, 0.0)
    level_db = np.where(mask, np.nan, level_db)
    return WaveletMap(
        time_ms=time_ms,
        frequency_hz=freqs,
        level_db=level_db,
        label=label,
        reference_db=20.0 * np.log10(reference),
        peak_time_ms=peak_time_ms,
        centroid_time_ms=centroid_time_ms,
        spread_ms=spread_ms,
        confidence=confidence,
        reflection_index=reflection_index,
    )


def with_wavelet_source_metadata(
    wavelet_map: WaveletMap,
    *,
    source_type: str,
    reconstruction_confidence: float,
    phase_available: bool,
    smoothing_detected: bool = False,
    estimated_smoothing_oct: float | None = None,
    smoothing_likelihood: float = 0.0,
    source_warning: str | None = None,
    scale_confidence: bool = True,
) -> WaveletMap:
    confidence = wavelet_map.confidence
    confidence_scale = float(np.clip(reconstruction_confidence, 0.0, 1.0))
    if scale_confidence and confidence is not None:
        confidence = np.clip(np.asarray(confidence, dtype=float) * confidence_scale, 0.0, 1.0)
    return replace(
        wavelet_map,
        confidence=confidence,
        source_type=str(source_type),
        reconstruction_confidence=confidence_scale,
        phase_available=bool(phase_available),
        smoothing_detected=bool(smoothing_detected),
        estimated_smoothing_oct=estimated_smoothing_oct,
        smoothing_likelihood=float(np.clip(smoothing_likelihood, 0.0, 1.0)),
        source_warning=source_warning,
    )


def wavelet_difference_map(source: WaveletMap, target: WaveletMap, *, label: str = "Source - Target") -> WaveletMap:
    source_level, target_level = _aligned_levels(source, target)
    level_db = source_level - target_level
    peak_time_ms, centroid_time_ms, spread_ms, confidence, reflection_index = _wavelet_metrics_from_signed_level(
        source.time_ms,
        source.frequency_hz,
        level_db,
    )
    return WaveletMap(
        time_ms=source.time_ms,
        frequency_hz=source.frequency_hz,
        level_db=level_db,
        label=label,
        reference_db=0.0,
        peak_time_ms=peak_time_ms,
        centroid_time_ms=centroid_time_ms,
        spread_ms=spread_ms,
        confidence=confidence,
        reflection_index=reflection_index,
        source_type=source.source_type,
        reconstruction_confidence=source.reconstruction_confidence,
        phase_available=source.phase_available,
        smoothing_detected=source.smoothing_detected,
        estimated_smoothing_oct=source.estimated_smoothing_oct,
        smoothing_likelihood=source.smoothing_likelihood,
        source_warning=source.source_warning,
    )


def wavelet_improvement_map(
    system: WaveletMap,
    source: WaveletMap,
    target: WaveletMap,
    *,
    label: str = "Improvement Map",
) -> WaveletMap:
    system_level, target_level = _aligned_levels(system, target)
    source_level, _ = _aligned_levels(source, target)
    before = np.abs(system_level - target_level)
    after = np.abs(source_level - target_level)
    level_db = before - after
    peak_time_ms, centroid_time_ms, spread_ms, confidence, reflection_index = _wavelet_metrics_from_signed_level(
        source.time_ms,
        source.frequency_hz,
        level_db,
    )
    return WaveletMap(
        time_ms=source.time_ms,
        frequency_hz=source.frequency_hz,
        level_db=level_db,
        label=label,
        reference_db=0.0,
        peak_time_ms=peak_time_ms,
        centroid_time_ms=centroid_time_ms,
        spread_ms=spread_ms,
        confidence=confidence,
        reflection_index=reflection_index,
        source_type=source.source_type,
        reconstruction_confidence=source.reconstruction_confidence,
        phase_available=source.phase_available,
        smoothing_detected=source.smoothing_detected,
        estimated_smoothing_oct=source.estimated_smoothing_oct,
        smoothing_likelihood=source.smoothing_likelihood,
        source_warning=source.source_warning,
    )


def adaptive_window_from_wavelet_map(
    impulse: np.ndarray,
    wavelet_map: WaveletMap,
    sample_rate: int,
    *,
    enabled: bool = False,
    confidence_threshold: float = 0.25,
    spread_multiplier: float = 4.0,
    min_width_ms: float = 2.0,
) -> AdaptiveWindowResult:
    """Optionally apply a confidence-weighted time window derived from CWT metrics."""
    x = np.asarray(impulse, dtype=float)
    if x.size == 0:
        return AdaptiveWindowResult(x.copy(), np.asarray([], dtype=float), 0.0, 0.0, 0.0)
    unity = np.ones_like(x, dtype=float)
    if not enabled:
        return AdaptiveWindowResult(x.copy(), unity, 0.0, x.size / float(sample_rate) * 1000.0, 1.0)

    center_ms, width_ms, confidence = _adaptive_window_bounds_ms(
        wavelet_map,
        spread_multiplier=spread_multiplier,
        min_width_ms=min_width_ms,
        confidence_threshold=confidence_threshold,
    )
    center_sample = (x.size - 1) / 2.0 + center_ms * float(sample_rate) / 1000.0
    half_width_samples = max(width_ms * float(sample_rate) / 2000.0, 1.0)
    sample_index = np.arange(x.size, dtype=float)
    distance = np.abs(sample_index - center_sample)
    plateau = half_width_samples * 0.75
    fade = max(half_width_samples - plateau, 1.0)
    window = np.ones_like(x, dtype=float)
    outside = distance > plateau
    window[outside] = 0.5 + 0.5 * np.cos(np.pi * np.clip((distance[outside] - plateau) / fade, 0.0, 1.0))
    window[distance >= half_width_samples] = 0.0
    return AdaptiveWindowResult(
        impulse=x * window,
        window=window,
        center_ms=center_ms,
        width_ms=width_ms,
        confidence=confidence,
    )


def center_from_wavelet_map(
    wavelet_map: WaveletMap,
    *,
    f_min: float = 300.0,
    f_max: float = 8000.0,
    min_confidence: float = 0.25,
) -> WaveletCenterResult:
    """Estimate bulk-delay center time from WaveletMap metrics.

    Peak time is the primary center. Centroid is used as a consistency check
    because late reflections can pull it behind the direct sound.
    """
    freq = np.asarray(wavelet_map.frequency_hz, dtype=float)
    peak = np.asarray(
        wavelet_map.peak_time_ms
        if wavelet_map.peak_time_ms is not None
        else np.full(freq.size, np.nan, dtype=float),
        dtype=float,
    )
    centroid = np.asarray(
        wavelet_map.centroid_time_ms
        if wavelet_map.centroid_time_ms is not None
        else np.full(freq.size, np.nan, dtype=float),
        dtype=float,
    )
    spread = np.asarray(
        wavelet_map.spread_ms
        if wavelet_map.spread_ms is not None
        else np.full(freq.size, np.nan, dtype=float),
        dtype=float,
    )
    confidence = np.asarray(
        wavelet_map.confidence
        if wavelet_map.confidence is not None
        else np.zeros(freq.size, dtype=float),
        dtype=float,
    )
    reflection = np.asarray(
        wavelet_map.reflection_index
        if wavelet_map.reflection_index is not None
        else np.zeros(freq.size, dtype=float),
        dtype=float,
    )
    if not (freq.size == peak.size == centroid.size == spread.size == confidence.size == reflection.size):
        return WaveletCenterResult(0.0, 0.0, "wavelet_unavailable", np.nan, np.nan, 0, "metric size mismatch")

    band = (freq >= float(f_min)) & (freq <= float(f_max))
    finite = (
        band
        & np.isfinite(peak)
        & np.isfinite(centroid)
        & np.isfinite(spread)
        & np.isfinite(confidence)
        & np.isfinite(reflection)
    )
    candidates = finite & (confidence >= float(min_confidence))
    if np.count_nonzero(candidates) < 3:
        candidates = finite & (confidence >= 0.05)
    if np.count_nonzero(candidates) < 1:
        return WaveletCenterResult(0.0, 0.0, "wavelet_unavailable", np.nan, np.nan, 0, "no reliable bins")

    weights = (
        np.clip(confidence[candidates], 0.0, 1.0)
        * (1.0 / (1.0 + np.maximum(reflection[candidates], 0.0)))
        * (1.0 / (1.0 + np.maximum(spread[candidates], 0.0) / 2.0))
        * _wavelet_center_band_weight(freq[candidates], float(f_min), float(f_max))
    )
    if not np.any(weights > 0.0):
        return WaveletCenterResult(0.0, 0.0, "wavelet_unavailable", np.nan, np.nan, 0, "zero center weights")

    peak_center = _weighted_median(peak[candidates], weights)
    centroid_center = _weighted_median(centroid[candidates], weights)
    peak_centroid_delta = abs(float(centroid_center) - float(peak_center))
    valid_bins = int(np.count_nonzero(candidates))
    mean_confidence = float(np.average(confidence[candidates], weights=weights))
    mean_reflection = float(np.average(np.maximum(reflection[candidates], 0.0), weights=weights))
    median_spread = float(np.nanmedian(spread[candidates]))
    consistency = 1.0 / (1.0 + peak_centroid_delta / 2.0)
    reflection_quality = 1.0 / (1.0 + mean_reflection)
    spread_quality = 1.0 / (1.0 + max(median_spread, 0.0) / 4.0)
    center_confidence = float(
        np.clip(
            0.50 * mean_confidence + 0.20 * consistency + 0.20 * reflection_quality + 0.10 * spread_quality,
            0.0,
            1.0,
        )
    )

    warning = None
    if peak_centroid_delta > 3.0:
        warning = "centroid is delayed; likely reflection or decay tail"
    elif mean_reflection >= 0.50:
        warning = "high reflection index"
    elif valid_bins < 5:
        warning = "few reliable bins"

    return WaveletCenterResult(
        center_time_ms=float(peak_center),
        confidence=center_confidence,
        method="Wavelet weighted peak median",
        peak_center_ms=float(peak_center),
        centroid_center_ms=float(centroid_center),
        valid_bins=valid_bins,
        warning=warning,
    )


def gain_phase_from_ir_fft(
    impulse: np.ndarray,
    sample_rate: int,
    *,
    fft_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate gain and unwrapped phase from an IR via FFT."""
    x = np.asarray(impulse, dtype=float)
    if x.size == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty
    n_fft = int(fft_size) if fft_size is not None else int(2 ** np.ceil(np.log2(max(x.size, 2))))
    n_fft = max(n_fft, x.size, 2)
    response = np.fft.rfft(x, n=n_fft)
    frequency = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
    gain_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))
    phase_deg = np.rad2deg(np.unwrap(np.angle(response)))
    return frequency, gain_db, phase_deg


def _morlet_envelope(
    impulse: np.ndarray,
    sample_rate: int,
    frequency: float,
    bandwidth_oct: float,
) -> np.ndarray:
    wavelet = _morlet_wavelet(impulse.size, sample_rate, frequency, bandwidth_oct)
    coefficient = signal.fftconvolve(impulse, np.conj(wavelet[::-1]), mode="same")
    return _normalized_wavelet_envelope(coefficient, wavelet)


def _morlet_envelope_at_samples(
    impulse: np.ndarray,
    sample_rate: int,
    frequency: float,
    bandwidth_oct: float,
    sample_positions: np.ndarray,
) -> np.ndarray:
    """Return an exact local-convolution envelope at requested sample positions.

    Only the input span that can contribute to the requested output window is
    convolved. This preserves the full linear-convolution result while avoiding
    work for the unshown part of a long impulse response.
    """

    x = np.asarray(impulse, dtype=float)
    positions = np.asarray(sample_positions, dtype=float)
    output = np.zeros_like(positions, dtype=float)
    if x.size == 0 or positions.size == 0:
        return output
    finite = np.isfinite(positions)
    in_range = finite & (positions >= 0.0) & (positions <= float(x.size - 1))
    if not np.any(in_range):
        return output

    wavelet = _morlet_wavelet(x.size, sample_rate, frequency, bandwidth_oct)
    half_samples = (wavelet.size - 1) // 2
    valid_positions = positions[in_range]
    output_start = max(int(np.floor(float(np.min(valid_positions)))), 0)
    output_stop = min(int(np.ceil(float(np.max(valid_positions)))), x.size - 1)
    input_start = max(output_start - half_samples, 0)
    input_stop = min(output_stop + half_samples + 1, x.size)

    kernel = np.conj(wavelet[::-1])
    local_full = signal.fftconvolve(x[input_start:input_stop], kernel, mode="full")
    output_indices = np.arange(output_start, output_stop + 1, dtype=int)
    convolution_indices = output_indices + half_samples - input_start
    local_envelope = _normalized_wavelet_envelope(local_full[convolution_indices], wavelet)
    output[in_range] = np.interp(valid_positions, output_indices, local_envelope)
    return output


def _morlet_wavelet(
    impulse_size: int,
    sample_rate: int,
    frequency: float,
    bandwidth_oct: float,
) -> np.ndarray:
    sigma_t = _sigma_t_for_bandwidth(frequency, bandwidth_oct)
    half_samples = max(int(np.ceil(4.0 * sigma_t * sample_rate)), 8)
    half_samples = min(half_samples, max(8, int(impulse_size) - 1))
    t = np.arange(-half_samples, half_samples + 1, dtype=float) / float(sample_rate)
    carrier = np.exp(1j * 2.0 * np.pi * frequency * t)
    window = np.exp(-0.5 * (t / sigma_t) ** 2)
    wavelet = carrier * window
    norm = np.sqrt(np.sum(np.abs(wavelet) ** 2))
    if norm > 0:
        wavelet = wavelet / norm
    return wavelet


def _normalized_wavelet_envelope(coefficient: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
    envelope = np.abs(coefficient)
    impulse_peak = float(np.max(np.abs(wavelet))) if wavelet.size else 0.0
    if impulse_peak > 0:
        envelope = envelope / impulse_peak
    return envelope


def _adaptive_window_bounds_ms(
    wavelet_map: WaveletMap,
    *,
    spread_multiplier: float,
    min_width_ms: float,
    confidence_threshold: float,
) -> tuple[float, float, float]:
    centroid = np.asarray(
        wavelet_map.centroid_time_ms
        if wavelet_map.centroid_time_ms is not None
        else np.full_like(wavelet_map.frequency_hz, np.nan, dtype=float),
        dtype=float,
    )
    spread = np.asarray(
        wavelet_map.spread_ms
        if wavelet_map.spread_ms is not None
        else np.full_like(wavelet_map.frequency_hz, np.nan, dtype=float),
        dtype=float,
    )
    confidence = np.asarray(
        wavelet_map.confidence
        if wavelet_map.confidence is not None
        else np.zeros_like(wavelet_map.frequency_hz, dtype=float),
        dtype=float,
    )
    valid = (
        np.isfinite(centroid)
        & np.isfinite(spread)
        & np.isfinite(confidence)
        & (confidence >= float(confidence_threshold))
    )
    if not np.any(valid):
        return 0.0, max(float(min_width_ms), 0.0), 0.0
    weights = np.maximum(confidence[valid], 1e-9)
    center = float(np.sum(centroid[valid] * weights) / np.sum(weights))
    width = float(np.nanmedian(spread[valid]) * max(float(spread_multiplier), 1.0) * 2.0)
    width = max(width, float(min_width_ms))
    return center, width, float(np.nanmean(confidence[valid]))


def _sigma_t_for_bandwidth(frequency: float, bandwidth_oct: float) -> float:
    octave = max(float(bandwidth_oct), 1.0 / 24.0)
    ratio = 2.0 ** (octave / 2.0)
    bandwidth_hz = max(float(frequency) * (ratio - 1.0 / ratio), 1.0)
    sigma_f = bandwidth_hz / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return float(max(1.0 / (2.0 * np.pi * sigma_f), 1.0 / (8.0 * max(float(frequency), 1.0))))


def _wavelet_metrics_from_signed_level(
    time_ms: np.ndarray,
    frequency_hz: np.ndarray,
    level_db: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    level = np.asarray(level_db, dtype=float)
    finite = np.isfinite(level)
    if not np.any(finite):
        energy = np.zeros_like(level, dtype=float)
    else:
        floor = float(np.nanmin(np.abs(level[finite])))
        energy = np.where(finite, np.maximum(np.abs(level) - floor, 0.0) ** 2, 0.0)
    return _wavelet_metrics_from_energy(time_ms, frequency_hz, energy)


def _wavelet_metrics_from_energy(
    time_ms: np.ndarray,
    frequency_hz: np.ndarray,
    energy: np.ndarray,
    *,
    neighborhood_db: float = 15.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(time_ms, dtype=float)
    freq = np.asarray(frequency_hz, dtype=float)
    values = np.asarray(energy, dtype=float)
    if time.size == 0 or freq.size == 0 or values.shape != (freq.size, time.size):
        empty = np.full(freq.size, np.nan, dtype=float)
        return empty, empty, empty, np.zeros(freq.size, dtype=float), empty

    peak_time = np.full(freq.size, np.nan, dtype=float)
    centroid = np.full(freq.size, np.nan, dtype=float)
    spread = np.full(freq.size, np.nan, dtype=float)
    confidence = np.zeros(freq.size, dtype=float)
    reflection_index = np.full(freq.size, np.nan, dtype=float)
    threshold_ratio = 10.0 ** (-abs(float(neighborhood_db)) / 10.0)

    for idx, row in enumerate(values):
        finite = np.isfinite(row) & (row > 0.0)
        if not np.any(finite):
            continue
        row_energy = np.where(finite, row, 0.0)
        peak_index = int(np.argmax(row_energy))
        peak_energy = float(row_energy[peak_index])
        if peak_energy <= 0.0:
            continue
        selected = row_energy >= peak_energy * threshold_ratio
        selected &= finite
        selected_energy = row_energy[selected]
        if selected_energy.size == 0 or float(np.sum(selected_energy)) <= 0.0:
            continue
        selected_time = time[selected]
        total_selected = float(np.sum(selected_energy))
        center = float(np.sum(selected_time * selected_energy) / total_selected)
        variance = float(np.sum(((selected_time - center) ** 2) * selected_energy) / total_selected)
        total_energy = float(np.sum(row_energy))
        post_start = float(time[peak_index]) + max(float(np.sqrt(max(variance, 0.0))), _time_step_ms(time))
        late_energy = float(np.sum(row_energy[finite & (time > post_start)]))
        background = float(np.median(row_energy[finite]))
        prominence = (peak_energy - background) / max(peak_energy, 1e-15)
        concentration = total_selected / max(total_energy, 1e-15)

        peak_time[idx] = float(time[peak_index])
        centroid[idx] = center
        spread[idx] = float(np.sqrt(max(variance, 0.0)))
        confidence[idx] = float(np.clip(0.65 * concentration + 0.35 * prominence, 0.0, 1.0))
        reflection_index[idx] = float(late_energy / max(total_selected, 1e-15))

    centroid = _smooth_metric_by_frequency(centroid, confidence)
    spread = _smooth_metric_by_frequency(spread, confidence, nonnegative=True)
    confidence = np.clip(_smooth_metric_by_frequency(confidence, np.ones_like(confidence), nonnegative=True), 0.0, 1.0)
    reflection_index = _smooth_metric_by_frequency(reflection_index, confidence, nonnegative=True)
    return peak_time, centroid, spread, confidence, reflection_index


def _smooth_metric_by_frequency(
    values: np.ndarray,
    confidence: np.ndarray,
    *,
    nonnegative: bool = False,
) -> np.ndarray:
    metric = np.asarray(values, dtype=float).copy()
    conf = np.asarray(confidence, dtype=float)
    valid = np.isfinite(metric) & np.isfinite(conf) & (conf > 0.05)
    if np.count_nonzero(valid) < 3:
        return np.maximum(metric, 0.0) if nonnegative else metric

    indices = np.arange(metric.size, dtype=float)
    interpolated = np.interp(indices, indices[valid], metric[valid])
    kernel_size = min(5, metric.size if metric.size % 2 == 1 else metric.size - 1)
    if kernel_size >= 3:
        median = signal.medfilt(interpolated, kernel_size=kernel_size)
    else:
        median = interpolated
    kernel = np.ones(min(5, metric.size), dtype=float)
    kernel = kernel / float(np.sum(kernel))
    pad = kernel.size // 2
    padded = np.pad(median, (pad, kernel.size - 1 - pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")

    residual = np.abs(interpolated - smoothed)
    residual_valid = residual[valid]
    if residual_valid.size:
        limit = float(np.nanmedian(residual_valid) + 4.0 * np.nanmedian(np.abs(residual_valid - np.nanmedian(residual_valid))))
        outlier = valid & (residual > max(limit, 1e-9))
        interpolated[outlier] = smoothed[outlier]

    metric[valid] = smoothed[valid]
    if nonnegative:
        metric = np.maximum(metric, 0.0)
    return metric


def _time_step_ms(time_ms: np.ndarray) -> float:
    time = np.asarray(time_ms, dtype=float)
    if time.size < 2:
        return 0.0
    steps = np.diff(time)
    steps = steps[np.isfinite(steps) & (steps > 0.0)]
    if steps.size == 0:
        return 0.0
    return float(np.median(steps))


def _wavelet_center_band_weight(frequency_hz: np.ndarray, f_min: float, f_max: float) -> np.ndarray:
    frequency = np.asarray(frequency_hz, dtype=float)
    if not (0.0 < f_min < f_max):
        return np.ones_like(frequency, dtype=float)
    center = np.sqrt(float(f_min) * float(f_max))
    distance_oct = np.abs(np.log2(np.maximum(frequency, 1e-9) / center))
    span_oct = max(np.log2(float(f_max) / float(f_min)) / 2.0, 1e-9)
    return np.clip(1.0 - 0.35 * distance_oct / span_oct, 0.35, 1.0)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    value = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    valid = np.isfinite(value) & np.isfinite(weight) & (weight > 0.0)
    if not np.any(valid):
        return float("nan")
    value = value[valid]
    weight = weight[valid]
    order = np.argsort(value)
    value = value[order]
    weight = weight[order]
    cumulative = np.cumsum(weight)
    cutoff = 0.5 * float(cumulative[-1])
    return float(value[min(int(np.searchsorted(cumulative, cutoff, side="left")), value.size - 1)])


def _aligned_levels(first: WaveletMap, second: WaveletMap) -> tuple[np.ndarray, np.ndarray]:
    if first.level_db.shape == second.level_db.shape:
        return _nan_to_wavelet_floor(first.level_db), _nan_to_wavelet_floor(second.level_db)
    raise ValueError("Wavelet maps must share the same grid.")


def _nan_to_wavelet_floor(level_db: np.ndarray) -> np.ndarray:
    level = np.asarray(level_db, dtype=float)
    finite = np.isfinite(level)
    if np.any(finite):
        floor = float(np.nanmin(level[finite]))
    else:
        floor = -60.0
    return np.where(finite, level, floor)


def _empty_map(settings: WaveletSettings, label: str) -> WaveletMap:
    time_ms = np.linspace(settings.t_min_ms, settings.t_max_ms, max(2, int(settings.time_bins)))
    frequency_hz = np.geomspace(max(settings.f_min, 1.0), max(settings.f_max, settings.f_min + 1.0), max(2, int(settings.frequency_bins)))
    empty_metric = np.full(frequency_hz.size, np.nan, dtype=float)
    return WaveletMap(
        time_ms=time_ms,
        frequency_hz=frequency_hz,
        level_db=np.full((frequency_hz.size, time_ms.size), np.nan, dtype=float),
        label=label,
        peak_time_ms=empty_metric,
        centroid_time_ms=empty_metric,
        spread_ms=empty_metric,
        confidence=np.zeros(frequency_hz.size, dtype=float),
        reflection_index=empty_metric,
    )
