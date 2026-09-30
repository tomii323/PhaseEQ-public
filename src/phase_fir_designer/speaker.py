from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO
import re
import urllib.request
import warnings

import numpy as np
from response_math import interpolate_values, interpolate_phase
from scipy.io import wavfile

from wavelet_analysis import WaveletSettings, complex_morlet_scalogram
from .config import SpeakerResponse
from .phase_curves import fractional_octave_smooth


@dataclass(frozen=True)
class ImpulseResponseInput:
    speaker_response: SpeakerResponse
    impulse: np.ndarray
    sample_rate: int
    source_type: str = "original_ir"


@dataclass(frozen=True)
class SpeakerGainShiftEstimate:
    shift_db: float
    reference_gain_db: float
    peak_gain_db: float
    threshold_db: float
    candidate_count: int
    used_wavelet: bool = False
    reference_low_hz: float = 0.0
    reference_high_hz: float = 0.0
    message: str = ""


@dataclass(frozen=True)
class WaveletImpulseSource:
    impulse: np.ndarray
    source_type: str
    reconstruction_confidence: float
    phase_available: bool
    smoothing_detected: bool = False
    estimated_smoothing_oct: float | None = None
    smoothing_likelihood: float = 0.0
    source_warning: str | None = None


@dataclass(frozen=True)
class TargetGainShiftEstimate:
    shift_db: float
    speaker_peak_gain_db: float
    speaker_threshold_db: float
    candidate_count: int
    reference_low_hz: float = 0.0
    reference_high_hz: float = 0.0
    message: str = ""


def load_speaker_response(path: str | Path) -> SpeakerResponse:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return _response_from_rows(_numeric_rows(text.splitlines()))


def load_speaker_response_url(url: str, timeout: float = 10.0) -> SpeakerResponse:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        text = response.read().decode(charset, errors="replace")
    return load_speaker_response_text(text)


def load_speaker_response_text(source: str | TextIO, delimiter: str | None = None) -> SpeakerResponse:
    if hasattr(source, "read"):
        text = source.read()
    else:
        text = source
    return _response_from_rows(_numeric_rows(str(text).splitlines(), delimiter=delimiter))


def load_impulse_response_wav(
    source: str | Path | bytes | TextIO,
    *,
    expected_sample_rate: int | None = None,
    fft_size: int | None = None,
) -> SpeakerResponse:
    """Load a mono WAV impulse response and convert it to gain/phase response."""
    return load_impulse_response_wav_with_ir(
        source,
        expected_sample_rate=expected_sample_rate,
        fft_size=fft_size,
    ).speaker_response


def load_impulse_response_wav_with_ir(
    source: str | Path | bytes | TextIO,
    *,
    expected_sample_rate: int | None = None,
    fft_size: int | None = None,
) -> ImpulseResponseInput:
    """Load a mono WAV impulse response and keep both original IR and response."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Reached EOF prematurely;.*",
            category=wavfile.WavFileWarning,
        )
        sample_rate, data = wavfile.read(source)
    if expected_sample_rate is not None and int(sample_rate) != int(expected_sample_rate):
        raise ValueError(
            f"impulse WAV sample rate mismatch: file={sample_rate} Hz, app={int(expected_sample_rate)} Hz"
        )
    impulse = _mono_wav_to_float64(data)
    if impulse.size < 2:
        raise ValueError("impulse WAV must contain at least 2 samples")
    if not np.all(np.isfinite(impulse)):
        raise ValueError("impulse WAV contains NaN or Inf")
    if float(np.max(np.abs(impulse))) <= 1e-12:
        raise ValueError("impulse WAV is silent or too small")

    n_fft = _response_fft_size(impulse.size, fft_size)
    spectrum = np.fft.rfft(impulse, n=n_fft)
    frequency = np.fft.rfftfreq(n_fft, 1.0 / float(sample_rate))
    magnitude = np.maximum(np.abs(spectrum), 1e-12)
    gain_db = 20.0 * np.log10(magnitude)
    phase_deg = np.rad2deg(np.unwrap(np.angle(spectrum)))
    return ImpulseResponseInput(
        speaker_response=SpeakerResponse(
            frequency=frequency.tolist(),
            gain_db=gain_db.tolist(),
            phase_deg=phase_deg.tolist(),
        ),
        impulse=impulse,
        sample_rate=int(sample_rate),
    )


def extract_first_url(text: str) -> str | None:
    match = re.search(r"https?://[^\s\]\)]+", text)
    return match.group(0) if match else None


def _mono_wav_to_float64(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    elif array.ndim != 1:
        raise ValueError("impulse WAV must be mono")

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scale = max(abs(float(info.min)), abs(float(info.max)))
        return array.astype(np.float64) / scale
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float64)
    raise ValueError(f"unsupported WAV dtype: {array.dtype}")


def _response_fft_size(sample_count: int, requested: int | None) -> int:
    if requested is not None and int(requested) >= int(sample_count):
        return int(requested)
    sample_count = int(sample_count)
    return sample_count if sample_count % 2 == 0 else sample_count + 1


def smooth_speaker_response(
    speaker: SpeakerResponse | None,
    smoothing_fraction: float,
    mode: str = "gain_phase",
) -> SpeakerResponse | None:
    if speaker is None or smoothing_fraction <= 0:
        return speaker

    frequency = np.asarray(speaker.frequency, dtype=float)
    gain = np.asarray(speaker.gain_db, dtype=float)
    if mode == "complex_gain" and speaker.phase_deg is not None:
        unwrapped_phase = np.rad2deg(np.unwrap(np.deg2rad(np.asarray(speaker.phase_deg, dtype=float))))
        complex_response = 10.0 ** (gain / 20.0) * np.exp(1j * np.deg2rad(unwrapped_phase))
        smoothed_complex = _fractional_octave_smooth_complex(
            frequency,
            complex_response,
            smoothing_fraction,
        )
        return SpeakerResponse(
            frequency=speaker.frequency,
            gain_db=(20.0 * np.log10(np.maximum(np.abs(smoothed_complex), 1e-12))).tolist(),
            phase_deg=np.rad2deg(np.unwrap(np.angle(smoothed_complex))).tolist(),
        )

    smoothed_gain = fractional_octave_smooth(frequency, gain, smoothing_fraction)

    smoothed_phase = None
    if speaker.phase_deg is not None:
        unwrapped_phase = np.rad2deg(np.unwrap(np.deg2rad(np.asarray(speaker.phase_deg, dtype=float))))
        smoothed_phase = fractional_octave_smooth(
            frequency,
            unwrapped_phase,
            smoothing_fraction,
        ).tolist()

    return SpeakerResponse(
        frequency=speaker.frequency,
        gain_db=smoothed_gain.tolist(),
        phase_deg=smoothed_phase,
    )


def estimate_speaker_auto_gain_shift(
    speaker: SpeakerResponse | None,
    *,
    manual_shift_db: float = 0.0,
    source_impulse: np.ndarray | None = None,
    source_sample_rate: int | None = None,
    peak_window_db: float = 15.0,
    low_noise_guard_hz: float = 80.0,
    measured_range: tuple[float, float] | None = None,
) -> SpeakerGainShiftEstimate:
    # Keep legacy keyword arguments for callers; the shared passband detector
    # uses magnitude alone, independent of IR/phase availability.
    from .level_matching import estimate_input_target_match

    if speaker is None:
        return SpeakerGainShiftEstimate(float(manual_shift_db), 0.0, 0.0, -3.0, 0)
    frequency = np.asarray(speaker.frequency, dtype=float)
    gain = np.asarray(speaker.gain_db, dtype=float)
    valid = np.isfinite(frequency) & np.isfinite(gain) & (frequency > 0.0)
    order = np.argsort(frequency[valid])
    frequency, gain = frequency[valid][order], gain[valid][order]
    if len(frequency) < 2:
        return SpeakerGainShiftEstimate(float(manual_shift_db), 0.0, 0.0, -3.0, 0,
            message="実測帯域が不足しています。手動シフトで調整してください。")
    clean = SpeakerResponse(frequency.tolist(), gain.tolist())
    flat = SpeakerResponse(frequency.tolist(), np.zeros_like(gain).tolist())
    window = min(abs(float(peak_window_db)), 3.0)
    match = estimate_input_target_match(clean, flat, measured_range=measured_range,
        trim_db=manual_shift_db, plateau_window_db=window, allow_automatic_fallback=True)
    lo, hi = max(5.0, float(frequency[0])), float(frequency[-1])
    if measured_range is not None:
        lo, hi = max(lo, measured_range[0]), min(hi, measured_range[1])
    peak = float(np.percentile(np.interp(np.geomspace(lo, hi, 2049), frequency, gain), 98)) if hi > lo else 0.0
    if not match.valid:
        return SpeakerGainShiftEstimate(float(manual_shift_db), 0.0, peak, peak-window, 0,
            message="スピーカー測定の高く平坦な帯域を検出できません。手動シフトで調整してください。")
    count = max(1, int(np.log2(match.high_hz / match.low_hz) * 96) + 1)
    return SpeakerGainShiftEstimate(match.shift_db, float(manual_shift_db)-match.shift_db,
        peak, peak-window, count, reference_low_hz=match.low_hz,
        reference_high_hz=match.high_hz)


def estimate_target_auto_gain_shift(
    target: SpeakerResponse | None,
    speaker: SpeakerResponse | None,
    *,
    manual_shift_db: float = 0.0,
    peak_window_db: float = 15.0,
    low_noise_guard_hz: float = 80.0,
    measured_range: tuple[float, float] | None = None,
) -> TargetGainShiftEstimate:
    if target is None or speaker is None:
        return TargetGainShiftEstimate(float(manual_shift_db), 0.0, -abs(float(peak_window_db)), 0)

    target_freq = np.asarray(target.frequency, dtype=float)
    target_gain = np.asarray(target.gain_db, dtype=float)
    speaker_freq = np.asarray(speaker.frequency, dtype=float)
    speaker_gain = np.asarray(speaker.gain_db, dtype=float)
    target_valid = np.isfinite(target_freq) & np.isfinite(target_gain) & (target_freq > 0.0)
    speaker_valid = np.isfinite(speaker_freq) & np.isfinite(speaker_gain) & (speaker_freq > 0.0)
    if np.count_nonzero(target_valid) < 1 or np.count_nonzero(speaker_valid) < 2:
        return TargetGainShiftEstimate(float(manual_shift_db), 0.0, -abs(float(peak_window_db)), 0)

    target_freq = target_freq[target_valid]
    target_gain = target_gain[target_valid]
    target_order = np.argsort(target_freq)
    target_freq = target_freq[target_order]
    target_gain = target_gain[target_order]
    speaker_freq = speaker_freq[speaker_valid]
    speaker_gain = speaker_gain[speaker_valid]
    speaker_order = np.argsort(speaker_freq)
    speaker_freq = speaker_freq[speaker_order]
    speaker_gain = speaker_gain[speaker_order]

    f_min = max(float(target_freq[0]), float(speaker_freq[0]))
    f_max = min(float(target_freq[-1]), float(speaker_freq[-1]))
    if measured_range is not None:
        f_min = max(f_min, measured_range[0])
        f_max = min(f_max, measured_range[1])
    if f_max > low_noise_guard_hz * 2**(1/3):
        f_min = max(f_min, low_noise_guard_hz)
    if f_max <= f_min:
        return TargetGainShiftEstimate(float(manual_shift_db), 0.0, 0.0, 0,
                                       message="共通の実測帯域がありません。")
    # A two-point flat Target must not sample only the speaker's noisy edges.
    # Equal octave spacing also prevents dense FFT bins dominating the level.
    grid = np.geomspace(f_min, f_max, 2049)
    speaker_grid = np.interp(grid, speaker_freq, speaker_gain)
    target_grid = np.interp(grid, target_freq, target_gain)
    from .level_matching import estimate_input_target_match
    # Reverse the shift direction: the real speaker is the reference here;
    # an intentionally tilted Target need not itself provide a flat plateau.
    match = estimate_input_target_match(
        SpeakerResponse(grid.tolist(), target_grid.tolist()),
        SpeakerResponse(grid.tolist(), speaker_grid.tolist()),
        trim_db=manual_shift_db,
        plateau_window_db=min(abs(float(peak_window_db)), 3.0),
        require_speaker_plateau=False,
        allow_automatic_fallback=True,
    )
    peak = float(np.percentile(speaker_grid, 98))
    if not match.valid:
        return TargetGainShiftEstimate(float(manual_shift_db), peak, peak-3.0, 0,
            message="スピーカー測定の高く平坦な帯域を検出できません。手動シフトで調整してください。")
    count = int(np.count_nonzero((grid >= match.low_hz) & (grid <= match.high_hz)))
    return TargetGainShiftEstimate(match.shift_db, peak, peak-3.0, count,
                                   match.low_hz, match.high_hz)


def _fractional_octave_smooth_complex(
    frequency: np.ndarray,
    values: np.ndarray,
    smoothing_fraction: float,
) -> np.ndarray:
    if smoothing_fraction <= 0:
        return values.copy()
    smoothed = values.astype(complex, copy=True)
    positive = frequency > 0
    if not np.any(positive):
        return smoothed
    log_freq = np.log2(frequency[positive])
    source_values = values[positive]
    half_width_oct = 0.5 / smoothing_fraction
    positive_indices = np.flatnonzero(positive)
    # Nonfinite input keeps the legacy local-window propagation semantics.
    if not np.all(np.isfinite(log_freq)) or not np.all(np.isfinite(source_values)):
        for idx, center in enumerate(log_freq):
            mask = np.abs(log_freq - center) <= half_width_oct
            if np.any(mask):
                smoothed[positive_indices[idx]] = complex(np.mean(source_values[mask]))
        return smoothed
    order = np.argsort(log_freq, kind="stable")
    centers, samples = log_freq[order], source_values[order]
    left = np.searchsorted(centers, centers - half_width_oct, side="left")
    right = np.searchsorted(centers, centers + half_width_oct, side="right")
    # Resolve floating-point ties using the original abs-distance predicate.
    for bounds, direction in ((left, -1), (right, 1)):
        candidate = bounds - 1 if direction < 0 else bounds
        valid = (candidate >= 0) & (candidate < centers.size)
        include = valid & (np.abs(centers[np.clip(candidate, 0, centers.size-1)]-centers) <= half_width_oct)
        bounds[include] += direction
    while np.any((left < right) & (np.abs(centers[left]-centers) > half_width_oct)):
        left[(left < right) & (np.abs(centers[left]-centers) > half_width_oct)] += 1
    while np.any((right > left) & (np.abs(centers[np.maximum(right-1, 0)]-centers) > half_width_oct)):
        right[(right > left) & (np.abs(centers[np.maximum(right-1, 0)]-centers) > half_width_oct)] -= 1
    counts = np.maximum(right-left, 1)
    with np.errstate(over="ignore", invalid="ignore"):
        cumulative = np.concatenate(([0j], np.cumsum(samples)))
        totals = cumulative[right] - cumulative[left]
        averaged = totals / counts
    # Prefix subtraction can lose tiny responses after very large values, or
    # near exact cancellation. Re-evaluate just those windows, not all bins.
    with np.errstate(over="ignore", invalid="ignore"):
        scale = np.concatenate(([0.0], np.cumsum(np.abs(samples))))
        unstable = ((scale[right] > 0) & (np.abs(totals) <= 1e-10 * scale[right])) | ~np.isfinite(averaged)
    for idx in np.flatnonzero(unstable):
        averaged[idx] = np.mean(samples[left[idx]:right[idx]])
    smoothed[positive_indices[order]] = averaged
    return smoothed


def _speaker_gain_shift_wavelet_weight(
    frequency: np.ndarray,
    *,
    speaker: SpeakerResponse,
    source_impulse: np.ndarray | None,
    source_sample_rate: int | None,
    low_noise_guard_hz: float,
) -> np.ndarray | None:
    if source_sample_rate is None:
        return None
    sample_rate = int(source_sample_rate)
    reconstruction_confidence = 1.0
    if source_impulse is not None:
        impulse = np.asarray(source_impulse, dtype=float).ravel()
    else:
        source = speaker_response_wavelet_impulse_source(speaker, sample_rate)
        if source is None:
            return None
        impulse = source.impulse
        reconstruction_confidence = source.reconstruction_confidence
    if impulse is None:
        return None
    if impulse.size < 32 or sample_rate <= 0 or not np.any(np.isfinite(impulse)):
        return None

    segment = _wavelet_gain_shift_segment(impulse, sample_rate)
    if segment.size < 32:
        return None
    f_min = max(float(low_noise_guard_hz), 50.0)
    f_max = min(float(np.nanmax(frequency)), float(sample_rate) / 2.0, 20_000.0)
    if not np.isfinite(f_max) or f_max <= f_min:
        return None
    duration_ms = segment.size * 1000.0 / float(sample_rate)
    settings = WaveletSettings(
        f_min=f_min,
        f_max=f_max,
        t_min_ms=-duration_ms / 2.0,
        t_max_ms=duration_ms / 2.0,
        bandwidth_oct=1.0 / 3.0,
        dynamic_range_db=60.0,
        frequency_bins=64,
        time_bins=192,
        mask_floor_db=-80.0,
    )
    try:
        wavelet_map = complex_morlet_scalogram(segment, sample_rate, settings, label="Gain Shift")
    except (ValueError, FloatingPointError):
        return None

    wavelet_frequency = np.asarray(wavelet_map.frequency_hz, dtype=float)
    if wavelet_frequency.size < 2:
        return None
    confidence = np.asarray(
        wavelet_map.confidence if wavelet_map.confidence is not None else np.ones_like(wavelet_frequency),
        dtype=float,
    )
    reflection = np.asarray(
        wavelet_map.reflection_index if wavelet_map.reflection_index is not None else np.zeros_like(wavelet_frequency),
        dtype=float,
    )
    spread = np.asarray(
        wavelet_map.spread_ms if wavelet_map.spread_ms is not None else np.zeros_like(wavelet_frequency),
        dtype=float,
    )
    level_db = np.asarray(wavelet_map.level_db, dtype=float)
    if level_db.ndim != 2 or level_db.shape[0] != wavelet_frequency.size:
        return None
    finite_level = np.isfinite(level_db)
    row_level = np.full(wavelet_frequency.shape, -80.0, dtype=float)
    valid_rows = np.any(finite_level, axis=1)
    if np.any(valid_rows):
        row_level[valid_rows] = np.max(
            np.where(finite_level[valid_rows], level_db[valid_rows], -np.inf),
            axis=1,
        )
    quality = (
        np.clip(np.nan_to_num(confidence, nan=0.0), 0.0, 1.0)
        * (1.0 / (1.0 + np.maximum(np.nan_to_num(reflection, nan=1.0), 0.0)))
        * (1.0 / (1.0 + np.maximum(np.nan_to_num(spread, nan=10.0), 0.0) / 8.0))
    )
    quality *= float(np.clip(reconstruction_confidence, 0.0, 1.0))
    active = np.clip((row_level + 30.0) / 30.0, 0.0, 1.0)
    quality *= active
    return np.interp(
        np.log(np.maximum(np.asarray(frequency, dtype=float), 1e-9)),
        np.log(np.maximum(wavelet_frequency, 1e-9)),
        quality,
        left=float(quality[0]),
        right=float(quality[-1]),
    )


def speaker_response_to_impulse_for_wavelet(speaker: SpeakerResponse, sample_rate: int) -> np.ndarray | None:
    """Reconstruct a wavelet-analysis impulse from a phase-bearing response."""
    source = speaker_response_wavelet_impulse_source(speaker, sample_rate)
    return None if source is None else source.impulse


def speaker_response_wavelet_impulse_source(
    speaker: SpeakerResponse,
    sample_rate: int,
) -> WaveletImpulseSource | None:
    if sample_rate <= 0:
        return None
    frequency = np.asarray(speaker.frequency, dtype=float)
    gain = np.asarray(speaker.gain_db, dtype=float)
    phase = None if speaker.phase_deg is None else np.asarray(speaker.phase_deg, dtype=float)
    valid = np.isfinite(frequency) & np.isfinite(gain) & (frequency >= 0.0)
    if phase is not None:
        valid &= np.isfinite(phase)
    if np.count_nonzero(valid) < 2:
        return None
    frequency = frequency[valid]
    gain = gain[valid]
    if phase is not None:
        phase = np.rad2deg(np.unwrap(np.deg2rad(phase[valid])))
    order = np.argsort(frequency)
    frequency = frequency[order]
    gain = gain[order]
    if phase is not None:
        phase = phase[order]
    frequency, unique_indices = np.unique(frequency, return_index=True)
    gain = gain[unique_indices]
    if phase is not None:
        phase = phase[unique_indices]
    if frequency.size < 2:
        return None

    smoothing = estimate_response_smoothing_oct(frequency, gain)
    phase_available = phase is not None
    source_type = "reconstructed_from_gain_phase" if phase_available else "reconstructed_from_minimum_phase"
    base_confidence = 0.90 if phase_available else 0.60
    source_warning = None
    if not phase_available:
        source_warning = "Minimum-phase reconstruction is advisory, not equivalent to measured IR."
    if smoothing["smoothing_detected"]:
        source_type = "reconstructed_from_smoothed_response"
        base_confidence = min(base_confidence, float(smoothing["confidence_multiplier"]))
        source_warning = (
            "Response appears smoothed; reconstructed Wavelet confidence was reduced."
            if source_warning is None
            else f"{source_warning} Response appears smoothed; reconstructed Wavelet confidence was reduced."
        )

    impulse = _speaker_response_to_impulse_for_wavelet_arrays(frequency, gain, phase, int(sample_rate))
    if impulse is None:
        return None
    if not np.any(np.isfinite(impulse)):
        return None
    impulse = np.where(np.isfinite(impulse), impulse, 0.0)
    return WaveletImpulseSource(
        impulse=impulse,
        source_type=source_type,
        reconstruction_confidence=float(np.clip(base_confidence, 0.0, 1.0)),
        phase_available=phase_available,
        smoothing_detected=bool(smoothing["smoothing_detected"]),
        estimated_smoothing_oct=smoothing["estimated_smoothing_oct"],
        smoothing_likelihood=float(smoothing["smoothing_likelihood"]),
        source_warning=source_warning,
    )


def _speaker_response_to_impulse_for_wavelet(speaker: SpeakerResponse, sample_rate: int) -> np.ndarray | None:
    if speaker.phase_deg is None or sample_rate <= 0:
        return None
    frequency = np.asarray(speaker.frequency, dtype=float)
    gain = np.asarray(speaker.gain_db, dtype=float)
    phase = np.asarray(speaker.phase_deg, dtype=float)
    valid = (
        np.isfinite(frequency)
        & np.isfinite(gain)
        & np.isfinite(phase)
        & (frequency >= 0.0)
    )
    if np.count_nonzero(valid) < 2:
        return None
    frequency = frequency[valid]
    gain = gain[valid]
    phase = np.rad2deg(np.unwrap(np.deg2rad(phase[valid])))
    order = np.argsort(frequency)
    frequency = frequency[order]
    gain = gain[order]
    phase = phase[order]
    frequency, unique_indices = np.unique(frequency, return_index=True)
    gain = gain[unique_indices]
    phase = phase[unique_indices]
    if frequency.size < 2:
        return None
    return _speaker_response_to_impulse_for_wavelet_arrays(frequency, gain, phase, sample_rate)


def _speaker_response_to_impulse_for_wavelet_arrays(
    frequency: np.ndarray,
    gain: np.ndarray,
    phase: np.ndarray | None,
    sample_rate: int,
) -> np.ndarray | None:
    if sample_rate <= 0:
        return None
    n_fft = 1 << int(np.ceil(np.log2(max(int(sample_rate), 16_384))))
    fft_frequency = np.fft.rfftfreq(n_fft, 1.0 / float(sample_rate))
    interp_gain = _interp_hold_log_frequency(fft_frequency, frequency, gain)
    magnitude = np.power(10.0, interp_gain / 20.0)
    if phase is None:
        response = _minimum_phase_spectrum_from_magnitude(magnitude, n_fft)
    else:
        interp_phase = interpolate_phase(frequency, phase, fft_frequency)
        response = magnitude * np.exp(1j * np.deg2rad(interp_phase))
    response[0] = complex(float(np.real(response[0])), 0.0)
    if n_fft % 2 == 0:
        response[-1] = complex(float(np.real(response[-1])), 0.0)
    return np.fft.irfft(response, n=n_fft)


def _minimum_phase_spectrum_from_magnitude(magnitude: np.ndarray, n_fft: int) -> np.ndarray:
    log_magnitude = np.log(np.maximum(np.asarray(magnitude, dtype=float), 1e-12))
    cepstrum = np.fft.irfft(log_magnitude, n=int(n_fft))
    minimum_phase_cepstrum = np.zeros_like(cepstrum)
    minimum_phase_cepstrum[0] = cepstrum[0]
    half = int(n_fft) // 2
    if int(n_fft) % 2 == 0:
        minimum_phase_cepstrum[1:half] = 2.0 * cepstrum[1:half]
        minimum_phase_cepstrum[half] = cepstrum[half]
    else:
        minimum_phase_cepstrum[1 : half + 1] = 2.0 * cepstrum[1 : half + 1]
    return np.exp(np.fft.rfft(minimum_phase_cepstrum, n=int(n_fft)))


def estimate_response_smoothing_oct(frequency: np.ndarray, gain_db: np.ndarray) -> dict[str, float | bool | None]:
    freq = np.asarray(frequency, dtype=float)
    gain = np.asarray(gain_db, dtype=float)
    valid = np.isfinite(freq) & np.isfinite(gain) & (freq > 0.0)
    if np.count_nonzero(valid) < 8:
        return {
            "smoothing_detected": False,
            "estimated_smoothing_oct": None,
            "smoothing_likelihood": 0.0,
            "confidence_multiplier": 1.0,
        }
    freq = freq[valid]
    gain = gain[valid]
    little_change: list[tuple[float, float]] = []
    for fraction, oct_width in ((12.0, 1.0 / 12.0), (6.0, 1.0 / 6.0), (3.0, 1.0 / 3.0)):
        if _median_fractional_octave_neighbor_count(freq, fraction) < 3:
            continue
        smoothed = fractional_octave_smooth(freq, gain, fraction)
        delta = gain - smoothed
        finite_delta = delta[np.isfinite(delta)]
        if finite_delta.size == 0:
            continue
        rms = float(np.sqrt(np.mean(finite_delta * finite_delta)))
        peak95 = float(np.percentile(np.abs(finite_delta), 95.0))
        score = max(rms / 0.15, peak95 / 0.5)
        if rms < 0.15 and peak95 < 0.5:
            likelihood = float(np.clip(1.0 - score * 0.5, 0.0, 1.0))
            little_change.append((oct_width, likelihood))
    if not little_change:
        return {
            "smoothing_detected": False,
            "estimated_smoothing_oct": None,
            "smoothing_likelihood": 0.0,
            "confidence_multiplier": 1.0,
        }
    estimated_oct, likelihood = little_change[-1]
    if estimated_oct <= 1.0 / 12.0 + 1e-9:
        return {
            "smoothing_detected": False,
            "estimated_smoothing_oct": None,
            "smoothing_likelihood": float(likelihood),
            "confidence_multiplier": 1.0,
        }
    if estimated_oct <= 1.0 / 6.0 + 1e-9:
        multiplier = 0.75
    else:
        multiplier = 0.55
    return {
        "smoothing_detected": True,
        "estimated_smoothing_oct": float(estimated_oct),
        "smoothing_likelihood": float(likelihood),
        "confidence_multiplier": float(multiplier),
    }


def _median_fractional_octave_neighbor_count(frequency: np.ndarray, smoothing_fraction: float) -> float:
    if smoothing_fraction <= 0:
        return 0.0
    freq = np.asarray(frequency, dtype=float)
    freq = freq[np.isfinite(freq) & (freq > 0.0)]
    if freq.size == 0:
        return 0.0
    log_freq = np.sort(np.log2(freq))
    half_width_oct = 0.5 / float(smoothing_fraction)
    left = np.searchsorted(log_freq, log_freq - half_width_oct, side="left")
    right = np.searchsorted(log_freq, log_freq + half_width_oct, side="right")
    return float(np.median(np.maximum(right - left, 0)))


def _interp_hold_log_frequency(
    frequency: np.ndarray,
    source_frequency: np.ndarray,
    source_value: np.ndarray,
) -> np.ndarray:
    return interpolate_values(source_frequency, source_value, frequency, axis="log")


def _wavelet_gain_shift_segment(impulse: np.ndarray, sample_rate: int) -> np.ndarray:
    finite_impulse = np.where(np.isfinite(impulse), impulse, 0.0)
    peak_index = int(np.argmax(np.abs(finite_impulse)))
    pre = int(round(0.08 * float(sample_rate)))
    post = int(round(0.16 * float(sample_rate)))
    start = max(0, peak_index - pre)
    end = min(finite_impulse.size, peak_index + post)
    return finite_impulse[start:end].astype(float, copy=False)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    midpoint = 0.5 * float(cumulative[-1])
    return float(values[int(np.searchsorted(cumulative, midpoint, side="left"))])


def _numeric_rows(lines: Iterable[str], delimiter: str | None = None) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in lines:
        cleaned = line.split("#", 1)[0].split(";", 1)[0].strip()
        if not cleaned:
            continue
        tokens = cleaned.split(delimiter) if delimiter else cleaned.replace(",", " ").split()
        values: list[float] = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                break
        if len(values) >= 2:
            rows.append(values[:3])
    return rows


def _response_from_rows(rows: list[list[float]]) -> SpeakerResponse:
    if not rows:
        raise ValueError("speaker response must contain numeric frequency and gain rows")
    if any(len(row) < 2 for row in rows):
        raise ValueError("speaker response must contain at least frequency and gain columns")
    ordered_rows = sorted(rows, key=lambda row: row[0])
    frequency = [row[0] for row in ordered_rows]
    gain_db = [row[1] for row in ordered_rows]
    phase = [row[2] for row in ordered_rows] if all(len(row) >= 3 for row in ordered_rows) else None
    return SpeakerResponse(
        frequency=frequency,
        gain_db=gain_db,
        phase_deg=phase,
    )
