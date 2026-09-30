from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from response_math import interpolate_values, interpolate_phase
from scipy import signal

from wavelet_analysis import WaveletSettings, center_from_wavelet_map, complex_morlet_scalogram
from .config import SpeakerResponse

PHASE_CENTER_MODES = (
    "Wavelet Hybrid",
    "Positive peak",
    "Negative peak",
    "Absolute peak",
    "Energy centroid",
    "Manual",
    "Off",
)


@dataclass(frozen=True)
class PhaseCenterInfo:
    applied_ms: float | None
    source_type: str
    method: str
    confidence: float | None = None
    warning: str | None = None
    search_start_ms: float | None = None
    search_end_ms: float | None = None


def estimate_phase_delay_ms(
    frequency: np.ndarray,
    phase_deg: np.ndarray,
    f_min: float,
    f_max: float,
) -> float:
    """Estimate pure delay from unwrapped phase slope."""
    frequency = np.asarray(frequency, dtype=float)
    phase = np.rad2deg(np.unwrap(np.deg2rad(np.asarray(phase_deg, dtype=float))))
    valid = (
        np.isfinite(frequency)
        & np.isfinite(phase)
        & (frequency >= float(f_min))
        & (frequency <= float(f_max))
    )
    if np.count_nonzero(valid) < 2:
        return 0.0
    slope_deg_per_hz, _ = np.polyfit(frequency[valid], phase[valid], 1)
    delay_seconds = -float(slope_deg_per_hz) / 360.0
    return delay_seconds * 1000.0


def remove_phase_delay(
    speaker: SpeakerResponse | None,
    delay_ms: float,
) -> SpeakerResponse | None:
    if speaker is None or speaker.phase_deg is None:
        return speaker
    frequency = np.asarray(speaker.frequency, dtype=float)
    phase = np.rad2deg(np.unwrap(np.deg2rad(np.asarray(speaker.phase_deg, dtype=float))))
    corrected_phase = phase + 360.0 * frequency * (float(delay_ms) / 1000.0)
    return SpeakerResponse(
        frequency=speaker.frequency,
        gain_db=speaker.gain_db,
        phase_deg=corrected_phase.tolist(),
    )


def phase_center_from_impulse(
    speaker: SpeakerResponse | None,
    sample_rate: int,
    *,
    enabled: bool = True,
    mode: str = "Positive peak",
    search_start_ms: float | None = 0.0,
    search_end_ms: float | None = 50.0,
    manual_offset_ms: float = 0.0,
    fft_size: int | None = None,
    source_impulse: np.ndarray | None = None,
    source_sample_rate: int | None = None,
) -> tuple[SpeakerResponse | None, float | None]:
    """Remove delay by detecting the impulse peak reconstructed from speaker response."""
    corrected, info = phase_center_from_impulse_with_info(
        speaker,
        sample_rate,
        enabled=enabled,
        mode=mode,
        search_start_ms=search_start_ms,
        search_end_ms=search_end_ms,
        manual_offset_ms=manual_offset_ms,
        fft_size=fft_size,
        source_impulse=source_impulse,
        source_sample_rate=source_sample_rate,
    )
    return corrected, info.applied_ms


def phase_center_from_impulse_with_info(
    speaker: SpeakerResponse | None,
    sample_rate: int,
    *,
    enabled: bool = True,
    mode: str = "Positive peak",
    search_start_ms: float | None = 0.0,
    search_end_ms: float | None = 50.0,
    manual_offset_ms: float = 0.0,
    fft_size: int | None = None,
    source_impulse: np.ndarray | None = None,
    source_sample_rate: int | None = None,
) -> tuple[SpeakerResponse | None, PhaseCenterInfo]:
    """Remove bulk delay and report whether original IR or reconstructed IR was used."""
    if speaker is None or speaker.phase_deg is None or not enabled or mode == "Off":
        return speaker, PhaseCenterInfo(None, "none", "Off")
    mode = mode if mode in PHASE_CENTER_MODES else "Positive peak"
    manual_offset_ms = float(manual_offset_ms)
    if mode == "Manual":
        return (
            remove_phase_delay(speaker, manual_offset_ms),
            PhaseCenterInfo(manual_offset_ms, "manual", "Manual", 1.0),
        )

    n_fft = _phase_center_fft_size(sample_rate, fft_size)
    impulse, impulse_sample_rate, source_type = _phase_center_source_impulse(
        speaker,
        sample_rate,
        n_fft,
        source_impulse=source_impulse,
        source_sample_rate=source_sample_rate,
    )
    if impulse is None:
        return speaker, PhaseCenterInfo(None, "unavailable", "No impulse source")
    search_start_ms, search_end_ms = _resolve_phase_center_search_window_ms(
        impulse,
        impulse_sample_rate,
        search_start_ms=search_start_ms,
        search_end_ms=search_end_ms,
    )

    if mode == "Wavelet Hybrid":
        detected_ms, confidence, warning, method = _detect_wavelet_hybrid_center_ms(
            impulse,
            impulse_sample_rate,
            search_start_ms=search_start_ms,
            search_end_ms=search_end_ms,
        )
    else:
        peak_index = _detect_phase_center_index(
            impulse,
            impulse_sample_rate,
            mode=mode,
            search_start_ms=search_start_ms,
            search_end_ms=search_end_ms,
        )
        if peak_index is None:
            return speaker, PhaseCenterInfo(
                None,
                source_type,
                mode,
                None,
                "no center peak",
                search_start_ms,
                search_end_ms,
            )
        detected_ms = float(peak_index) * 1000.0 / float(impulse_sample_rate)
        confidence = None
        warning = None
        method = mode

    if detected_ms is None:
        return speaker, PhaseCenterInfo(None, source_type, method, confidence, warning, search_start_ms, search_end_ms)
    delay_ms = float(detected_ms) + manual_offset_ms
    return (
        remove_phase_delay(speaker, delay_ms),
        PhaseCenterInfo(delay_ms, source_type, method, confidence, warning, search_start_ms, search_end_ms),
    )


def _phase_center_source_impulse(
    speaker: SpeakerResponse,
    sample_rate: int,
    n_fft: int,
    *,
    source_impulse: np.ndarray | None,
    source_sample_rate: int | None,
) -> tuple[np.ndarray | None, int, str]:
    if source_impulse is not None and source_sample_rate is not None:
        impulse = np.asarray(source_impulse, dtype=float)
        if impulse.size > 0 and np.all(np.isfinite(impulse)):
            return impulse, int(source_sample_rate), "original_ir"

    response = _speaker_complex_response_on_fft_axis(speaker, sample_rate, n_fft)
    if response is None:
        return None, int(sample_rate), "reconstructed_from_response"
    return np.fft.irfft(response, n=n_fft), int(sample_rate), "reconstructed_from_response"


def _detect_wavelet_hybrid_center_ms(
    impulse: np.ndarray,
    sample_rate: int,
    *,
    search_start_ms: float,
    search_end_ms: float,
) -> tuple[float | None, float | None, str | None, str]:
    segment, offset_ms = _phase_center_search_segment(
        impulse,
        sample_rate,
        search_start_ms=search_start_ms,
        search_end_ms=search_end_ms,
    )
    if segment.size < 8:
        return None, None, "empty search segment", "Wavelet Hybrid"

    duration_ms = segment.size * 1000.0 / float(sample_rate)
    center_offset_ms = (segment.size - 1) * 500.0 / float(sample_rate)
    settings = WaveletSettings(
        f_min=100.0,
        f_max=min(10_000.0, float(sample_rate) / 2.0),
        t_min_ms=-center_offset_ms,
        t_max_ms=max(duration_ms - center_offset_ms, -center_offset_ms + 0.1),
        bandwidth_oct=1.0 / 3.0,
        dynamic_range_db=60.0,
        frequency_bins=96,
        time_bins=max(256, min(1024, int(segment.size))),
        mask_floor_db=-80.0,
    )
    wavelet_map = complex_morlet_scalogram(segment, sample_rate, settings, label="Phase Center")
    center = center_from_wavelet_map(
        wavelet_map,
        f_min=300.0,
        f_max=min(8000.0, float(sample_rate) / 2.0),
        min_confidence=0.25,
    )
    if center.method != "wavelet_unavailable" and np.isfinite(center.center_time_ms):
        return (
            offset_ms + center_offset_ms + float(center.center_time_ms),
            center.confidence,
            center.warning,
            center.method,
        )

    fallback_index = _band_limited_envelope_peak_index(segment, sample_rate)
    if fallback_index is not None:
        return (
            offset_ms + float(fallback_index) * 1000.0 / float(sample_rate),
            0.25,
            center.warning or "wavelet center fallback",
            "Band-limited envelope peak",
        )

    peak_index = _detect_phase_center_index(
        segment,
        sample_rate,
        mode="Absolute peak",
        search_start_ms=0.0,
        search_end_ms=duration_ms,
    )
    if peak_index is None:
        return None, 0.0, center.warning or "no fallback peak", "Wavelet Hybrid"
    return (
        offset_ms + float(peak_index) * 1000.0 / float(sample_rate),
        0.05,
        center.warning or "absolute peak fallback",
        "Absolute peak fallback",
    )


def _resolve_phase_center_search_window_ms(
    impulse: np.ndarray,
    sample_rate: int,
    *,
    search_start_ms: float | None,
    search_end_ms: float | None,
) -> tuple[float, float]:
    if search_start_ms is not None and search_end_ms is not None:
        start_ms = max(0.0, float(search_start_ms))
        end_ms = max(float(search_end_ms), start_ms)
        return start_ms, end_ms
    return auto_phase_center_search_window_ms(impulse, sample_rate)


def auto_phase_center_search_window_ms(
    impulse: np.ndarray,
    sample_rate: int,
    *,
    max_scan_ms: float = 1000.0,
    pre_margin_ms: float = 10.0,
    post_margin_ms: float = 30.0,
    min_window_ms: float = 10.0,
) -> tuple[float, float]:
    impulse = np.asarray(impulse, dtype=float).ravel()
    sample_rate = int(sample_rate)
    if impulse.size <= 1 or sample_rate <= 0:
        return 0.0, 50.0
    duration_ms = impulse.size * 1000.0 / float(sample_rate)
    finite = np.isfinite(impulse)
    if not np.any(finite):
        return 0.0, min(50.0, duration_ms)

    scan_end = int(round(min(float(max_scan_ms), duration_ms) * float(sample_rate) / 1000.0))
    scan_end = min(max(scan_end, 1), impulse.size)
    scan = np.where(finite[:scan_end], np.abs(impulse[:scan_end]), 0.0)
    peak_index = int(np.argmax(scan))
    peak_ms = peak_index * 1000.0 / float(sample_rate)
    start_ms = max(0.0, peak_ms - float(pre_margin_ms))
    end_ms = min(duration_ms, peak_ms + float(post_margin_ms))
    if end_ms - start_ms < float(min_window_ms):
        end_ms = min(duration_ms, start_ms + float(min_window_ms))
        start_ms = max(0.0, end_ms - float(min_window_ms))
    return start_ms, max(end_ms, start_ms + (1000.0 / float(sample_rate)))


def _phase_center_search_segment(
    impulse: np.ndarray,
    sample_rate: int,
    *,
    search_start_ms: float,
    search_end_ms: float,
) -> tuple[np.ndarray, float]:
    impulse = np.asarray(impulse, dtype=float)
    if impulse.size == 0:
        return np.asarray([], dtype=float), 0.0
    start_ms = max(0.0, float(search_start_ms))
    end_ms = max(float(search_end_ms), start_ms)
    start = int(round(start_ms * float(sample_rate) / 1000.0))
    end = int(round(end_ms * float(sample_rate) / 1000.0))
    start = min(max(start, 0), impulse.size - 1)
    end = min(max(end, start + 1), impulse.size)
    return impulse[start:end].astype(float, copy=False), start * 1000.0 / float(sample_rate)


def _band_limited_envelope_peak_index(
    impulse: np.ndarray,
    sample_rate: int,
    *,
    f_min: float = 300.0,
    f_max: float = 8000.0,
) -> int | None:
    impulse = np.asarray(impulse, dtype=float)
    if impulse.size < 8:
        return None
    nyquist = float(sample_rate) / 2.0
    high = min(float(f_max), nyquist * 0.95)
    low = max(float(f_min), 1.0)
    if high <= low:
        return int(np.argmax(np.abs(impulse)))
    try:
        sos = signal.butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
        filtered = signal.sosfiltfilt(sos, impulse) if impulse.size > 32 else signal.sosfilt(sos, impulse)
        envelope = np.abs(signal.hilbert(filtered))
        return int(np.argmax(envelope))
    except ValueError:
        return int(np.argmax(np.abs(impulse)))


def _phase_center_fft_size(sample_rate: int, requested: int | None) -> int:
    if requested is not None and int(requested) >= 1024:
        return int(requested)
    target = max(int(sample_rate), 16_384)
    return 1 << int(np.ceil(np.log2(target)))


def _speaker_complex_response_on_fft_axis(
    speaker: SpeakerResponse,
    sample_rate: int,
    n_fft: int,
) -> np.ndarray | None:
    source_frequency = np.asarray(speaker.frequency, dtype=float)
    source_gain = np.asarray(speaker.gain_db, dtype=float)
    source_phase = np.asarray(speaker.phase_deg, dtype=float)
    valid = (
        np.isfinite(source_frequency)
        & np.isfinite(source_gain)
        & np.isfinite(source_phase)
        & (source_frequency >= 0.0)
    )
    if np.count_nonzero(valid) < 2:
        return None
    source_frequency = source_frequency[valid]
    source_gain = source_gain[valid]
    source_phase = np.rad2deg(np.unwrap(np.deg2rad(source_phase[valid])))
    order = np.argsort(source_frequency)
    source_frequency = source_frequency[order]
    source_gain = source_gain[order]
    source_phase = source_phase[order]
    unique_frequency, unique_indices = np.unique(source_frequency, return_index=True)
    source_frequency = unique_frequency
    source_gain = source_gain[unique_indices]
    source_phase = source_phase[unique_indices]
    if len(source_frequency) < 2:
        return None

    frequency = np.fft.rfftfreq(n_fft, 1.0 / float(sample_rate))
    gain = _interp_hold_log_frequency(frequency, source_frequency, source_gain)
    phase = interpolate_phase(source_frequency, source_phase, frequency)
    magnitude = np.power(10.0, gain / 20.0)
    response = magnitude * np.exp(1j * np.deg2rad(phase))
    response[0] = complex(float(np.real(response[0])), 0.0)
    if n_fft % 2 == 0:
        response[-1] = complex(float(np.real(response[-1])), 0.0)
    return response


def _interp_hold_log_frequency(
    frequency: np.ndarray,
    source_frequency: np.ndarray,
    source_value: np.ndarray,
) -> np.ndarray:
    return interpolate_values(source_frequency, source_value, frequency, axis="log")


def _detect_phase_center_index(
    impulse: np.ndarray,
    sample_rate: int,
    *,
    mode: str,
    search_start_ms: float,
    search_end_ms: float,
) -> int | None:
    impulse = np.asarray(impulse, dtype=float)
    if impulse.size == 0:
        return None
    start = int(round(max(0.0, float(search_start_ms)) * float(sample_rate) / 1000.0))
    end = int(round(max(float(search_end_ms), float(search_start_ms)) * float(sample_rate) / 1000.0))
    start = min(max(start, 0), impulse.size - 1)
    end = min(max(end, start + 1), impulse.size)
    segment = impulse[start:end]
    if segment.size == 0:
        return None
    if mode == "Negative peak":
        relative = int(np.argmin(segment))
    elif mode == "Absolute peak":
        relative = int(np.argmax(np.abs(segment)))
    elif mode == "Energy centroid":
        energy = segment * segment
        energy_sum = float(np.sum(energy))
        if energy_sum <= 0.0 or not np.isfinite(energy_sum):
            relative = int(np.argmax(np.abs(segment)))
        else:
            indices = np.arange(segment.size, dtype=float)
            relative = int(round(float(np.sum(indices * energy) / energy_sum)))
            relative = min(max(relative, 0), segment.size - 1)
    else:
        relative = int(np.argmax(segment))
    return start + relative
