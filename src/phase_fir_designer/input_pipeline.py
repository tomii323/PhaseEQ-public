from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .config import SpeakerResponse
from response_display.cache import content_cached
from .speaker import (
    SpeakerGainShiftEstimate,
    estimate_speaker_auto_gain_shift,
    smooth_speaker_response,
)
from .speaker_phase import PhaseCenterInfo, phase_center_from_impulse_with_info


# Each stage key contains only its own inputs. A final Gain edit must not
# repeat centering/extension/smoothing. Cache restores own their arrays.
@content_cached(revision="input-center-v1", max_entries=4, max_bytes=8 * 1024 * 1024)
def _center_cached(*args, **kwargs):
    return phase_center_from_impulse_with_info(*args, **kwargs)


@content_cached(revision="input-extension-v2", max_entries=4, max_bytes=8 * 1024 * 1024)
def _extend_cached(*args, **kwargs):
    from response_completion import complete_speaker
    speaker = args[0]
    if speaker is None or not kwargs.get("enabled"):
        return speaker
    return complete_speaker(speaker, sample_rate=kwargs["sample_rate"])


@content_cached(revision="input-smooth-v1", max_entries=4, max_bytes=8 * 1024 * 1024)
def _smooth_cached(*args, **kwargs):
    return smooth_speaker_response(*args, **kwargs)


@content_cached(revision="input-auto-level-v1", max_entries=4, max_bytes=8 * 1024 * 1024)
def _auto_level_cached(*args, **kwargs):
    return estimate_speaker_auto_gain_shift(*args, **kwargs)


@dataclass(frozen=True)
class InputProcessingSettings:
    sample_rate: int
    analysis_fft_size: int = 16_384
    apply_mic_calibration: bool = False
    polarity_invert: bool = False
    phase_center_enabled: bool = True
    phase_center_mode: str = "Wavelet Hybrid"
    phase_center_manual_offset_ms: float = 0.0
    phase_mode: str = "Use measured phase"
    band_extension_enabled: bool = False
    hf_extension_start_hz: float = 12_000.0
    hf_custom_slope_db_per_oct: float = -3.0
    extension_transition_oct: float = 1.0
    extension_strength: float = 1.0
    hf_gain_mode: str = "Custom slope"
    hf_phase_mode: str = "Off"
    lf_extension_mode: str = "Off"
    hf_phase_fit_start_hz: float = 12_000.0
    hf_min_slope_db_per_oct: float = -24.0
    hf_max_slope_db_per_oct: float = 0.0
    smoothing_fraction: float = 0.0
    smoothing_mode: str = "gain_phase"
    gain_shift_mode: str = "Off"
    manual_gain_shift_db: float = 0.0


@dataclass(frozen=True)
class InputSource:
    response: SpeakerResponse | None
    mic_calibration: SpeakerResponse | None = None
    source_impulse: np.ndarray | None = field(default=None, repr=False, compare=False)
    source_sample_rate: int | None = None
    near_field_response: SpeakerResponse | None = None
    port_response: SpeakerResponse | None = None


@dataclass(frozen=True)
class InputProcessingResult:
    """Read-only stage views: unchanged lists may be shared within one result.

    To edit a stage, replace the response and copy the columns being changed.
    Cache restores and independent calls own separate mutable containers.
    """
    original_response: SpeakerResponse | None
    calibrated_response: SpeakerResponse | None
    polarity_response: SpeakerResponse | None
    centered_response: SpeakerResponse | None
    phase_adjusted_response: SpeakerResponse | None
    extended_response: SpeakerResponse | None
    smoothed_response: SpeakerResponse | None
    processed_response: SpeakerResponse | None
    applied_gain_shift_db: float
    gain_shift_estimate: SpeakerGainShiftEstimate
    phase_center_info: PhaseCenterInfo
    messages: tuple[str, ...] = ()
    phase_origin: str = "unknown"
    input_range_hz: tuple[float, float] | None = None
    processed_range_hz: tuple[float, float] | None = None
    extension_affected_ranges: tuple[tuple[str, float, float], ...] = ()


def apply_mic_calibration(
    speaker: SpeakerResponse | None,
    mic_calibration: SpeakerResponse | None,
    *, share_unchanged: bool = False,
) -> SpeakerResponse | None:
    if speaker is None or mic_calibration is None:
        return speaker
    speaker_frequency = np.asarray(speaker.frequency, dtype=float)
    calibration_frequency = np.asarray(mic_calibration.frequency, dtype=float)
    calibration_gain = np.asarray(mic_calibration.gain_db, dtype=float)
    valid = (
        np.isfinite(calibration_frequency)
        & np.isfinite(calibration_gain)
        & (calibration_frequency >= 0.0)
    )
    if not np.any(valid):
        return speaker
    order = np.argsort(calibration_frequency[valid])
    frequency = calibration_frequency[valid][order]
    gain = calibration_gain[valid][order]
    # Calibration data is the microphone deviation, not an additive EQ curve.
    corrected = np.asarray(speaker.gain_db, dtype=float) - np.interp(
        speaker_frequency,
        frequency,
        gain,
        left=float(gain[0]),
        right=float(gain[-1]),
    )
    return SpeakerResponse(
        frequency=_response_column(speaker.frequency, share_unchanged),
        gain_db=corrected.tolist(),
        phase_deg=_response_column(speaker.phase_deg, share_unchanged),
    )


@content_cached(revision="input-processing-v5-shared-columns", max_entries=8)
def process_input(source: InputSource, settings: InputProcessingSettings) -> InputProcessingResult:
    if int(settings.sample_rate) <= 0:
        raise ValueError("sample_rate must be positive")
    messages: list[str] = []
    # Own the mutable containers once; float values are immutable. Result
    # stages may share columns, but never mutate the caller's input containers.
    original = None if source.response is None else replace(
        source.response, frequency=list(source.response.frequency),
        gain_db=list(source.response.gain_db),
        phase_deg=_response_column(source.response.phase_deg, False),
    )
    calibrated = (
        apply_mic_calibration(original, source.mic_calibration, share_unchanged=True)
        if settings.apply_mic_calibration
        else original
    )
    if settings.apply_mic_calibration and source.mic_calibration is not None:
        messages.append("Mic Calibration applied")

    polarity_response = invert_response_polarity(calibrated, share_unchanged=True) if settings.polarity_invert else calibrated
    source_impulse = None if source.source_impulse is None else np.asarray(source.source_impulse, dtype=float)
    if settings.polarity_invert and source_impulse is not None:
        source_impulse = -source_impulse
        messages.append("Polarity inverted before phase centering")

    centered, center_info = _center_cached(
        polarity_response,
        int(settings.sample_rate),
        enabled=bool(settings.phase_center_enabled),
        mode=str(settings.phase_center_mode),
        search_start_ms=None,
        search_end_ms=None,
        manual_offset_ms=float(settings.phase_center_manual_offset_ms),
        fft_size=int(settings.analysis_fft_size),
        source_impulse=source_impulse,
        source_sample_rate=source.source_sample_rate,
    )
    # Stage caches restore independent objects. Reconnect only columns whose
    # invariance follows from the operation, without scanning sample values.
    centered = _reconnect_columns(centered, polarity_response, "frequency", "gain_db")
    if (not settings.phase_center_enabled or settings.phase_center_mode == "Off"
            or polarity_response is None or polarity_response.phase_deg is None):
        centered = polarity_response
    phase_adjusted = apply_phase_mode(centered, settings.phase_mode, share_unchanged=True)
    extended = _extend_cached(
        phase_adjusted,
        enabled=bool(settings.band_extension_enabled),
        sample_rate=int(settings.sample_rate),
    )
    if not settings.band_extension_enabled:
        extended = phase_adjusted
    if settings.band_extension_enabled and phase_adjusted is not None:
        f = np.asarray(phase_adjusted.frequency, float)
        if len(f) >= 2 and f[-1] < settings.sample_rate / 2 and np.count_nonzero(f >= f[-1] / 2**(1/6)) < 3:
            messages.append("Hi補完：近傍の測定点が不足するためGainを端値保持しています。")
    smoothing_label = str(settings.smoothing_mode)
    smoothing_mode = "complex_gain" if smoothing_label == "Complex Gain" else smoothing_label
    if smoothing_label == "Gain + Phase":
        smoothing_mode = "gain_phase"
    if smoothing_mode not in {"gain_phase", "complex_gain"}:
        smoothing_mode = "gain_phase"
    smoothed = _smooth_cached(extended, float(settings.smoothing_fraction), mode=smoothing_mode)
    if settings.smoothing_fraction <= 0:
        smoothed = extended
    else:
        smoothed = _reconnect_columns(smoothed, extended, "frequency")
    smoothed = unwrap_response_phase(smoothed, share_unchanged=True)

    mode = str(settings.gain_shift_mode).lower()
    if mode in {"auto", "automatic"} or mode.startswith("auto:"):
        gain_estimate = _auto_level_cached(
            smoothed,
            manual_shift_db=0.0,
            source_impulse=source_impulse,
            source_sample_rate=source.source_sample_rate or int(settings.sample_rate),
            measured_range=_response_range(original),
        )
        gain_estimate = replace(gain_estimate, shift_db=gain_estimate.shift_db + float(settings.manual_gain_shift_db))
    elif mode in {"off", "none", "disabled"}:
        gain_estimate = estimate_speaker_auto_gain_shift(None, manual_shift_db=0.0)
    else:
        gain_estimate = estimate_speaker_auto_gain_shift(
            None,
            manual_shift_db=float(settings.manual_gain_shift_db),
        )
    if gain_estimate.message:
        messages.append(gain_estimate.message)
    processed = shift_response_gain(smoothed, gain_estimate.shift_db, share_unchanged=True)
    return InputProcessingResult(
        original_response=original,
        calibrated_response=calibrated,
        polarity_response=polarity_response,
        centered_response=centered,
        phase_adjusted_response=phase_adjusted,
        extended_response=extended,
        smoothed_response=smoothed,
        processed_response=processed,
        applied_gain_shift_db=float(gain_estimate.shift_db),
        gain_shift_estimate=gain_estimate,
        phase_center_info=center_info,
        messages=tuple(messages),
        phase_origin=("missing" if processed is None or processed.phase_deg is None else
                      "estimated_minimum_phase" if str(settings.phase_mode).strip().lower() in
                      {"gainのみから最小位相を生成", "minimum phase", "minimum_phase"} else "measured"),
        input_range_hz=_response_range(original),
        processed_range_hz=_response_range(processed),
        extension_affected_ranges=_extension_changes(phase_adjusted, extended),
    )


def _response_range(response):
    if response is None or len(response.frequency) < 2:
        return None
    return float(min(response.frequency)), float(max(response.frequency))


@content_cached(revision="input-extension-provenance-v1", max_entries=4, max_bytes=1024 * 1024)
def _extension_changes(before, after):
    """Record changes at original samples, separately from added outer bands."""
    if before is None or after is None:
        return ()
    f = np.asarray(before.frequency, float)
    x = np.asarray(after.frequency, float)
    ranges = []
    for label, a, b in (("Gain", before.gain_db, after.gain_db),
                        ("Phase", before.phase_deg, after.phase_deg)):
        if a is None or b is None:
            continue
        changed = np.abs(np.asarray(a) - np.interp(f, x, b)) > 1e-8
        edges = np.diff(np.r_[False, changed, False].astype(int))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            ranges.append((label, float(f[start]), float(f[end - 1])))
    return tuple(ranges)


def _response_column(values, share):
    return values if values is None or share else list(values)


def _reconnect_columns(response, source, *columns):
    """Restore operation-guaranteed aliases only within this result graph."""
    if response is None or source is None or response is source:
        return response
    updates = {}
    for name in columns:
        current, original = getattr(response, name), getattr(source, name)
        if current is original:
            continue
        if current is None or original is None or len(current) != len(original):
            return response
        updates[name] = original
    return replace(response, **updates) if updates else response


def shift_response_gain(
    response: SpeakerResponse | None, shift_db: float, *, share_unchanged: bool = False,
) -> SpeakerResponse | None:
    if response is None:
        return None
    return SpeakerResponse(
        frequency=_response_column(response.frequency, share_unchanged),
        gain_db=(np.asarray(response.gain_db, dtype=float) + float(shift_db)).tolist(),
        phase_deg=_response_column(response.phase_deg, share_unchanged),
    )


def invert_response_polarity(
    response: SpeakerResponse | None, *, share_unchanged: bool = False,
) -> SpeakerResponse | None:
    if response is None:
        return None
    if response.phase_deg is None:
        return SpeakerResponse(
            _response_column(response.frequency, share_unchanged),
            _response_column(response.gain_db, share_unchanged), None,
        )
    phase = np.asarray(response.phase_deg, dtype=float) + 180.0
    return SpeakerResponse(
        _response_column(response.frequency, share_unchanged),
        _response_column(response.gain_db, share_unchanged), phase.tolist(),
    )


def unwrap_response_phase(
    response: SpeakerResponse | None, *, share_unchanged: bool = False,
) -> SpeakerResponse | None:
    if response is None or response.phase_deg is None:
        return response
    phase = np.rad2deg(np.unwrap(np.deg2rad(np.asarray(response.phase_deg, dtype=float))))
    return SpeakerResponse(
        _response_column(response.frequency, share_unchanged),
        _response_column(response.gain_db, share_unchanged), phase.tolist(),
    )


def apply_phase_mode(
    response: SpeakerResponse | None, phase_mode: str, *, share_unchanged: bool = False,
) -> SpeakerResponse | None:
    if response is None:
        return None
    normalized = str(phase_mode).strip().lower()
    if normalized in {"位相を使わない", "ignore", "none", "gain only"}:
        return SpeakerResponse(
            _response_column(response.frequency, share_unchanged),
            _response_column(response.gain_db, share_unchanged), None,
        )
    if normalized in {"gainのみから最小位相を生成", "minimum phase", "minimum_phase"}:
        phase = minimum_phase_from_gain_db(
            np.asarray(response.frequency, dtype=float),
            np.asarray(response.gain_db, dtype=float),
        )
        return SpeakerResponse(
            _response_column(response.frequency, share_unchanged),
            _response_column(response.gain_db, share_unchanged), phase.tolist(),
        )
    return response


def minimum_phase_from_gain_db(frequency: np.ndarray, gain_db: np.ndarray) -> np.ndarray:
    frequency = np.asarray(frequency, dtype=float)
    gain_db = np.asarray(gain_db, dtype=float)
    valid = np.isfinite(frequency) & np.isfinite(gain_db) & (frequency >= 0.0)
    if np.count_nonzero(valid) < 2:
        return np.zeros_like(frequency)
    source_frequency = frequency[valid]
    source_gain = gain_db[valid]
    order = np.argsort(source_frequency)
    source_frequency = source_frequency[order]
    source_gain = source_gain[order]
    positive = source_frequency > 0.0
    if not np.any(positive):
        return np.zeros_like(frequency)
    f_min = float(np.min(source_frequency[positive]))
    f_max = float(np.max(source_frequency))
    n_fft = 1 << 15
    uniform_frequency = np.linspace(0.0, f_max, n_fft // 2 + 1)
    uniform_gain = np.interp(
        np.maximum(uniform_frequency, f_min),
        source_frequency,
        source_gain,
        left=float(source_gain[0]),
        right=float(source_gain[-1]),
    )
    log_magnitude = np.log(np.maximum(10.0 ** (uniform_gain / 20.0), 1e-9))
    full = np.concatenate([log_magnitude, log_magnitude[-2:0:-1]])
    cepstrum = np.fft.ifft(full).real
    minimum_cepstrum = np.zeros_like(cepstrum)
    minimum_cepstrum[0] = cepstrum[0]
    minimum_cepstrum[1 : n_fft // 2] = 2.0 * cepstrum[1 : n_fft // 2]
    minimum_cepstrum[n_fft // 2] = cepstrum[n_fft // 2]
    minimum_spectrum = np.fft.fft(minimum_cepstrum)
    uniform_phase = np.rad2deg(np.unwrap(np.imag(minimum_spectrum[: n_fft // 2 + 1])))
    return np.interp(
        np.clip(frequency, 0.0, f_max),
        uniform_frequency,
        uniform_phase,
        left=float(uniform_phase[0]),
        right=float(uniform_phase[-1]),
    )
