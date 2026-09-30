from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import signal

from wavelet_analysis import WaveletSettings, complex_morlet_scalogram, settings_from_preset

from .models import HighPrecisionResult, MeasurementSettings, ShotQuality, ShotResult, ShotStatus, readonly


def generate_ess_reference(settings: MeasurementSettings) -> np.ndarray:
    count = max(2, int(round(settings.sweep_duration_s * settings.sample_rate)))
    duration = count / settings.sample_rate
    time = np.arange(count, dtype=np.float64) / settings.sample_rate
    sweep = signal.chirp(
        time,
        f0=settings.start_frequency_hz,
        f1=min(settings.end_frequency_hz, settings.sample_rate / 2 - 1),
        t1=duration,
        method="logarithmic",
        phi=-90.0,
    )
    # Short fades avoid a discontinuity without changing the ESS timing model.
    fade = min(count // 8, max(8, int(0.005 * settings.sample_rate)))
    envelope = np.ones(count)
    envelope[:fade] = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
    envelope[-fade:] = np.cos(np.linspace(0, np.pi / 2, fade)) ** 2
    return readonly(sweep * envelope)


def normalized_correlation(recording: np.ndarray, reference: np.ndarray) -> np.ndarray:
    x = np.asarray(recording, dtype=np.float64)
    y = np.asarray(reference, dtype=np.float64)
    if x.size < y.size or y.size < 2:
        return np.empty(0, dtype=np.float64)
    centered = y - np.mean(y)
    numerator = signal.fftconvolve(x, centered[::-1], mode="valid")
    ones = np.ones(y.size, dtype=np.float64)
    sum_x = signal.fftconvolve(x, ones, mode="valid")
    sum_x2 = signal.fftconvolve(x * x, ones, mode="valid")
    variance = np.maximum(sum_x2 - sum_x * sum_x / y.size, 0.0)
    denominator = np.sqrt(variance * np.sum(centered * centered))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-15)


def detect_ess_start(recording: np.ndarray, reference: np.ndarray) -> tuple[int | None, float]:
    array = np.asarray(recording, dtype=np.float64)
    if array.ndim == 2:
        # Do not average dual-gain channels: an unexpected polarity inversion
        # would cancel the ESS and hide a condition the quality stage must
        # report. Detect on the channel with more energy instead.
        energies = np.sum(array * array, axis=0)
        mono = array[:, int(np.argmax(energies))]
    else:
        mono = array
    corr = normalized_correlation(mono, reference)
    if corr.size == 0:
        return None, 0.0
    index = int(np.argmax(np.abs(corr)))
    return index, float(abs(corr[index]))


def detect_first_ess_start(
    recording: np.ndarray,
    reference: np.ndarray,
    threshold: float,
) -> tuple[int | None, float]:
    """Return the first qualifying correlation peak in a streaming buffer."""

    array = np.asarray(recording, dtype=np.float64)
    if array.ndim == 2:
        energies = np.sum(array * array, axis=0)
        mono = array[:, int(np.argmax(energies))]
    else:
        mono = array
    scores = np.abs(normalized_correlation(mono, reference))
    if scores.size == 0:
        return None, 0.0
    qualified = np.flatnonzero(scores >= float(threshold))
    if qualified.size == 0:
        best = int(np.argmax(scores))
        return None, float(scores[best])
    first = int(qualified[0])
    end = first + 1
    while end < scores.size and scores[end] >= float(threshold):
        end += 1
    peak = first + int(np.argmax(scores[first:end]))
    return peak, float(scores[peak])


def channel_quality(stereo: np.ndarray, settings: MeasurementSettings) -> ShotQuality:
    audio = np.asarray(stereo, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == 1:
        mono = audio[:, 0]
        peak = float(np.max(np.abs(mono)))
        rms = float(np.sqrt(np.mean(mono * mono)))
        clipped = peak >= settings.clip_threshold
        noise_count = max(8, min(mono.size // 8, int(0.03 * settings.sample_rate)))
        noise_rms = float(np.sqrt(np.mean(mono[-noise_count:] ** 2)))
        snr_db = 20 * np.log10(max(rms, 1e-15) / max(noise_rms, 1e-15))
        messages: list[str] = []
        status = ShotStatus.CLIPPED if clipped else ShotStatus.VALID
        if clipped:
            messages.append("Microphone input clipped.")
        elif snr_db < 12:
            status = ShotStatus.WARNING
            messages.append("Low estimated S/N.")
        return ShotQuality(
            status=status,
            snr_db=float(snr_db),
            left_peak=peak,
            right_peak=peak,
            left_rms=rms,
            right_rms=rms,
            gain_difference_db=0.0,
            lr_correlation=1.0,
            delay_difference_samples=0.0,
            polarity_inverted=False,
            left_clipped=clipped,
            right_clipped=clipped,
            selected_channel="mono",
            messages=tuple(messages),
        )
    if audio.shape[1] < 2:
        raise ValueError("shot audio has no input channel")
    left, right = audio[:, 0], audio[:, 1]
    left_peak, right_peak = float(np.max(np.abs(left))), float(np.max(np.abs(right)))
    left_rms = float(np.sqrt(np.mean(left * left)))
    right_rms = float(np.sqrt(np.mean(right * right)))
    gain_db = 20 * np.log10(max(left_rms, 1e-15) / max(right_rms, 1e-15))
    corr = float(np.corrcoef(left, right)[0, 1]) if left.size > 1 else 0.0
    polarity = corr < 0
    cross = signal.correlate(left, right, mode="full", method="fft")
    delay = float(np.argmax(np.abs(cross)) - (right.size - 1))
    left_clip = left_peak >= settings.clip_threshold
    right_clip = right_peak >= settings.clip_threshold
    high_is_left = left_rms >= right_rms
    high_peak = left_peak if high_is_left else right_peak
    low_peak = right_peak if high_is_left else left_peak
    high_label = "L high gain" if high_is_left else "R high gain"
    low_label = "R low gain" if high_is_left else "L low gain"
    selected = high_label
    messages: list[str] = []
    status = ShotStatus.VALID
    if left_clip and right_clip:
        status = ShotStatus.CLIPPED
        selected = "none"
        messages.append("Both gain channels clipped.")
    elif left_clip or right_clip:
        status = ShotStatus.WARNING
        non_clipped_index = 1 if left_clip else 0
        selected = (
            high_label
            if non_clipped_index == (0 if high_is_left else 1)
            else low_label
        )
        messages.append("Using the non-clipped gain channel for this shot.")
    elif high_peak >= 0.90:
        status = ShotStatus.WARNING
        selected = low_label
        messages.append(
            f"High-gain peak {high_peak:.3f} has insufficient headroom; using the low-gain channel "
            f"(peak {low_peak:.3f})."
        )
    elif abs(delay) > 2 or abs(corr) < 0.8:
        status = ShotStatus.WARNING
        messages.append("L/R alignment or correlation needs review.")
    noise_count = max(8, min(left.size // 8, int(0.03 * settings.sample_rate)))
    signal_rms = max(left_rms, right_rms)
    # The ESS period ends with nominal silence; use that region as the noise
    # estimate instead of the beginning, where the sweep is already active.
    noise_rms = float(np.sqrt(np.mean(audio[-noise_count:] ** 2)))
    snr_db = 20 * np.log10(max(signal_rms, 1e-15) / max(noise_rms, 1e-15))
    if snr_db < 12 and status == ShotStatus.VALID:
        status = ShotStatus.WARNING
        messages.append("Low estimated S/N.")
    return ShotQuality(
        status=status,
        snr_db=float(snr_db),
        left_peak=left_peak,
        right_peak=right_peak,
        left_rms=left_rms,
        right_rms=right_rms,
        gain_difference_db=float(gain_db),
        lr_correlation=corr,
        delay_difference_samples=delay,
        polarity_inverted=polarity,
        left_clipped=left_clip,
        right_clipped=right_clip,
        selected_channel=selected,
        messages=tuple(messages),
    )


def deconvolve_ess(recording: np.ndarray, reference: np.ndarray, fft_size: int) -> np.ndarray:
    """Return the full circular IR; ``fft_size`` is a minimum, not a crop.

    Keep both late decay and negative-time components at the circular tail.
    Downstream centering and gating rely on that shared FFT time origin.
    """
    if len(recording) == 0 or len(reference) == 0 or int(fft_size) < 1:
        raise ValueError("Deconvolution requires nonempty audio and a positive FFT size")
    required = max(int(fft_size), len(recording) + len(reference) - 1)
    size = 1 << (required - 1).bit_length()
    observed = np.fft.rfft(recording, size)
    source = np.fft.rfft(reference, size)
    regularization = max(float(np.max(np.abs(source) ** 2)) * 1e-10, 1e-20)
    impulse = np.fft.irfft(observed * np.conj(source) / (np.abs(source) ** 2 + regularization), size)
    return impulse


def combine_dual_gain_ir(stereo_ir: np.ndarray, raw_audio: np.ndarray, quality: ShotQuality) -> np.ndarray:
    ir = np.asarray(stereo_ir, dtype=np.float64)
    if quality.status == ShotStatus.CLIPPED:
        return ir[:, 0] if quality.left_peak <= quality.right_peak else ir[:, 1]
    high_index = 0 if quality.left_rms >= quality.right_rms else 1
    low_index = 1 - high_index
    high_rms = quality.left_rms if high_index == 0 else quality.right_rms
    low_rms = quality.right_rms if high_index == 0 else quality.left_rms
    scale = high_rms / max(low_rms, 1e-15)
    high, low = ir[:, high_index], ir[:, low_index] * scale
    low_shift = int(round(quality.delay_difference_samples if high_index == 0 else -quality.delay_difference_samples))
    low = _shift_without_wrap(low, low_shift)
    if quality.polarity_inverted:
        low = -low
    high_peak = float(np.max(np.abs(np.asarray(raw_audio)[:, high_index])))
    high_clipped = quality.left_clipped if high_index == 0 else quality.right_clipped
    # Select one gain path for the whole shot. The low-gain IR is level,
    # delay, and polarity aligned above, so changing paths does not create a
    # roughly 20 dB jump between measurements at different sound pressures.
    return low if high_clipped or high_peak >= 0.90 else high


def _shift_without_wrap(values: np.ndarray, samples: int) -> np.ndarray:
    source = np.asarray(values)
    shifted = np.zeros_like(source)
    if samples > 0:
        shifted[samples:] = source[:-samples]
    elif samples < 0:
        shifted[:samples] = source[-samples:]
    else:
        shifted[:] = source
    return shifted


def select_ir_center(
    impulse: np.ndarray,
    polarity: str = "auto",
    *,
    preferred_polarity: str | None = None,
    search_start_sample: int | None = None,
    search_end_sample: int | None = None,
) -> tuple[float, str, float]:
    """Select a reproducible direct-IR center without implying absolute time."""

    source = np.asarray(impulse, dtype=np.float64)
    if source.size == 0 or not np.isfinite(source).any():
        return 0.0, "unknown", 0.0
    clean = np.where(np.isfinite(source), source, 0.0)
    lower = 0 if search_start_sample is None else int(np.clip(search_start_sample, 0, clean.size - 1))
    upper = clean.size if search_end_sample is None else int(np.clip(search_end_sample, lower + 1, clean.size))
    window = clean[lower:upper]

    def direct_index(metric: np.ndarray) -> int:
        strongest = float(np.max(metric))
        if strongest <= 1e-15:
            return int(np.argmax(metric))
        median = float(np.median(metric))
        mad = float(np.median(np.abs(metric - median)))
        threshold = max(0.20 * strongest, median + 8.0 * max(mad, 1e-15))
        peaks, properties = signal.find_peaks(
            metric,
            height=threshold,
            prominence=max(0.03 * strongest, 1e-15),
        )
        if peaks.size:
            return int(peaks[0])
        return int(np.argmax(metric))

    positive_local = direct_index(np.maximum(window, 0.0))
    negative_local = direct_index(np.maximum(-window, 0.0))
    positive_index = lower + positive_local
    negative_index = lower + negative_local
    positive_peak = max(float(clean[positive_index]), 0.0)
    negative_peak = max(float(-clean[negative_index]), 0.0)
    peak_max = max(positive_peak, negative_peak, 1e-15)
    if polarity == "positive":
        selected_index, selected_polarity, selected_peak = positive_index, "positive", positive_peak
    elif polarity == "negative":
        selected_index, selected_polarity, selected_peak = negative_index, "negative", negative_peak
    else:
        delta_db = abs(20.0 * np.log10(max(positive_peak, 1e-15) / max(negative_peak, 1e-15)))
        if delta_db < 3.0 and preferred_polarity in {"positive", "negative"}:
            if preferred_polarity == "positive":
                selected_index, selected_polarity, selected_peak = positive_index, "positive", positive_peak
            else:
                selected_index, selected_polarity, selected_peak = negative_index, "negative", negative_peak
        elif positive_peak >= negative_peak:
            selected_index, selected_polarity, selected_peak = positive_index, "positive", positive_peak
        else:
            selected_index, selected_polarity, selected_peak = negative_index, "negative", negative_peak
    center = _fractional_sample_at(clean, selected_index, sign=1.0 if selected_polarity == "positive" else -1.0)
    magnitude_window = np.abs(window)
    noise = float(np.median(magnitude_window) + 1.4826 * np.median(np.abs(magnitude_window - np.median(magnitude_window))))
    snr_score = np.clip((20.0 * np.log10(max(selected_peak, 1e-15) / max(noise, 1e-15)) - 6.0) / 24.0, 0.0, 1.0)
    confidence = float(np.clip((selected_peak / peak_max) * snr_score, 0.0, 1.0))
    return center, selected_polarity, confidence


def ir_center_search_end_sample(settings: MeasurementSettings, impulse_size: int) -> int:
    """Limit direct-center search to the causal arrival interval of one ESS period."""

    capture_s = settings.shot_capture_duration_s or settings.shot_period_s
    quiet_tail_s = max(capture_s - settings.sweep_duration_s, 0.0)
    # Correlation has already aligned the recorded ESS onset. Keep enough room
    # for interface latency and microphone relocation, but never enter the next
    # repeat or the circular tail of the deconvolution FFT.
    arrival_s = max(0.005, min(max(quiet_tail_s, 0.020), settings.shot_period_s, 0.250))
    return max(2, min(int(impulse_size), int(np.ceil(arrival_s * settings.sample_rate))))


def _fractional_sample_at(values: np.ndarray, index: int, *, sign: float) -> float:
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    y0, y1, y2 = sign * values[index - 1], sign * values[index], sign * values[index + 1]
    denominator = y0 - 2.0 * y1 + y2
    offset = 0.0 if abs(denominator) < 1e-15 else 0.5 * (y0 - y2) / denominator
    return float(index + np.clip(offset, -0.5, 0.5))


def apply_gate(
    impulse: np.ndarray,
    settings: MeasurementSettings,
    *,
    center_sample: float | None = None,
) -> np.ndarray:
    source = np.asarray(impulse, dtype=np.float64)
    peak = _fractional_peak_sample(source) if center_sample is None else float(center_sample)
    start_offset = int(round(settings.gate_start_ms * settings.sample_rate / 1000))
    end_offset = int(round(settings.gate_end_ms * settings.sample_rate / 1000))
    window_length = end_offset - start_offset
    if window_length <= 2:
        raise ValueError("gate end must be after gate start")
    if window_length >= source.size:
        return readonly(source.copy())
    window = np.zeros(source.size)
    first = int(round(peak)) + start_offset
    indices = (first + np.arange(window_length)) % source.size
    window[indices] = signal.windows.tukey(window_length, alpha=0.15)
    return readonly(source * window)


def response_from_ir(impulse: np.ndarray, settings: MeasurementSettings) -> tuple[np.ndarray, np.ndarray]:
    n_fft = max(settings.fft_size, int(2 ** np.ceil(np.log2(len(impulse)))))
    return np.fft.rfftfreq(n_fft, 1 / settings.sample_rate), np.fft.rfft(impulse, n_fft)


def center_response_on_absolute_peak(
    response: np.ndarray,
    frequency_hz: np.ndarray,
    impulse: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Remove propagation delay by placing the IR absolute peak at 0 ms.

    The stored impulse is intentionally left untouched so its pre-arrival and
    reflection timing remain available for Gate, Impulse, and Wavelet views.
    """
    values = np.asarray(impulse, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.asarray(response, dtype=np.complex128)
    peak_sample = _fractional_peak_sample(values)
    return center_response_on_ir_sample(response, frequency_hz, peak_sample, sample_rate)


def center_response_on_ir_sample(
    response: np.ndarray,
    frequency_hz: np.ndarray,
    center_sample: float,
    sample_rate: int,
) -> np.ndarray:
    """Return a relative-phase response centered on a selected IR sample."""

    peak_sample = float(center_sample)
    delay_s = peak_sample / float(sample_rate)
    phase_advance = np.exp(1j * 2.0 * np.pi * np.asarray(frequency_hz, dtype=float) * delay_s)
    return np.asarray(response, dtype=np.complex128) * phase_advance


def time_align_impulse_to_center(
    impulse: np.ndarray,
    center_sample: float,
    *,
    target_sample: float = 0.0,
) -> np.ndarray:
    """Circularly align an impulse to one shared fractional-sample center."""

    values = np.asarray(impulse, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(center_sample) or not np.isfinite(target_sample):
        return values.copy()
    shift_samples = float(target_sample) - float(center_sample)
    frequency = np.fft.fftfreq(values.size)
    shifted = np.fft.ifft(
        np.fft.fft(values) * np.exp(-2j * np.pi * frequency * shift_samples)
    )
    return np.asarray(shifted.real, dtype=np.float64)


def effective_merge_crossover_hz(
    gate_right_duration_s: float,
    crossover_hz: float = 0.0,
    *,
    ungated: np.ndarray | None = None,
    gated: np.ndarray | None = None,
    frequency: np.ndarray | None = None,
) -> float:
    """Return the manual or response-matched Gate/Full merge crossover.

    Only the right-side Gate duration is relevant to the useful observation
    time after the direct impulse. The negative (pre-arrival) Gate width is
    deliberately excluded.
    """

    if crossover_hz > 0:
        return float(crossover_hz)
    if ungated is None or gated is None or frequency is None:
        return float("nan")

    full = np.asarray(ungated, dtype=np.complex128)
    windowed = np.asarray(gated, dtype=np.complex128)
    hz = np.asarray(frequency, dtype=np.float64)
    if full.shape != windowed.shape or full.shape != hz.shape:
        return float("nan")
    finite = (
        np.isfinite(hz)
        & np.isfinite(full.real)
        & np.isfinite(full.imag)
        & np.isfinite(windowed.real)
        & np.isfinite(windowed.imag)
        & (hz > 0.0)
        & (np.abs(full) > 1e-15)
        & (np.abs(windowed) > 1e-15)
    )
    if np.count_nonzero(finite) < 16:
        return float("nan")
    hz = hz[finite]
    full = full[finite]
    windowed = windowed[finite]

    # The useful low-frequency limit comes only from the post-arrival window.
    # Pre-arrival samples do not add causal observation time.
    search_low = max(20.0, 1.0 / max(float(gate_right_duration_s), 1e-4))
    search_high = min(10_000.0, float(hz[-1]) * 0.90)
    if search_high <= search_low:
        return float("nan")

    octave_span = max(np.log2(search_high / search_low), 1.0)
    grid_count = max(96, int(np.ceil(octave_span * 48.0)) + 1)
    grid = np.geomspace(search_low, search_high, grid_count)
    log_hz = np.log(hz)
    log_grid = np.log(grid)
    gain_delta_db = np.interp(
        log_grid,
        log_hz,
        20.0 * np.log10(np.abs(windowed) / np.abs(full)),
    )
    phase_delta_deg = np.interp(
        log_grid,
        log_hz,
        np.abs(np.rad2deg(np.angle(windowed * np.conj(full)))),
    )
    smoothing_points = max(3, int(round(48.0 / 6.0)))
    kernel = np.ones(smoothing_points, dtype=np.float64) / smoothing_points
    gain_rms = np.sqrt(np.convolve(gain_delta_db * gain_delta_db, kernel, mode="same"))
    phase_rms = np.sqrt(np.convolve(phase_delta_deg * phase_delta_deg, kernel, mode="same"))
    edge = smoothing_points // 2
    candidate_slice = slice(edge, grid_count - edge)
    score = gain_rms + 0.25 * phase_rms / 45.0
    candidate_score = score[candidate_slice]
    if not np.isfinite(candidate_score).any():
        return float("nan")
    best_local = int(np.nanargmin(candidate_score)) + edge
    if gain_rms[best_local] > 3.0 or phase_rms[best_local] > 90.0:
        return float("nan")
    # Prefer the lowest frequency already inside the near-optimal overlap
    # plateau, rather than chasing a numerically perfect point at the top end.
    eligible = np.flatnonzero(
        (score <= score[best_local] + 0.20)
        & (gain_rms <= gain_rms[best_local] + 0.25)
        & (phase_rms <= max(45.0, phase_rms[best_local] + 15.0))
    )
    eligible = eligible[(eligible >= edge) & (eligible < grid_count - edge)]
    selected = int(eligible[0]) if eligible.size else best_local
    return float(grid[selected])


def merge_responses(
    ungated: np.ndarray,
    gated: np.ndarray,
    frequency: np.ndarray,
    gate_right_duration_s: float,
    crossover_hz: float = 0.0,
) -> np.ndarray:
    crossover = effective_merge_crossover_hz(
        gate_right_duration_s,
        crossover_hz,
        ungated=ungated,
        gated=gated,
        frequency=frequency,
    )
    if not np.isfinite(crossover) or crossover <= 0.0:
        # A derived merge must never invent a fixed crossover when Full and
        # Gate do not have a trustworthy overlap. Full remains the safe result.
        return np.asarray(ungated, dtype=np.complex128).copy()
    log_ratio = np.log2(np.maximum(frequency, 1e-9) / crossover)
    # Blend over half an octave (+/- 1/4 octave) around the selected match.
    # A full-octave transition would extend below the verified overlap band.
    high_weight = np.clip((log_ratio + 0.25) / 0.5, 0.0, 1.0)
    high_weight = high_weight * high_weight * (3 - 2 * high_weight)
    return ungated * (1 - high_weight) + gated * high_weight


def analyze_shot(
    raw_stereo_audio: np.ndarray,
    reference: np.ndarray,
    settings: MeasurementSettings,
    *,
    shot_index: int,
    detected_start_sample: int,
    correlation: float,
    timing_error_ms: float = 0.0,
    detected_period_s: float | None = None,
    calculate_wavelet: bool = True,
    preferred_polarity: str | None = None,
    preferred_center_sample: float | None = None,
    marker_detected: bool = False,
    marker_verified: bool = False,
    reference_plane_offset_samples: int = 0,
) -> ShotResult:
    raw = np.asarray(raw_stereo_audio, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[:, None]
    if raw.ndim != 2 or raw.shape[1] != settings.channels:
        raise ValueError(f"shot audio must have {settings.channels} input channel(s)")
    quality = channel_quality(raw, settings)
    channel_ir = np.column_stack(
        [deconvolve_ess(raw[:, channel], reference, settings.fft_size) for channel in range(settings.channels)]
    )
    if settings.channels == 1:
        combined_values = channel_ir[:, 0]
    else:
        combined_values = combine_dual_gain_ir(channel_ir, raw, quality)
    plane_offset = max(0, int(reference_plane_offset_samples))
    if plane_offset:
        # Tweeter-reference capture starts before the marker-defined ESS time
        # plane so an acoustically earlier target is not truncated. Move that
        # known pre-roll to the circular tail before locating the direct IR.
        combined_values = np.roll(combined_values, -plane_offset)
    combined = readonly(combined_values.astype(np.float32))
    post_search_samples = ir_center_search_end_sample(settings, combined.size)
    pre_search_samples = min(
        combined.size // 2,
        max(
            int(round(0.005 * settings.sample_rate)),
            min(
                int(round(settings.search_margin_before_s * settings.sample_rate)),
                int(round(0.020 * settings.sample_rate)),
            ),
        ),
    )
    circular_search_ir = np.concatenate((combined[-pre_search_samples:], combined[:post_search_samples]))
    broad_center, broad_polarity, broad_confidence = select_ir_center(
        circular_search_ir,
        settings.sync_polarity,
        preferred_polarity=preferred_polarity,
        search_start_sample=0,
        search_end_sample=circular_search_ir.size,
    )
    broad_center -= pre_search_samples
    center_sample = broad_center
    center_polarity = broad_polarity
    center_confidence = broad_confidence
    center_tracking_state = "initial"
    search_start = -pre_search_samples
    search_end = post_search_samples
    tracking_target = (
        float(preferred_center_sample)
        if preferred_center_sample is not None and np.isfinite(preferred_center_sample)
        else 0.0
    )
    tracking_radius = max(
        int(round(0.001 * settings.sample_rate)),
        min(int(round(0.005 * settings.sample_rate)), max(1, post_search_samples // 8)),
    )
    local_start_signed = max(-pre_search_samples, int(np.floor(tracking_target)) - tracking_radius)
    local_end_signed = min(post_search_samples, int(np.ceil(tracking_target)) + tracking_radius + 1)
    local_start = local_start_signed + pre_search_samples
    local_end = local_end_signed + pre_search_samples
    if local_end > local_start + 2:
        local_center, local_polarity, local_confidence = select_ir_center(
            circular_search_ir,
            settings.sync_polarity,
            preferred_polarity=preferred_polarity,
            search_start_sample=local_start,
            search_end_sample=local_end,
        )
        local_center -= pre_search_samples
        local_level = abs(float(combined[int(round(local_center)) % combined.size]))
        broad_level = abs(float(combined[int(round(broad_center)) % combined.size]))
        if local_level >= 0.20 * max(broad_level, 1e-15) and local_confidence >= 0.20:
            center_sample, center_polarity, center_confidence = local_center, local_polarity, local_confidence
            center_tracking_state = (
                "tracked" if preferred_center_sample is not None and np.isfinite(preferred_center_sample) else "initial"
            )
            search_start, search_end = local_start_signed, local_end_signed
        else:
            center_tracking_state = "reacquired"
    raw_peak_sample = _fractional_peak_sample(np.asarray(combined))
    gated = readonly(np.asarray(apply_gate(combined, settings, center_sample=center_sample), dtype=np.float32))
    frequency, ungated_response = response_from_ir(combined, settings)
    _, gated_response = response_from_ir(gated, settings)
    timing_ungated_response = np.asarray(ungated_response, dtype=np.complex128).copy()
    timing_gated_response = np.asarray(gated_response, dtype=np.complex128).copy()
    current_gain_db = (
        float(np.mean(settings.input_gain_db_channels))
        if settings.input_gain_db_channels else float(settings.input_gain_db)
    )
    reference_gain_db = (
        current_gain_db
        if settings.response_gain_reference_db is None
        else float(settings.response_gain_reference_db)
    )
    input_gain_normalization_db = reference_gain_db - current_gain_db
    gain_normalization = 10.0 ** (input_gain_normalization_db / 20.0)
    ungated_response = ungated_response * gain_normalization
    gated_response = gated_response * gain_normalization
    timing_ungated_response = timing_ungated_response * gain_normalization
    timing_gated_response = timing_gated_response * gain_normalization
    ungated_response = center_response_on_ir_sample(
        ungated_response, frequency, center_sample, settings.sample_rate
    )
    gated_response = center_response_on_ir_sample(
        gated_response, frequency, center_sample, settings.sample_rate
    )
    if settings.mic_calibration_frequency_hz and settings.mic_calibration_gain_db:
        cal_frequency = np.asarray(settings.mic_calibration_frequency_hz, dtype=float)
        cal_gain = np.asarray(settings.mic_calibration_gain_db, dtype=float)
        correction = np.interp(frequency, cal_frequency, cal_gain, left=cal_gain[0], right=cal_gain[-1])
        if settings.mic_calibration_extrapolation == "no_correction":
            correction[(frequency < cal_frequency[0]) | (frequency > cal_frequency[-1])] = 0.0
        # Calibration files describe the microphone's measured deviation.
        # Remove that deviation from the captured response.
        multiplier = 10 ** (-correction / 20)
        ungated_response = ungated_response * multiplier
        gated_response = gated_response * multiplier
        timing_ungated_response = timing_ungated_response * multiplier
        timing_gated_response = timing_gated_response * multiplier
    if settings.mic_phase_calibration_frequency_hz and settings.mic_phase_calibration_deg:
        phase_frequency = np.asarray(settings.mic_phase_calibration_frequency_hz, dtype=float)
        phase_deg = np.asarray(settings.mic_phase_calibration_deg, dtype=float)
        phase_correction = np.interp(
            frequency,
            phase_frequency,
            phase_deg,
            left=phase_deg[0],
            right=phase_deg[-1],
        )
        phase_multiplier = np.exp(-1j * np.deg2rad(phase_correction))
        ungated_response = ungated_response * phase_multiplier
        gated_response = gated_response * phase_multiplier
        timing_ungated_response = timing_ungated_response * phase_multiplier
        timing_gated_response = timing_gated_response * phase_multiplier
    gate_right_s = max(settings.gate_end_ms, 0.1) / 1000
    effective_crossover = effective_merge_crossover_hz(
        gate_right_s,
        settings.merge_crossover_hz,
        ungated=ungated_response,
        gated=gated_response,
        frequency=frequency,
    )
    merge = merge_responses(
        ungated_response,
        gated_response,
        frequency,
        gate_right_s,
        settings.merge_crossover_hz,
    )
    timing_merge = merge_responses(
        timing_ungated_response,
        timing_gated_response,
        frequency,
        gate_right_s,
        effective_crossover if np.isfinite(effective_crossover) else 0.0,
    )
    wavelet = None
    if calculate_wavelet:
        wavelet = wavelet_from_ir(
            combined,
            settings,
            label=f"Shot {shot_index}",
            center_sample=center_sample,
        )
    end_sample = detected_start_sample + raw.shape[0]
    return ShotResult(
        shot_index=shot_index,
        status=quality.status,
        detected_start_sample=detected_start_sample,
        detected_end_sample=end_sample,
        correlation=float(correlation),
        detection_confidence=float(np.clip((correlation - settings.correlation_threshold) / max(1 - settings.correlation_threshold, 1e-9), 0, 1)),
        timing_error_ms=float(timing_error_ms),
        detected_duration_s=raw.shape[0] / settings.sample_rate,
        detected_period_s=detected_period_s,
        raw_stereo_audio=readonly(raw.astype(np.float32)),
        combined_raw_ir=combined,
        gated_ir=gated,
        frequency_hz=readonly(frequency.astype(np.float32)),
        ungated_complex_response=readonly(ungated_response.astype(np.complex64)),
        gated_complex_response=readonly(gated_response.astype(np.complex64)),
        merged_complex_response=readonly(merge.astype(np.complex64)),
        wavelet_map=wavelet,
        quality=quality,
        settings_snapshot=settings,
        included_in_average=quality.status in {ShotStatus.VALID, ShotStatus.WARNING},
        sweep_index=shot_index,
        reference_mode=settings.reference_mode,
        reference_plane="peak_centered",
        absolute_timing_valid=bool(settings.reference_mode == "timing_marker" and marker_verified),
        marker_detected=bool(marker_detected),
        marker_verified=bool(marker_verified),
        center_method=f"{settings.sync_polarity}_polarity_peak",
        center_polarity=center_polarity,
        center_sample=float(center_sample),
        center_confidence=float(center_confidence),
        center_tracking_state=center_tracking_state,
        center_search_start_sample=int(search_start),
        center_search_end_sample=int(search_end),
        input_gain_normalization_db=float(input_gain_normalization_db),
        raw_peak_sample=float(raw_peak_sample),
        gain_valid=True,
        relative_phase_valid=True,
        timing_valid=bool(settings.reference_mode == "timing_marker" and marker_verified),
        quality_tier="basic",
        timing_ungated_complex_response=readonly(timing_ungated_response.astype(np.complex64)),
        timing_gated_complex_response=readonly(timing_gated_response.astype(np.complex64)),
        timing_merged_complex_response=readonly(timing_merge.astype(np.complex64)),
        merge_valid=bool(np.isfinite(effective_crossover)),
        effective_merge_crossover_hz=float(effective_crossover),
    )


def wavelet_from_ir(
    impulse: np.ndarray,
    settings: MeasurementSettings,
    *,
    label: str,
    center_sample: float | None = None,
) -> object:
    values = np.asarray(impulse, dtype=np.float64)
    center = (
        _fractional_peak_sample(values)
        if center_sample is None or not np.isfinite(center_sample)
        else float(center_sample)
    )
    centered = time_align_impulse_to_center(
        values,
        center,
        target_sample=(len(values) - 1) / 2.0,
    )
    wavelet_settings = settings_from_preset(
        "Overview",
        WaveletSettings(
            f_max=min(20_000.0, settings.sample_rate / 2 - 1),
            frequency_bins=96,
            time_bins=320,
        ),
    )
    return replace(
        complex_morlet_scalogram(
            centered,
            settings.sample_rate,
            wavelet_settings,
            label=label,
        ),
        alignment_center_sample=float(center),
    )


def reanalyze_gate(shot: ShotResult, settings: MeasurementSettings, *, force_recenter: bool = False) -> ShotResult:
    if shot.combined_raw_ir is None:
        return shot
    center_sample = shot.center_sample
    center_polarity = shot.center_polarity
    center_confidence = shot.center_confidence
    center_tracking_state = shot.center_tracking_state
    search_start = shot.center_search_start_sample
    search_end = shot.center_search_end_sample
    post_search_samples = ir_center_search_end_sample(settings, len(shot.combined_raw_ir))
    pre_search_samples = min(
        len(shot.combined_raw_ir) // 2,
        max(
            int(round(0.005 * settings.sample_rate)),
            min(int(round(settings.search_margin_before_s * settings.sample_rate)), int(round(0.020 * settings.sample_rate))),
        ),
    )
    legacy_or_outside_window = (
        bool(force_recenter)
        or shot.center_polarity not in {"positive", "negative", "auto", ""}
        or (
            settings.sync_polarity in {"positive", "negative"}
            and shot.center_polarity != settings.sync_polarity
        )
        or not np.isfinite(center_sample)
        or center_sample < -pre_search_samples
        or center_sample >= post_search_samples
        or (search_start == 0 and search_end == 0)
    )
    if legacy_or_outside_window:
        circular_search_ir = np.concatenate((
            np.asarray(shot.combined_raw_ir)[-pre_search_samples:],
            np.asarray(shot.combined_raw_ir)[:post_search_samples],
        ))
        center_sample, center_polarity, center_confidence = select_ir_center(
            circular_search_ir,
            settings.sync_polarity,
            preferred_polarity=(shot.center_polarity if shot.center_polarity in {"positive", "negative"} else None),
        )
        center_sample -= pre_search_samples
        center_tracking_state = "reacquired"
        search_start, search_end = -pre_search_samples, post_search_samples
    gated = readonly(np.asarray(apply_gate(shot.combined_raw_ir, settings, center_sample=center_sample), dtype=np.float32))
    frequency, ungated = response_from_ir(shot.combined_raw_ir, settings)
    _, gated_response = response_from_ir(gated, settings)
    timing_ungated = np.asarray(ungated, dtype=np.complex128).copy()
    timing_gated = np.asarray(gated_response, dtype=np.complex128).copy()
    gain_normalization = 10.0 ** (float(shot.input_gain_normalization_db) / 20.0)
    ungated = ungated * gain_normalization
    gated_response = gated_response * gain_normalization
    timing_ungated = timing_ungated * gain_normalization
    timing_gated = timing_gated * gain_normalization
    ungated = center_response_on_ir_sample(ungated, frequency, center_sample, settings.sample_rate)
    gated_response = center_response_on_ir_sample(gated_response, frequency, center_sample, settings.sample_rate)
    if settings.mic_calibration_frequency_hz and settings.mic_calibration_gain_db:
        cal_frequency = np.asarray(settings.mic_calibration_frequency_hz, dtype=float)
        cal_gain = np.asarray(settings.mic_calibration_gain_db, dtype=float)
        correction = np.interp(frequency, cal_frequency, cal_gain, left=cal_gain[0], right=cal_gain[-1])
        if settings.mic_calibration_extrapolation == "no_correction":
            correction[(frequency < cal_frequency[0]) | (frequency > cal_frequency[-1])] = 0.0
        multiplier = 10 ** (-correction / 20)
        ungated = ungated * multiplier
        gated_response = gated_response * multiplier
        timing_ungated = timing_ungated * multiplier
        timing_gated = timing_gated * multiplier
    if settings.mic_phase_calibration_frequency_hz and settings.mic_phase_calibration_deg:
        phase_frequency = np.asarray(settings.mic_phase_calibration_frequency_hz, dtype=float)
        phase_deg = np.asarray(settings.mic_phase_calibration_deg, dtype=float)
        phase_correction = np.interp(
            frequency,
            phase_frequency,
            phase_deg,
            left=phase_deg[0],
            right=phase_deg[-1],
        )
        phase_multiplier = np.exp(-1j * np.deg2rad(phase_correction))
        ungated = ungated * phase_multiplier
        gated_response = gated_response * phase_multiplier
        timing_ungated = timing_ungated * phase_multiplier
        timing_gated = timing_gated * phase_multiplier
    gate_right_s = max(settings.gate_end_ms, 0.1) / 1000
    effective_crossover = effective_merge_crossover_hz(
        gate_right_s,
        settings.merge_crossover_hz,
        ungated=ungated,
        gated=gated_response,
        frequency=frequency,
    )
    merged = merge_responses(
        ungated,
        gated_response,
        frequency,
        gate_right_s,
        settings.merge_crossover_hz,
    )
    timing_merged = merge_responses(
        timing_ungated,
        timing_gated,
        frequency,
        gate_right_s,
        effective_crossover if np.isfinite(effective_crossover) else 0.0,
    )
    wavelet = (
        wavelet_from_ir(
            shot.combined_raw_ir,
            settings,
            label=f"Shot {shot.shot_index}",
            center_sample=center_sample,
        )
        if shot.wavelet_map is not None else None
    )
    return replace(
        shot,
        gated_ir=gated,
        frequency_hz=readonly(frequency.astype(np.float32)),
        ungated_complex_response=readonly(ungated.astype(np.complex64)),
        gated_complex_response=readonly(gated_response.astype(np.complex64)),
        merged_complex_response=readonly(merged.astype(np.complex64)),
        timing_ungated_complex_response=readonly(timing_ungated.astype(np.complex64)),
        timing_gated_complex_response=readonly(timing_gated.astype(np.complex64)),
        timing_merged_complex_response=readonly(timing_merged.astype(np.complex64)),
        wavelet_map=wavelet,
        center_sample=float(center_sample),
        center_polarity=center_polarity,
        center_confidence=float(center_confidence),
        center_tracking_state=center_tracking_state,
        center_search_start_sample=int(search_start),
        center_search_end_sample=int(search_end),
        merge_valid=bool(np.isfinite(effective_crossover)),
        effective_merge_crossover_hz=float(effective_crossover),
        settings_snapshot=settings,
    )


def average_responses(shots: list[ShotResult] | tuple[ShotResult, ...], response: str = "merged") -> tuple[np.ndarray, np.ndarray] | None:
    attribute = {"merged": "merged_complex_response", "ungated": "ungated_complex_response", "gated": "gated_complex_response"}[response]
    usable = [shot for shot in shots if shot.included_in_average and getattr(shot, attribute) is not None]
    if not usable:
        return None
    frequency = np.asarray(usable[0].frequency_hz)
    values = [np.asarray(getattr(shot, attribute)) for shot in usable if np.array_equal(shot.frequency_hz, frequency)]
    if not values:
        return None
    return readonly(frequency), readonly(np.mean(np.stack(values), axis=0))


def high_precision_result(
    shots: list[ShotResult] | tuple[ShotResult, ...],
) -> HighPrecisionResult | None:
    """Combine repeat measurements without replacing or modifying any source shot.

    Each response is first aligned to the best-quality shot with sub-sample peak
    timing. A quality-weighted complex estimate is then refined with per-bin
    Huber weights. This rejects isolated disturbances while retaining the
    measured complex phase instead of smoothing magnitude or phase curves.
    """

    usable = [
        shot for shot in shots
        if shot.included_in_average
        and shot.relative_phase_valid
        and shot.frequency_hz is not None
        and shot.ungated_complex_response is not None
        and shot.gated_complex_response is not None
        and shot.merged_complex_response is not None
        and shot.combined_raw_ir is not None
        and shot.status in {ShotStatus.VALID, ShotStatus.WARNING}
    ]
    if not usable:
        return None
    frequency = np.asarray(usable[0].frequency_hz, dtype=np.float64)
    usable = [shot for shot in usable if np.array_equal(np.asarray(shot.frequency_hz), frequency)]
    if not usable:
        return None

    scalar_weights = np.asarray([_shot_reliability(shot) for shot in usable], dtype=np.float64)
    reference_position = int(np.argmax(scalar_weights))
    reference = usable[reference_position]
    # analyze_shot() has already placed every absolute IR peak at zero, down
    # to a fractional sample.  Applying the full raw-IR peak displacement here
    # would add that propagation delay back into a multi-shot result.  Keep
    # only the residual fractional-center difference for legacy/restored shots.
    if all(np.isfinite(shot.center_sample) for shot in usable):
        # New shots are already centered at the selected fractional sample.
        offsets = np.zeros(len(usable), dtype=np.float64)
    else:
        reference_ir = np.asarray(reference.combined_raw_ir)
        reference_integer_peak = int(np.argmax(np.abs(reference_ir)))
        reference_fraction = _fractional_peak_sample(reference_ir) - reference_integer_peak
        offsets = np.asarray([
            (_fractional_peak_sample(np.asarray(shot.combined_raw_ir)) - int(np.argmax(np.abs(shot.combined_raw_ir))))
            - reference_fraction
            for shot in usable
        ], dtype=np.float64)
    phase_alignment = np.exp(2j * np.pi * offsets[:, None] * frequency[None, :] / reference.settings_snapshot.sample_rate)

    combined: dict[str, np.ndarray] = {}
    single_shot = len(usable) == 1
    coherence = np.full(frequency.size, np.nan if single_shot else 1.0, dtype=np.float64)
    repeatability = np.full(frequency.size, np.nan if single_shot else 0.0, dtype=np.float64)
    confidence = np.full(frequency.size, _shot_reliability(usable[0]) if single_shot else 1.0, dtype=np.float64)
    for attribute in ("ungated_complex_response", "gated_complex_response", "merged_complex_response"):
        responses = np.stack([np.asarray(getattr(shot, attribute), dtype=np.complex128) for shot in usable])
        aligned = responses * phase_alignment
        estimate, bin_weights = _robust_complex_average(aligned, scalar_weights)
        combined[attribute] = estimate
        if attribute == "merged_complex_response" and not single_shot:
            weight_sum = np.maximum(np.sum(bin_weights, axis=0), 1e-15)
            unit = np.divide(aligned, np.maximum(np.abs(aligned), 1e-15))
            coherence = np.abs(np.sum(bin_weights * unit, axis=0)) / weight_sum
            gain_db = 20.0 * np.log10(np.maximum(np.abs(aligned), 1e-15))
            center_db = np.sum(bin_weights * gain_db, axis=0) / weight_sum
            repeatability = np.sqrt(np.sum(bin_weights * (gain_db - center_db) ** 2, axis=0) / weight_sum)
            repeatability_score = np.exp(-((repeatability / 1.5) ** 2))
            confidence = np.clip(coherence * repeatability_score, 0.0, 1.0)

    # Integrate the actual pre-calibration IRs. All derived responses must be
    # reproducible from this one original, including after a calibration change.
    raw_responses = []
    for shot in usable:
        raw_frequency, raw_response = response_from_ir(shot.combined_raw_ir, shot.settings_snapshot)
        center = shot.center_sample if np.isfinite(shot.center_sample) else _fractional_peak_sample(shot.combined_raw_ir)
        raw_responses.append(center_response_on_ir_sample(
            raw_response * 10 ** (shot.input_gain_normalization_db / 20),
            raw_frequency, center, shot.settings_snapshot.sample_rate,
        ))
    raw_estimate, _ = _robust_complex_average(np.stack(raw_responses) * phase_alignment, scalar_weights)
    original_ir = readonly(np.fft.irfft(raw_estimate).astype(np.float32))
    integrated = reanalyze_gate(replace(
        reference, combined_raw_ir=original_ir, raw_stereo_audio=None,
        center_sample=0.0, center_search_start_sample=-1, center_search_end_sample=len(original_ir),
        input_gain_normalization_db=0.0, wavelet_map=None,
    ), reference.settings_snapshot)
    for attribute in combined:
        combined[attribute] = getattr(integrated, attribute)
    arrivals = [shot.center_sample for shot in usable if shot.timing_valid and np.isfinite(shot.center_sample)]

    valid_band = (
        (frequency >= max(10.0, reference.settings_snapshot.start_frequency_hz))
        & (frequency <= reference.settings_snapshot.end_frequency_hz)
    )
    coherence = np.where(valid_band, coherence, np.nan)
    repeatability = np.where(valid_band, repeatability, np.nan)
    confidence = np.where(valid_band, confidence, 0.0)

    return HighPrecisionResult(
        frequency_hz=readonly(frequency.astype(np.float32)),
        ungated_complex_response=readonly(combined["ungated_complex_response"].astype(np.complex64)),
        gated_complex_response=readonly(combined["gated_complex_response"].astype(np.complex64)),
        merged_complex_response=readonly(combined["merged_complex_response"].astype(np.complex64)),
        used_shot_indices=tuple(shot.shot_index for shot in usable),
        reference_shot_index=reference.shot_index,
        alignment_samples=tuple(float(value) for value in offsets),
        integrated_uncalibrated_ir=original_ir,
        arrival_sample=float(np.median(arrivals)) if arrivals else float("nan"),
        observation_samples=min(max(0, shot.detected_end_sample - shot.detected_start_sample) for shot in usable),
        minimum_center_confidence=min(shot.center_confidence for shot in usable),
        coherence=readonly(coherence.astype(np.float32)),
        repeatability_db=readonly(repeatability.astype(np.float32)),
        confidence=readonly(confidence.astype(np.float32)),
        reference_mode=reference.settings_snapshot.reference_mode,
        timing_valid_shot_count=sum(1 for shot in usable if shot.timing_valid),
        absolute_timing_valid=bool(
            reference.settings_snapshot.reference_mode == "timing_marker"
            and sum(1 for shot in usable if shot.timing_valid) == len(usable)
        ),
    )


def build_standard_result(
    shots: list[ShotResult] | tuple[ShotResult, ...],
    reference: np.ndarray,
    settings: MeasurementSettings,
    *,
    clock_mode: str = "auto",
) -> HighPrecisionResult | None:
    """Build the immutable post-stop Standard result from retained Raw shots."""

    allowed_statuses = {ShotStatus.VALID}
    if settings.include_warnings_in_average:
        allowed_statuses.add(ShotStatus.WARNING)
    raw_shots = [
        shot for shot in shots
        if shot.included_in_average
        and shot.status in allowed_statuses
        and shot.raw_stereo_audio is not None
    ]
    if not raw_shots:
        return high_precision_result(shots)
    timing_mode = settings.reference_mode == "timing_marker"
    timing_raw_shots = [shot for shot in raw_shots if shot.timing_valid]
    drift_ppm, clock_ratio, residual = estimate_clock_drift(
        timing_raw_shots if settings.reference_mode == "timing_marker" else raw_shots,
        settings,
    )
    timing_segment_counts: dict[int, int] = {}
    for shot in (timing_raw_shots if timing_mode else raw_shots):
        timing_segment_counts[shot.timing_segment] = timing_segment_counts.get(shot.timing_segment, 0) + 1
    clock_segment_shots = max(timing_segment_counts.values(), default=0)
    sweep_error_samples = abs(clock_ratio - 1.0) * len(reference)
    correction_valid = (
        timing_mode
        and
        clock_segment_shots >= 3
        and np.isfinite(residual)
        and residual <= 1.0
        and abs(drift_ppm) <= 5_000.0
    )
    apply_clock = (timing_mode and clock_mode == "force") or (
        clock_mode == "auto" and correction_valid and sweep_error_samples >= 0.1
    )
    corrected: list[ShotResult] = []
    for shot in raw_shots:
        raw = np.asarray(shot.raw_stereo_audio, dtype=np.float64)
        if apply_clock and shot.timing_valid:
            target_count = max(2, int(round(raw.shape[0] / clock_ratio)))
            resampled = signal.resample(raw, target_count, axis=0)
            if target_count < raw.shape[0]:
                resampled = np.pad(resampled, ((0, raw.shape[0] - target_count), (0, 0)))
            raw = resampled[: raw.shape[0]]
        shot_gain_settings = replace(
            settings,
            input_gain_db=shot.settings_snapshot.input_gain_db,
            input_gain_db_channels=shot.settings_snapshot.input_gain_db_channels,
            input_gain_max_db_channels=shot.settings_snapshot.input_gain_max_db_channels,
            input_analog_gain_db=shot.settings_snapshot.input_analog_gain_db,
            minidsp_gain_adjustment_db=shot.settings_snapshot.minidsp_gain_adjustment_db,
            response_gain_reference_db=shot.settings_snapshot.response_gain_reference_db,
        )
        rebuilt = analyze_shot(
            raw,
            reference,
            shot_gain_settings,
            shot_index=shot.shot_index,
            detected_start_sample=shot.detected_start_sample,
            correlation=shot.correlation,
            timing_error_ms=shot.timing_error_ms,
            detected_period_s=shot.detected_period_s,
            calculate_wavelet=False,
            preferred_polarity=shot.center_polarity,
            preferred_center_sample=shot.center_sample,
            marker_detected=shot.marker_detected,
            marker_verified=shot.marker_verified,
            reference_plane_offset_samples=(
                int(round(settings.timing_reference_capture_preroll_s * settings.sample_rate))
                if settings.timing_reference_kind == "phaseeq_tweeter_reference" else 0
            ),
        )
        corrected.append(replace(
            rebuilt,
            included_in_average=shot.included_in_average,
            sweep_index=shot.sweep_index,
            timing_segment=shot.timing_segment,
            timing_index=shot.timing_index,
            loop_boundary_before=shot.loop_boundary_before,
        ))
    result_candidates = [shot for shot in corrected if shot.timing_valid] if apply_clock else corrected
    if timing_mode:
        corrected_segment_counts: dict[int, int] = {}
        for shot in result_candidates:
            corrected_segment_counts[shot.timing_segment] = corrected_segment_counts.get(shot.timing_segment, 0) + 1
        selected_segment = max(
            corrected_segment_counts,
            key=lambda segment: (corrected_segment_counts[segment], -segment),
        )
        selected = _best_stable_window([
            shot for shot in result_candidates if shot.timing_segment == selected_segment
        ])
    else:
        # External playback may restart or vary its loop gap. Peak-centered
        # shots remain comparable, so timing segments are diagnostic only.
        selected = _best_stable_window(result_candidates)
    result = high_precision_result(selected)
    if result is None:
        return None
    return replace(
        result,
        clock_drift_ppm=float(drift_ppm),
        clock_residual_samples=float(residual),
        clock_corrected=bool(apply_clock),
        algorithm=(
            "timing_marker_clock_corrected_robust_complex_average"
            if timing_mode
            else "ir_peak_centered_robust_complex_average"
        ),
    )


def estimate_clock_drift(
    shots: list[ShotResult] | tuple[ShotResult, ...],
    settings: MeasurementSettings,
) -> tuple[float, float, float]:
    usable = [shot for shot in shots if shot.detected_end_sample > shot.detected_start_sample]
    if len(usable) < 2:
        return 0.0, 1.0, float("nan")
    segments: dict[int, list[ShotResult]] = {}
    for shot in usable:
        segments.setdefault(int(shot.timing_segment), []).append(shot)
    usable = max(
        segments.values(),
        key=lambda segment: (len(segment), -int(segment[0].timing_segment)),
    )
    if len(usable) < 2:
        return 0.0, 1.0, float("nan")
    if all(shot.timing_index >= 0 for shot in usable):
        origin = usable[0].timing_index
        x = np.asarray([shot.timing_index - origin for shot in usable], dtype=np.float64)
    else:
        x = np.asarray([shot.shot_index - usable[0].shot_index for shot in usable], dtype=np.float64)
    y = np.asarray([shot.detected_start_sample for shot in usable], dtype=np.float64)
    weights = np.ones_like(x)
    slope = settings.shot_period_s * settings.sample_rate
    intercept = y[0]
    for _ in range(4):
        matrix = np.column_stack((np.ones_like(x), x))
        weighted = matrix * np.sqrt(weights)[:, None]
        target = y * np.sqrt(weights)
        intercept, slope = np.linalg.lstsq(weighted, target, rcond=None)[0]
        residuals = y - (intercept + slope * x)
        scale = max(1.4826 * float(np.median(np.abs(residuals))), 0.25)
        normalized = np.abs(residuals) / (1.5 * scale)
        weights = np.ones_like(normalized)
        np.divide(1.0, normalized, out=weights, where=normalized > 1.0)
    residual_rms = float(np.sqrt(np.mean((y - (intercept + slope * x)) ** 2)))
    expected = settings.shot_period_s * settings.sample_rate
    ratio = float(slope / expected)
    return (ratio - 1.0) * 1_000_000.0, ratio, residual_rms


def conservative_denoised_result(
    standard: HighPrecisionResult,
    *,
    max_attenuation_db: float = 3.0,
) -> HighPrecisionResult:
    """Create an optional phase-fixed preview; never modifies Standard."""

    maximum = max(0.0, float(max_attenuation_db))
    confidence = np.nan_to_num(np.asarray(standard.confidence, dtype=np.float64), nan=0.0)
    attenuation_db = -maximum * np.clip(1.0 - confidence, 0.0, 1.0)
    multiplier = 10 ** (attenuation_db / 20.0)
    return replace(
        standard,
        ungated_complex_response=readonly((standard.ungated_complex_response * multiplier).astype(np.complex64)),
        gated_complex_response=readonly((standard.gated_complex_response * multiplier).astype(np.complex64)),
        merged_complex_response=readonly((standard.merged_complex_response * multiplier).astype(np.complex64)),
        algorithm=f"phase_fixed_confidence_attenuation_{maximum:g}db",
    )


def _best_stable_window(shots: list[ShotResult]) -> list[ShotResult]:
    if len(shots) <= 8:
        return shots
    best = shots[:8]
    best_score = -float("inf")
    for window_size in range(5, 9):
        for start in range(0, len(shots) - window_size + 1):
            candidate = shots[start : start + window_size]
            result = high_precision_result(candidate)
            if result is None:
                continue
            coherence = result.median_coherence
            repeatability = result.repeatability_p90_db
            score = coherence - 0.05 * min(repeatability, 20.0)
            if score > best_score:
                best, best_score = candidate, score
    return best


def _shot_reliability(shot: ShotResult) -> float:
    snr_score = float(np.clip((shot.quality.snr_db - 6.0) / 34.0, 0.05, 1.0))
    correlation_score = float(np.clip(shot.correlation, 0.05, 1.0)) ** 2
    warning_penalty = 0.65 if shot.status == ShotStatus.WARNING else 1.0
    return max(1e-3, snr_score * correlation_score * warning_penalty)


def _fractional_peak_sample(impulse: np.ndarray) -> float:
    magnitude = np.abs(np.asarray(impulse, dtype=np.float64))
    index = int(np.argmax(magnitude))
    if index <= 0 or index >= magnitude.size - 1:
        return float(index)
    left, center, right = magnitude[index - 1:index + 2]
    denominator = left - 2.0 * center + right
    fraction = 0.0 if abs(denominator) < 1e-15 else 0.5 * (left - right) / denominator
    return float(index + np.clip(fraction, -0.5, 0.5))


def _direct_window_alignment_offset(source: np.ndarray, reference: np.ndarray, sample_rate: int) -> float:
    source_values = np.asarray(source, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    source_peak = int(np.argmax(np.abs(source_values)))
    reference_peak = int(np.argmax(np.abs(reference_values)))
    before = max(2, int(round(0.001 * sample_rate)))
    after = max(4, int(round(0.003 * sample_rate)))
    source_lo, source_hi = max(0, source_peak - before), min(source_values.size, source_peak + after)
    reference_lo, reference_hi = max(0, reference_peak - before), min(reference_values.size, reference_peak + after)
    length = min(source_hi - source_lo, reference_hi - reference_lo)
    if length < 5:
        return _fractional_peak_sample(source_values) - _fractional_peak_sample(reference_values)
    source_window = source_values[source_lo : source_lo + length]
    reference_window = reference_values[reference_lo : reference_lo + length]
    correlation = signal.correlate(source_window, reference_window, mode="full", method="fft")
    magnitude = np.abs(correlation)
    index = int(np.argmax(magnitude))
    lag = index - (length - 1)
    fraction = 0.0
    if 0 < index < magnitude.size - 1:
        left, center, right = magnitude[index - 1 : index + 2]
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-15:
            fraction = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    return float((source_peak - reference_peak) + lag + fraction)


def _robust_complex_average(values: np.ndarray, scalar_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = np.broadcast_to(np.asarray(scalar_weights, dtype=np.float64)[:, None], values.shape).copy()
    weight_sum = np.maximum(np.sum(base, axis=0), 1e-15)
    estimate = np.sum(base * values, axis=0) / weight_sum
    if values.shape[0] < 3:
        return estimate, base
    # This median is only the robust starting location. The published result
    # is recomputed from the original complex responses with continuous
    # weights, so no median-filtered frequency curve replaces measured detail.
    estimate = np.median(values.real, axis=0) + 1j * np.median(values.imag, axis=0)
    for _ in range(2):
        residual = np.abs(values - estimate[None, :])
        scale = 1.4826 * np.median(residual, axis=0)
        scale = np.maximum(scale, np.maximum(np.abs(estimate) * 1e-5, 1e-12))
        normalized = residual / (1.5 * scale[None, :])
        huber = np.ones_like(normalized)
        np.divide(1.0, normalized, out=huber, where=normalized > 1.0)
        weights = base * huber
        weight_sum = np.maximum(np.sum(weights, axis=0), 1e-15)
        estimate = np.sum(weights * values, axis=0) / weight_sum
    return estimate, weights
