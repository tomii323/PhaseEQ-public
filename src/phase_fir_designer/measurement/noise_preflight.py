from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .levels import level_calibration_offset_db
from .models import MeasurementSettings, ShotQuality


@dataclass(frozen=True)
class AmbientNoiseProfile:
    duration_s: float
    channel_rms: tuple[float, ...]
    channel_p90_rms: tuple[float, ...]
    channel_peak: tuple[float, ...]
    frequency_hz: np.ndarray
    channel_band_rms: np.ndarray

    def shifted_by_gain(self, gain_delta_db: float) -> "AmbientNoiseProfile":
        scale = 10.0 ** (float(gain_delta_db) / 20.0)
        return AmbientNoiseProfile(
            duration_s=self.duration_s,
            channel_rms=tuple(value * scale for value in self.channel_rms),
            channel_p90_rms=tuple(value * scale for value in self.channel_p90_rms),
            channel_peak=tuple(value * scale for value in self.channel_peak),
            frequency_hz=self.frequency_hz,
            channel_band_rms=self.channel_band_rms * scale,
        )


@dataclass(frozen=True)
class PilotAssessment:
    ready: bool
    action: str
    message: str
    selected_channel_index: int
    peak_dbfs: float
    headroom_db: float
    snr_median_db: float
    snr_p10_db: float
    passing_band_fraction: float
    speaker_adjustment_db: float
    microphone_gain_adjustment_db: float
    active_band_start_hz: float = 0.0
    active_band_end_hz: float = 0.0
    active_band_count: int = 0
    active_band_span_octaves: float = 0.0
    level_mask_valid: bool = False
    playback_spl_db: float | None = None
    snr_target_db: float = 0.0
    snr_minimum_db: float = 0.0
    snr_passing_fraction_required: float = 0.0


@dataclass(frozen=True)
class PilotSnrThresholds:
    median_db: float
    p10_db: float
    passing_fraction: float


def pilot_snr_thresholds(settings: MeasurementSettings) -> PilotSnrThresholds:
    """Return preparation thresholds appropriate to the selected quality gate."""

    if settings.quality_gate_mode == "strict":
        return PilotSnrThresholds(
            float(settings.target_snr_db),
            float(settings.minimum_snr_db),
            float(settings.minimum_passing_band_fraction),
        )
    if settings.quality_gate_mode == "lenient":
        return PilotSnrThresholds(6.0, 0.0, 0.01)
    return PilotSnrThresholds(15.0, 6.0, 0.25)


@dataclass(frozen=True)
class PreflightLevelMask:
    """Speaker-independent band selection used only for Pilot level setup."""

    mask: np.ndarray
    start_hz: float
    end_hz: float
    band_count: int
    span_octaves: float
    valid: bool


def measure_ambient_noise(audio: np.ndarray, settings: MeasurementSettings) -> AmbientNoiseProfile:
    values = _channels(audio)
    frame = max(64, int(round(0.050 * settings.sample_rate)))
    framed = _frame_rms(values, frame)
    frequencies, band_rms = _fractional_octave_rms(values, settings.sample_rate)
    return AmbientNoiseProfile(
        duration_s=values.shape[0] / settings.sample_rate,
        channel_rms=tuple(np.sqrt(np.mean(values * values, axis=0))),
        channel_p90_rms=tuple(np.percentile(framed, 90, axis=0)),
        channel_peak=tuple(np.max(np.abs(values), axis=0)),
        frequency_hz=frequencies,
        channel_band_rms=band_rms,
    )


def assess_pilot(
    audio: np.ndarray,
    quality: ShotQuality,
    noise: AmbientNoiseProfile,
    settings: MeasurementSettings,
) -> PilotAssessment:
    values = _channels(audio)
    channel = _selected_channel_index(quality.selected_channel, values)
    peak = float(np.max(np.abs(values[:, channel])))
    peak_dbfs = 20.0 * np.log10(max(peak, 1e-15))
    headroom = -peak_dbfs
    sweep_count = min(values.shape[0], int(round(settings.sweep_duration_s * settings.sample_rate)))
    frequencies, signal_bands = _fractional_octave_rms(values[:sweep_count], settings.sample_rate)
    noise_bands = noise.channel_band_rms[:, min(channel, noise.channel_band_rms.shape[1] - 1)]
    signal_band = signal_bands[:, channel]
    count = min(frequencies.size, noise.frequency_hz.size, signal_band.size, noise_bands.size)
    frequencies = frequencies[:count]
    signal_band = signal_band[:count]
    noise_bands = noise_bands[:count]
    signal_only = np.sqrt(np.maximum(signal_band * signal_band - noise_bands * noise_bands, 1e-30))
    snr = 20.0 * np.log10(signal_only / np.maximum(noise_bands, 1e-15))
    measurement_range = (
        (frequencies >= max(20.0, settings.start_frequency_hz))
        & (frequencies <= min(settings.end_frequency_hz, settings.sample_rate / 2 * 0.98))
    )
    signal_db = 20.0 * np.log10(np.maximum(signal_only, 1e-15))
    level_mask = _preflight_level_mask(frequencies, signal_db, measurement_range, settings)
    thresholds = pilot_snr_thresholds(settings)
    usable = snr[level_mask.mask]
    if usable.size:
        median_snr = float(np.median(usable))
        p10_snr = float(np.percentile(usable, 10))
        passing = float(np.mean(usable >= thresholds.p10_db))
    else:
        median_snr, p10_snr, passing = -300.0, -300.0, 0.0

    snr_ready = (
        level_mask.valid
        and median_snr >= thresholds.median_db
        and passing >= thresholds.passing_fraction
        and p10_snr >= thresholds.p10_db
    )
    playback_spl = _pilot_playback_spl(values[:sweep_count, channel], noise, channel, settings)
    gain_delta = float(np.clip(headroom - settings.target_headroom_db, -24.0, 24.0))
    if quality.left_clipped and quality.right_clipped:
        return _assessment(False, "reduce_input", "全入力チャンネルがクリップしました。入力Gainを下げ、次のPilotで再確認します。", channel, peak_dbfs, headroom, median_snr, p10_snr, passing, 0.0, min(gain_delta, -3.0), level_mask, playback_spl, thresholds)
    if not level_mask.valid:
        return _assessment(
            False,
            "invalid_level_band",
            (
                "音量調整用の有効帯域を十分な幅で検出できません。"
                "ESSの再生、接続、スピーカー音量を確認してPilotを再測定してください。"
            ),
            channel,
            peak_dbfs,
            headroom,
            median_snr,
            p10_snr,
            passing,
            0.0,
            0.0,
            level_mask, playback_spl, thresholds,
        )
    if headroom < settings.minimum_headroom_db:
        return _assessment(False, "reduce_input", "入力ヘッドルームが6 dB未満です。マイクGainを下げ、次のPilotで再確認します。", channel, peak_dbfs, headroom, median_snr, p10_snr, passing, 0.0, min(gain_delta, -1.0), level_mask, playback_spl, thresholds)
    if not snr_ready:
        if playback_spl is not None and playback_spl >= settings.playback_target_spl_db:
            return _assessment(False, "level_limit_reached", f"再生レベルは{playback_spl:.1f} dB SPLです。75 dB以上では音量を上げず、低S/Nのまま続行できるか判定します。", channel, peak_dbfs, headroom, median_snr, p10_snr, passing, 0.0, 0.0, level_mask, playback_spl, thresholds)
        needed = _required_speaker_adjustment_db(median_snr, p10_snr, settings)
        return _assessment(False, "raise_speaker", f"音量調整用有効帯域のS/Nが不足しています。スピーカー音量を最大{needed:.1f} dBだけ上げて次のPilotを再生してください。", channel, peak_dbfs, headroom, median_snr, p10_snr, passing, needed, 0.0, level_mask, playback_spl, thresholds)
    is_umik1 = settings.microphone_profile_id.strip().lower() == "umik-1"
    if abs(gain_delta) >= 1.0 and not is_umik1:
        direction = "上げ" if gain_delta > 0 else "下げ"
        return _assessment(False, "adjust_input", f"S/Nは合格です。マイクGainを{abs(gain_delta):.1f} dB{direction}てヘッドルームを最適化します。", channel, peak_dbfs, headroom, median_snr, p10_snr, passing, 0.0, gain_delta, level_mask, playback_spl, thresholds)
    return _assessment(True, "ready", "Pilot合格。次のESSから正式ショットとして取得します。", channel, peak_dbfs, headroom, median_snr, p10_snr, passing, 0.0, 0.0, level_mask, playback_spl, thresholds)


def _assessment(ready, action, message, channel, peak, headroom, median, p10, passing, speaker, gain, level_mask, playback_spl, thresholds):
    return PilotAssessment(
        bool(ready),
        str(action),
        str(message),
        int(channel),
        float(peak),
        float(headroom),
        float(median),
        float(p10),
        float(passing),
        float(speaker),
        float(gain),
        active_band_start_hz=float(level_mask.start_hz),
        active_band_end_hz=float(level_mask.end_hz),
        active_band_count=int(level_mask.band_count),
        active_band_span_octaves=float(level_mask.span_octaves),
        level_mask_valid=bool(level_mask.valid),
        playback_spl_db=None if playback_spl is None else float(playback_spl),
        snr_target_db=float(thresholds.median_db),
        snr_minimum_db=float(thresholds.p10_db),
        snr_passing_fraction_required=float(thresholds.passing_fraction),
    )


def _required_speaker_adjustment_db(
    median_snr_db: float,
    p10_snr_db: float,
    settings: MeasurementSettings,
) -> float:
    thresholds = pilot_snr_thresholds(settings)
    if not np.isfinite(median_snr_db) or not np.isfinite(p10_snr_db):
        return float(settings.maximum_speaker_increase_db)
    shortfall = max(
        thresholds.median_db - float(median_snr_db),
        thresholds.p10_db - float(p10_snr_db),
        0.0,
    )
    return float(np.clip(max(shortfall, 1.0), 1.0, settings.maximum_speaker_increase_db))


def _pilot_playback_spl(
    signal_audio: np.ndarray,
    noise: AmbientNoiseProfile,
    channel: int,
    settings: MeasurementSettings,
) -> float | None:
    offset = level_calibration_offset_db(settings)
    if offset is None:
        return None
    total_rms = float(np.sqrt(np.mean(np.asarray(signal_audio, dtype=np.float64) ** 2)))
    noise_rms = float(noise.channel_rms[min(channel, len(noise.channel_rms) - 1)])
    signal_rms = np.sqrt(max(total_rms * total_rms - noise_rms * noise_rms, 1e-30))
    return float(20.0 * np.log10(max(signal_rms, 1e-15)) + offset)


def _preflight_level_mask(
    frequencies: np.ndarray,
    signal_db: np.ndarray,
    measurement_range: np.ndarray,
    settings: MeasurementSettings,
) -> PreflightLevelMask:
    """Find the main reproduced band without changing formal measurement data."""
    frequency = np.asarray(frequencies, dtype=np.float64)
    level = np.asarray(signal_db, dtype=np.float64)
    in_range = np.asarray(measurement_range, dtype=bool)
    empty = np.zeros(frequency.shape, dtype=bool)
    if frequency.shape != level.shape or frequency.shape != in_range.shape or not np.any(in_range):
        return PreflightLevelMask(empty, 0.0, 0.0, 0, 0.0, False)

    finite_range = (
        in_range
        & np.isfinite(level)
        & np.isfinite(frequency)
        & (frequency > 0.0)
        & (level > -250.0)
    )
    if not np.any(finite_range):
        return PreflightLevelMask(empty, 0.0, 0.0, 0, 0.0, False)
    robust_peak = float(np.percentile(level[finite_range], 90))
    candidate = finite_range & (level >= robust_peak + settings.preflight_mask_relative_db)
    indices = np.flatnonzero(candidate)
    if not indices.size:
        return PreflightLevelMask(empty, 0.0, 0.0, 0, 0.0, False)

    split_points = np.flatnonzero(np.diff(indices) > settings.preflight_mask_max_gap_bands + 1) + 1
    groups = np.split(indices, split_points)
    main = max(groups, key=lambda group: (group.size, int(group[-1] - group[0])))
    lower = int(main[0]) + settings.preflight_mask_edge_trim_bands
    upper = int(main[-1]) - settings.preflight_mask_edge_trim_bands
    selected = candidate & (np.arange(candidate.size) >= lower) & (np.arange(candidate.size) <= upper)
    selected_indices = np.flatnonzero(selected)
    if not selected_indices.size:
        return PreflightLevelMask(selected, 0.0, 0.0, 0, 0.0, False)
    start_hz = float(frequency[selected_indices[0]])
    end_hz = float(frequency[selected_indices[-1]])
    span_octaves = float(np.log2(end_hz / start_hz)) if end_hz > start_hz else 0.0
    band_count = int(selected_indices.size)
    valid = band_count >= settings.preflight_mask_min_bands and span_octaves >= settings.preflight_mask_min_octaves
    return PreflightLevelMask(selected, start_hz, end_hz, band_count, span_octaves, valid)


def _channels(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not values.size:
        raise ValueError("audio must contain one or more channels")
    return values


def _frame_rms(values: np.ndarray, frame: int) -> np.ndarray:
    usable = max(1, values.shape[0] // frame)
    trimmed = values[: usable * frame]
    return np.sqrt(np.mean(trimmed.reshape(usable, frame, values.shape[1]) ** 2, axis=1))


def _fractional_octave_rms(values: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(values.shape[0], max(1024, int(2 ** np.floor(np.log2(max(1024, sample_rate // 2))))))
    frequency, psd = signal.welch(values, fs=sample_rate, nperseg=nperseg, axis=0, scaling="density")
    centers = 20.0 * 2.0 ** (np.arange(0, int(np.ceil(12 * np.log2((sample_rate / 2) / 20.0))) + 1) / 12.0)
    output = np.zeros((centers.size, values.shape[1]), dtype=np.float64)
    ratio = 2.0 ** (1.0 / 24.0)
    for index, center in enumerate(centers):
        mask = (frequency >= center / ratio) & (frequency < center * ratio)
        if np.any(mask):
            output[index] = np.sqrt(np.trapezoid(psd[mask], frequency[mask], axis=0))
    return centers, output


def _selected_channel_index(label: str, values: np.ndarray) -> int:
    if values.shape[1] == 1 or label == "mono":
        return 0
    if label.startswith("R"):
        return 1
    if label.startswith("L"):
        return 0
    peaks = np.max(np.abs(values), axis=0)
    usable = np.where(peaks < 0.995, peaks, -1.0)
    return int(np.argmax(usable))
