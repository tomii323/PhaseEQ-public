from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from typing import Any

import numpy as np
from scipy import signal

from .models import readonly


GENERATOR_VERSION = 2
MIN_MEASUREMENT_SWEEP_S = 0.100
MAX_MEASUREMENT_SWEEP_S = 60.0
MIN_MEASUREMENT_SILENCE_S = 0.050
MAX_MEASUREMENT_SILENCE_S = 30.0
MIN_MEASUREMENT_REPEATS = 3
MAX_MEASUREMENT_REPEATS = 1_000
MAX_GENERATED_DURATION_S = 600.0
MAX_GENERATED_FRAMES = 24_000_000


@dataclass(frozen=True)
class EssGeneratorSettings:
    sample_rate: int = 48_000
    bit_depth: int = 24
    start_frequency_hz: float = 10.0
    end_frequency_hz: float = 22_000.0
    sweep_duration_s: float = 2.0
    silence_duration_s: float = 1.0
    repeats: int = 8
    leader_silence_s: float = 5.0
    trailer_silence_s: float = 3.0
    peak_dbfs: float = -18.0
    fade_s: float = 0.015
    start_cue: bool = True
    output_channels: str = "both"
    timing_marker: bool = True
    timing_marker_duration_s: float = 0.080
    timing_marker_gap_s: float = 0.120
    timing_marker_peak_dbfs: float = -24.0
    timing_marker_output_channels: str = "same"
    terminal_timing_marker: bool = False

    def __post_init__(self) -> None:
        if self.sample_rate not in {48_000, 96_000, 192_000}:
            raise ValueError("sample_rate must be 48000, 96000, or 192000 Hz")
        if self.bit_depth != 24:
            raise ValueError("ESS export currently uses 24-bit PCM")
        if not 0 < self.start_frequency_hz < self.end_frequency_hz <= self.sample_rate / 2:
            raise ValueError("ESS frequencies must be positive, ordered, and no higher than Nyquist")
        if self.sweep_duration_s <= 0 or self.silence_duration_s <= 0:
            raise ValueError("sweep and silence durations must be positive")
        if self.repeats < 1:
            raise ValueError("repeats must be at least one")
        if self.fade_s < 0 or self.fade_s * 2 >= self.sweep_duration_s:
            raise ValueError("fade duration is invalid")
        if self.peak_dbfs > 0:
            raise ValueError("peak level cannot exceed 0 dBFS")
        if self.output_channels not in {"both", "left", "right"}:
            raise ValueError("output_channels must be both, left, or right")
        if self.timing_marker_duration_s < 0.040 or self.timing_marker_duration_s > 0.250:
            raise ValueError("timing marker duration must be between 40 and 250 ms")
        if self.timing_marker_gap_s < 0.040:
            raise ValueError("timing marker gap must be at least 40 ms")
        if self.timing_marker_peak_dbfs > 0:
            raise ValueError("timing marker peak cannot exceed 0 dBFS")
        if self.timing_marker_output_channels not in {"same", "both", "left", "right"}:
            raise ValueError("timing marker output channels must be same, both, left, or right")


@dataclass(frozen=True)
class GeneratedEssBundle:
    stereo_samples: np.ndarray
    reference_samples: np.ndarray
    shot_start_samples: tuple[int, ...]
    manifest: dict[str, Any]
    wav_bytes: bytes
    flac_bytes: bytes
    manifest_bytes: bytes
    timing_marker_samples: np.ndarray | None = None


@dataclass(frozen=True)
class GeneratedLevelCheckSignal:
    """Standalone broadband-noise file used only for playback-level setup."""

    stereo_samples: np.ndarray
    sample_rate: int
    rms_dbfs: float
    output_channels: str
    start_frequency_hz: float
    end_frequency_hz: float
    wav_bytes: bytes


PRESETS: dict[str, dict[str, Any]] = {
    "General measurement": dict(sample_rate=48_000, start_frequency_hz=10.0, end_frequency_hz=22_000.0, sweep_duration_s=2.0, silence_duration_s=1.0, repeats=8, peak_dbfs=-18.0),
    "Quick check": dict(sample_rate=48_000, start_frequency_hz=20.0, end_frequency_hz=20_000.0, sweep_duration_s=1.0, silence_duration_s=0.5, repeats=5, peak_dbfs=-24.0),
    "Compatibility": dict(sample_rate=48_000, start_frequency_hz=10.0, end_frequency_hz=22_000.0, sweep_duration_s=2.0, silence_duration_s=0.75, repeats=8, peak_dbfs=-12.0),
    "Precision": dict(sample_rate=96_000, start_frequency_hz=10.0, end_frequency_hz=40_000.0, sweep_duration_s=4.0, silence_duration_s=1.0, repeats=8, peak_dbfs=-12.0),
    "Tweeter extended": dict(sample_rate=192_000, start_frequency_hz=500.0, end_frequency_hz=50_000.0, sweep_duration_s=4.0, silence_duration_s=1.0, repeats=8, peak_dbfs=-24.0),
    "Noisy environment": dict(sample_rate=96_000, start_frequency_hz=10.0, end_frequency_hz=40_000.0, sweep_duration_s=4.0, silence_duration_s=1.0, repeats=12, peak_dbfs=-18.0),
}


def generator_preset(name: str) -> EssGeneratorSettings:
    if name not in PRESETS:
        raise ValueError(f"unknown ESS preset: {name}")
    return EssGeneratorSettings(**PRESETS[name])


def ess_generator_rejection_reasons(settings: EssGeneratorSettings) -> tuple[str, ...]:
    """Return reasons why settings are unsuitable for real microphone measurement."""

    reasons: list[str] = []
    if settings.sweep_duration_s < MIN_MEASUREMENT_SWEEP_S:
        reasons.append(f"Sweep secondsは{MIN_MEASUREMENT_SWEEP_S:g}秒以上にしてください。")
    elif settings.sweep_duration_s > MAX_MEASUREMENT_SWEEP_S:
        reasons.append(f"Sweep secondsは{MAX_MEASUREMENT_SWEEP_S:g}秒以下にしてください。")
    if settings.silence_duration_s < MIN_MEASUREMENT_SILENCE_S:
        reasons.append(f"Silence secondsは{MIN_MEASUREMENT_SILENCE_S:g}秒以上にしてください。0秒では連続Sweepを分離できません。")
    elif settings.silence_duration_s > MAX_MEASUREMENT_SILENCE_S:
        reasons.append(f"Silence secondsは{MAX_MEASUREMENT_SILENCE_S:g}秒以下にしてください。")
    if settings.repeats < MIN_MEASUREMENT_REPEATS:
        reasons.append(f"Repeatsは{MIN_MEASUREMENT_REPEATS}回以上にしてください。連続測定と自動解析には3回以上必要です。")
    elif settings.repeats > MAX_MEASUREMENT_REPEATS:
        reasons.append(f"Repeatsは{MAX_MEASUREMENT_REPEATS}回以下にしてください。")
    if not -60.0 <= settings.peak_dbfs <= -3.0:
        reasons.append("Peak dBFSは-60～-3 dBFSにしてください。0 dBFS付近は再生系でクリップする可能性があります。")
    if settings.start_frequency_hz > 0 and settings.end_frequency_hz / settings.start_frequency_hz < 2.0:
        reasons.append("Sweepの開始周波数と終了周波数は1 octave以上離してください。")
    if settings.start_frequency_hz > 0 and settings.sweep_duration_s * settings.start_frequency_hz < 2.0:
        minimum = 2.0 / settings.start_frequency_hz
        reasons.append(
            f"{settings.start_frequency_hz:g} Hzから開始するにはSweep secondsを{minimum:.3g}秒以上にしてください（最低2周期）。"
        )
    minimum_silence = max(MIN_MEASUREMENT_SILENCE_S, min(0.5, settings.sweep_duration_s * 0.05))
    if settings.silence_duration_s < minimum_silence:
        reasons.append(
            f"Sweep {settings.sweep_duration_s:g}秒との組み合わせではSilence secondsを{minimum_silence:.3g}秒以上にしてください。"
        )
    if settings.start_cue and settings.leader_silence_s < 0.2:
        reasons.append("1 kHz start cueを使用する場合はLeader secondsを0.2秒以上にしてください。")
    marker_overhead = (
        settings.timing_marker_duration_s + settings.timing_marker_gap_s
        if settings.timing_marker else 0.0
    )
    total_duration = (
        settings.leader_silence_s
        + settings.repeats * (marker_overhead + settings.sweep_duration_s + settings.silence_duration_s)
        + settings.trailer_silence_s
        + (
            settings.timing_marker_duration_s
            if settings.timing_marker and settings.terminal_timing_marker else 0.0
        )
    )
    if total_duration > MAX_GENERATED_DURATION_S:
        reasons.append(f"生成ファイルは{MAX_GENERATED_DURATION_S / 60:g}分以下にしてください（現在 {total_duration / 60:.1f}分）。")
    estimated_frames = int(math.ceil(max(0.0, total_duration) * max(0, settings.sample_rate)))
    if estimated_frames > MAX_GENERATED_FRAMES:
        reasons.append(
            "Sample rate・時間・Repeatsの組み合わせが大きすぎます。"
            f"生成フレーム数を{MAX_GENERATED_FRAMES:,}以下にしてください（現在 約{estimated_frames:,}）。"
        )
    return tuple(dict.fromkeys(reasons))


def generate_ess_bundle(settings: EssGeneratorSettings) -> GeneratedEssBundle:
    rejection_reasons = ess_generator_rejection_reasons(settings)
    if rejection_reasons:
        raise ValueError("ESS settings rejected: " + " / ".join(rejection_reasons))
    sample_rate = settings.sample_rate
    sweep_count = max(2, int(round(settings.sweep_duration_s * sample_rate)))
    duration = sweep_count / sample_rate
    time = np.arange(sweep_count, dtype=np.float64) / sample_rate
    # A sampled sinusoid exactly at Nyquist is phase-degenerate. Keep the
    # excitation on the highest representable bin below it while the app's
    # response and FIR grids remain available through the Nyquist bin.
    maximum_end = sample_rate / 2 - sample_rate / sweep_count
    actual_end = min(float(settings.end_frequency_hz), maximum_end)
    sweep = signal.chirp(
        time,
        f0=float(settings.start_frequency_hz),
        f1=actual_end,
        t1=duration,
        method="logarithmic",
        phi=-90.0,
    )
    fade = min(sweep_count // 4, int(round(settings.fade_s * sample_rate)))
    if fade > 1:
        envelope = np.ones(sweep_count, dtype=np.float64)
        envelope[:fade] = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
        envelope[-fade:] = np.cos(np.linspace(0, np.pi / 2, fade)) ** 2
        sweep *= envelope
    sweep *= 10 ** (settings.peak_dbfs / 20.0)

    return generate_marker_ess_bundle(
        sweep,
        sample_rate,
        silence_duration_s=settings.silence_duration_s,
        repeats=settings.repeats,
        leader_silence_s=settings.leader_silence_s,
        trailer_silence_s=settings.trailer_silence_s,
        start_cue=settings.start_cue,
        start_cue_peak_dbfs=min(settings.peak_dbfs - 6.0, -18.0),
        output_channels=settings.output_channels,
        timing_marker=settings.timing_marker,
        timing_marker_duration_s=settings.timing_marker_duration_s,
        timing_marker_gap_s=settings.timing_marker_gap_s,
        timing_marker_peak_dbfs=settings.timing_marker_peak_dbfs,
        timing_marker_output_channels=settings.timing_marker_output_channels,
        terminal_timing_marker=settings.terminal_timing_marker,
        settings_manifest=asdict(settings),
        actual_end_frequency_hz=actual_end,
    )


def generate_level_check_signal(
    *,
    sample_rate: int = 48_000,
    duration_s: float = 3.0,
    rms_dbfs: float = -30.0,
    output_channels: str = "both",
    start_frequency_hz: float = 500.0,
    end_frequency_hz: float = 2_000.0,
    seed: int = 0x504551,
) -> GeneratedLevelCheckSignal:
    """Generate a safe, standalone pink-noise level-check WAV.

    The file intentionally has no ESS timing marker or measurement manifest,
    so it cannot be selected as an ESS analysis reference.
    """

    if sample_rate not in {44_100, 48_000, 96_000, 192_000}:
        raise ValueError("sample_rate must be 44100, 48000, 96000, or 192000 Hz")
    if not 1.0 <= float(duration_s) <= 120.0:
        raise ValueError("level-check duration must be between 1 and 120 seconds")
    if not -60.0 <= float(rms_dbfs) <= -18.0:
        raise ValueError("level-check RMS must be between -60 and -18 dBFS")
    if output_channels not in {"both", "left", "right"}:
        raise ValueError("output_channels must be both, left, or right")
    if not (
        0.0 < float(start_frequency_hz) < float(end_frequency_hz) < sample_rate / 2
    ):
        raise ValueError("level-check frequencies must be positive, ordered, and below Nyquist")

    count = max(2, int(round(float(duration_s) * sample_rate)))
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal(count)
    # Frequency-domain 1/sqrt(f) shaping gives equal energy per octave.
    spectrum = np.fft.rfft(noise)
    frequency = np.fft.rfftfreq(count, d=1.0 / sample_rate)
    shaping = np.zeros_like(frequency)
    shaping[1:] = 1.0 / np.sqrt(frequency[1:])
    noise = np.fft.irfft(spectrum * shaping, n=count)

    # Remove inaudible/DC energy, limit ultrasonic energy, then set broadband
    # RMS.  The final limiter only protects the file from an improbable random
    # peak; normal output retains the requested RMS.
    sos = signal.butter(
        4,
        [float(start_frequency_hz), float(end_frequency_hz)],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    noise = signal.sosfilt(sos, noise)
    fade_count = min(count // 4, int(round(0.050 * sample_rate)))
    if fade_count > 1:
        envelope = np.ones(count, dtype=np.float64)
        envelope[:fade_count] = np.sin(np.linspace(0.0, np.pi / 2.0, fade_count)) ** 2
        envelope[-fade_count:] = np.cos(np.linspace(0.0, np.pi / 2.0, fade_count)) ** 2
        noise *= envelope
    rms = float(np.sqrt(np.mean(noise * noise)))
    if not np.isfinite(rms) or rms <= 0.0:
        raise RuntimeError("level-check noise generation failed")
    noise *= 10.0 ** (float(rms_dbfs) / 20.0) / rms
    peak = float(np.max(np.abs(noise)))
    maximum_peak = 10.0 ** (-3.0 / 20.0)
    if peak > maximum_peak:
        noise *= maximum_peak / peak

    silent = np.zeros_like(noise)
    stereo = {
        "both": np.column_stack((noise, noise)),
        "left": np.column_stack((noise, silent)),
        "right": np.column_stack((silent, noise)),
    }[output_channels]
    quantized = _quantize_24(stereo)
    rendered = quantized.astype(np.float32) / (2**23)
    actual_rms = float(np.sqrt(np.mean(rendered[:, 0 if output_channels != "right" else 1] ** 2)))
    return GeneratedLevelCheckSignal(
        stereo_samples=readonly(rendered),
        sample_rate=sample_rate,
        rms_dbfs=float(20.0 * np.log10(max(actual_rms, 1e-15))),
        output_channels=output_channels,
        start_frequency_hz=float(start_frequency_hz),
        end_frequency_hz=float(end_frequency_hz),
        wav_bytes=_encode_audio(quantized, sample_rate, format_name="WAV"),
    )


def generate_timing_marker(
    sample_rate: int,
    *,
    duration_s: float = 0.080,
    peak_dbfs: float = -24.0,
) -> np.ndarray:
    """Generate the PhaseEQ acoustic timing marker signature."""

    count = max(16, int(round(float(duration_s) * int(sample_rate))))
    marker = np.zeros(count, dtype=np.float64)
    burst_count = max(8, int(round(count * 0.375)))
    gap_count = max(1, int(round(count * 0.125)))
    second_start = burst_count + gap_count
    second_count = min(burst_count, count - second_start)
    for start, length, f0, f1 in (
        (0, burst_count, 5_000.0, min(18_000.0, sample_rate / 2 * 0.92)),
        (second_start, second_count, min(18_000.0, sample_rate / 2 * 0.92), 7_000.0),
    ):
        if length < 2:
            continue
        time = np.arange(length, dtype=np.float64) / sample_rate
        burst = signal.chirp(time, f0=min(f0, sample_rate / 2 * 0.85), f1=min(f1, sample_rate / 2 * 0.92), t1=max(time[-1], 1 / sample_rate), method="linear")
        burst *= signal.windows.tukey(length, alpha=0.35)
        marker[start : start + length] = burst
    marker *= 10 ** (float(peak_dbfs) / 20.0)
    return readonly(marker)


def generate_marker_ess_bundle(
    reference_samples: np.ndarray,
    sample_rate: int,
    *,
    silence_duration_s: float,
    repeats: int,
    leader_silence_s: float = 2.0,
    trailer_silence_s: float = 2.0,
    start_cue: bool = False,
    start_cue_peak_dbfs: float = -24.0,
    output_channels: str = "both",
    timing_marker: bool = True,
    timing_marker_duration_s: float = 0.080,
    timing_marker_gap_s: float = 0.120,
    timing_marker_peak_dbfs: float = -24.0,
    timing_marker_output_channels: str = "same",
    terminal_timing_marker: bool = False,
    settings_manifest: dict[str, Any] | None = None,
    actual_end_frequency_hz: float = 0.0,
    leader_audio: np.ndarray | None = None,
) -> GeneratedEssBundle:
    """Repeat an exact ESS waveform with optional machine timing markers."""

    sample_rate = int(sample_rate)
    sweep = np.asarray(reference_samples, dtype=np.float64).reshape(-1)
    if sample_rate <= 0 or sweep.size < 2 or not np.all(np.isfinite(sweep)):
        raise ValueError("reference ESS is empty or invalid")
    if repeats < 1 or silence_duration_s <= 0:
        raise ValueError("repeats and silence duration must be positive")
    if output_channels not in {"both", "left", "right"}:
        raise ValueError("output_channels must be both, left, or right")
    marker_output_channels = (
        output_channels if timing_marker_output_channels == "same"
        else timing_marker_output_channels
    )
    if marker_output_channels not in {"both", "left", "right"}:
        raise ValueError("timing marker output channels must be same, both, left, or right")

    if leader_audio is not None:
        leader = np.asarray(leader_audio, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(leader)):
            raise ValueError("leader audio contains invalid samples")
        leader = leader.copy()
    else:
        leader = np.zeros(max(0, int(round(leader_silence_s * sample_rate))), dtype=np.float64)
    if start_cue and leader.size:
        cue_count = min(leader.size, int(round(0.2 * sample_rate)))
        cue_time = np.arange(cue_count) / sample_rate
        cue = 10 ** (float(start_cue_peak_dbfs) / 20.0) * np.sin(2 * np.pi * 1_000 * cue_time)
        cue *= signal.windows.tukey(cue_count, alpha=0.2)
        leader[:cue_count] = cue
    silence = np.zeros(max(1, int(round(silence_duration_s * sample_rate))), dtype=np.float64)
    trailer = np.zeros(max(0, int(round(trailer_silence_s * sample_rate))), dtype=np.float64)
    marker = (
        np.asarray(generate_timing_marker(
            sample_rate,
            duration_s=timing_marker_duration_s,
            peak_dbfs=timing_marker_peak_dbfs,
        ))
        if timing_marker else np.empty(0, dtype=np.float64)
    )
    marker_gap = np.zeros(max(0, int(round(timing_marker_gap_s * sample_rate))), dtype=np.float64)
    def routed(values: np.ndarray, route: str) -> np.ndarray:
        silent = np.zeros_like(values)
        return {
            "both": np.column_stack((values, values)),
            "left": np.column_stack((values, silent)),
            "right": np.column_stack((silent, values)),
        }[route]

    parts = [routed(leader, output_channels)]
    starts: list[int] = []
    marker_starts: list[int] = []
    position = leader.size
    for _ in range(repeats):
        if marker.size:
            marker_starts.append(position)
            parts.extend((routed(marker, marker_output_channels), routed(marker_gap, "both")))
            position += marker.size + marker_gap.size
        starts.append(position)
        parts.extend((routed(sweep, output_channels), routed(silence, "both")))
        position += sweep.size + silence.size
    if marker.size and terminal_timing_marker:
        marker_starts.append(position)
        parts.append(routed(marker, marker_output_channels))
        position += marker.size
    parts.append(routed(trailer, "both"))
    stereo = np.concatenate(parts, axis=0)
    quantized = _quantize_24(stereo)
    pcm_hash = hashlib.sha256(quantized.astype("<i4", copy=False).tobytes()).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generator": "PhaseEQ ESS Generator",
        "generator_version": GENERATOR_VERSION,
        "settings": dict(settings_manifest or {
            "sample_rate": sample_rate,
            "bit_depth": 24,
            "sweep_duration_s": sweep.size / sample_rate,
            "silence_duration_s": silence.size / sample_rate,
            "repeats": repeats,
            "output_channels": output_channels,
            "timing_marker": timing_marker,
            "timing_marker_duration_s": timing_marker_duration_s,
            "timing_marker_gap_s": timing_marker_gap_s,
            "timing_marker_peak_dbfs": timing_marker_peak_dbfs,
            "timing_marker_output_channels": timing_marker_output_channels,
            "terminal_timing_marker": terminal_timing_marker,
            "preserved_intro": leader_audio is not None,
        }),
        "channels": 2,
        "actual_end_frequency_hz": float(actual_end_frequency_hz),
        "shot_start_samples": starts,
        "shot_period_samples": sweep.size + silence.size + marker.size + marker_gap.size,
        "shot_capture_samples": sweep.size + silence.size,
        "reference_start_sample": starts[0],
        "reference_samples": sweep.size,
        "timing_marker_start_samples": marker_starts,
        "timing_marker_samples": marker.size,
        "timing_marker_to_ess_samples": marker.size + marker_gap.size if marker.size else 0,
        "timing_marker_output_channels": marker_output_channels if marker.size else "none",
        "terminal_timing_marker": bool(marker.size and terminal_timing_marker),
        "pcm_sha256": pcm_hash,
        "playback_loop_policy": {
            "file_repeat_gap": "variable",
            "measurement_timing": "start_new_segment_after_each_file_group",
        },
    }
    wav_bytes = _encode_audio(quantized, sample_rate, format_name="WAV")
    flac_bytes = _encode_audio(quantized, sample_rate, format_name="FLAC")
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    return GeneratedEssBundle(
        stereo_samples=readonly(quantized.astype(np.float32) / (2**23)),
        reference_samples=readonly(_quantize_24(sweep[:, None])[:, 0].astype(np.float64) / (2**23)),
        shot_start_samples=tuple(starts),
        manifest=manifest,
        wav_bytes=wav_bytes,
        flac_bytes=flac_bytes,
        manifest_bytes=manifest_bytes,
        timing_marker_samples=readonly(marker) if marker.size else None,
    )


def _quantize_24(values: np.ndarray) -> np.ndarray:
    maximum = 2**23 - 1
    return np.rint(np.clip(np.asarray(values, dtype=np.float64), -1.0, maximum / 2**23) * 2**23).astype(np.int32)


def _encode_audio(pcm24: np.ndarray, sample_rate: int, *, format_name: str) -> bytes:
    import soundfile as sf

    output = io.BytesIO()
    sf.write(output, pcm24.astype(np.float64) / (2**23), sample_rate, format=format_name, subtype="PCM_24")
    return output.getvalue()
