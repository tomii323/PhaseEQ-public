from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import gcd

import numpy as np
from scipy import ndimage, signal

from .models import readonly


@dataclass(frozen=True)
class KnownEssReference:
    samples: np.ndarray
    source_sample_rate: int
    analysis_sample_rate: int
    source_channels: int
    sweep_start_s: float
    sweep_duration_s: float
    shot_period_s: float
    repeat_count: int
    source_name: str = ""
    start_frequency_hz: float = 0.0
    end_frequency_hz: float = 0.0
    source_channel: int = 1
    ess_channels: tuple[int, ...] = (1,)
    variable_loop_gap: bool = False
    silence_duration_s: float = 0.0
    peak_dbfs: float = 0.0
    shot_capture_duration_s: float = 0.0
    timing_marker_samples: np.ndarray | None = None
    timing_marker_to_ess_s: float = 0.0
    intro_samples: np.ndarray | None = None
    intro_duration_s: float = 0.0
    shot_start_samples: tuple[int, ...] = ()
    reference_samples_source: int = 0
    timing_marker_start_samples: tuple[int, ...] = ()
    ess_output_channels: str = "unknown"
    timing_marker_output_channels: str = "unknown"
    terminal_timing_marker: bool = False
    routing_manifest_verified: bool = False

    @property
    def has_timing_marker(self) -> bool:
        return self.timing_marker_samples is not None and self.timing_marker_to_ess_s > 0

    @property
    def has_separate_tweeter_routing(self) -> bool:
        """Return whether a trusted manifest proves exclusive opposite outputs."""

        return bool(
            self.routing_manifest_verified
            and self.source_channels >= 2
            and self.ess_output_channels in {"left", "right"}
            and self.timing_marker_output_channels in {"left", "right"}
            and self.ess_output_channels != self.timing_marker_output_channels
        )


def ess_reference_rejection_reasons(reference: KnownEssReference) -> tuple[str, ...]:
    """Return reasons why an analyzed ESS cannot be used for measurement."""

    reasons: list[str] = []
    samples = np.asarray(reference.samples, dtype=np.float64).reshape(-1)
    if reference.analysis_sample_rate <= 0 or samples.size < round(0.1 * max(1, reference.analysis_sample_rate)):
        reasons.append("ESS Sweepは0.1秒以上必要です。")
    if samples.size == 0 or not np.all(np.isfinite(samples)) or float(np.max(np.abs(samples), initial=0.0)) <= 1e-8:
        reasons.append("ESS Sweepが無音、または無効なサンプルを含んでいます。")
    if not 0.1 <= reference.sweep_duration_s <= 60.0:
        reasons.append("解析されたSweep secondsは0.1～60秒の範囲にしてください。")
    if not 0.02 <= reference.silence_duration_s <= 30.0:
        reasons.append("解析されたSilence secondsは0.02～30秒必要です。0秒のESSは使用できません。")
    if not 3 <= reference.repeat_count <= 1_000:
        reasons.append("実際に使用するESSは3～1000回の連続Sweepが必要です。")
    if reference.shot_period_s <= reference.sweep_duration_s:
        reasons.append("ESSのShot periodがSweepより長くありません。連続Sweepを分離できません。")
    if reference.shot_capture_duration_s and reference.shot_capture_duration_s < reference.sweep_duration_s + 0.02:
        reasons.append("ESSの収録区間にSweep後の無音が0.02秒以上ありません。")
    if not -80.0 <= reference.peak_dbfs <= 0.0:
        reasons.append("ESSレベルが測定用途の範囲外です（-80～0 dBFS）。")
    if reference.has_timing_marker:
        marker_duration = len(np.asarray(reference.timing_marker_samples)) / max(1, reference.analysis_sample_rate)
        if reference.timing_marker_to_ess_s < marker_duration + 0.04:
            reasons.append("Timing markerとESSの間隔が短すぎます。40 ms以上のgapが必要です。")
    total_duration = reference.intro_duration_s + reference.repeat_count * reference.shot_period_s
    if total_duration > 1_800.0:
        reasons.append("ESSトラックは30分以下にしてください。")
    return tuple(dict.fromkeys(reasons))


def tweeter_reference_routing_rejection_reasons(
    reference: KnownEssReference,
) -> tuple[str, ...]:
    """Return blockers specific to acoustic Tweeter-reference measurements."""

    reasons: list[str] = []
    if not reference.has_timing_marker:
        reasons.append("固定TweeterのTiming markerがありません。")
    if not reference.routing_manifest_verified:
        reasons.append(
            "左右Routingを検証できるPhaseEQ manifestがありません。"
            "ESS音声からの推定だけでは距離測定に使用できません。"
        )
    if reference.source_channels < 2:
        reasons.append("距離測定用ESSは2チャンネルである必要があります。")
    if reference.ess_output_channels not in {"left", "right"}:
        reasons.append("測定ESS出力をLeftまたはRightの片側だけにしてください。")
    if reference.timing_marker_output_channels not in {"left", "right"}:
        reasons.append("固定Tweeter marker出力をLeftまたはRightの片側だけにしてください。")
    if (
        reference.ess_output_channels in {"left", "right"}
        and reference.timing_marker_output_channels in {"left", "right"}
        and reference.ess_output_channels == reference.timing_marker_output_channels
    ):
        reasons.append("固定Tweeter markerと測定ESSは反対側の出力へ分離してください。")
    return tuple(dict.fromkeys(reasons))


def extract_known_ess_reference(
    audio: np.ndarray,
    source_sample_rate: int,
    analysis_sample_rate: int,
    *,
    source_name: str = "",
) -> KnownEssReference:
    """Extract one repeated sweep from a narration-plus-sweep CD track.

    The extractor intentionally uses the track's repeated active/silent timing
    rather than assuming a generated chirp. This preserves the exact sweep
    phase, level, fades, and frequency law present on the CD.
    """

    source_rate = int(source_sample_rate)
    target_rate = int(analysis_sample_rate)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("WAV sample rates must be positive")
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < source_rate:
        raise ValueError("Reference WAV must contain at least one second of audio")
    channels = int(values.shape[1])
    selected_channel, ess_channels = _select_repeating_ess_channel(values, source_rate)
    mono = np.asarray(values[:, selected_channel], dtype=np.float64)
    if not np.all(np.isfinite(mono)) or float(np.max(np.abs(mono))) < 1e-8:
        raise ValueError("Reference WAV is silent or contains invalid samples")

    starts, ends = _active_runs(mono, source_rate)
    chain = _longest_repeating_chain(starts, ends, source_rate)
    if chain.size < 3:
        raise ValueError(
            "Could not find at least three repeated sweeps after the narration. "
            "Upload the complete repeating-sweep WAV track."
        )

    repeated_starts = starts[chain]
    repeated_ends = ends[chain]
    period_samples = int(round(float(np.median(np.diff(repeated_starts)))))
    active_samples = int(round(float(np.median(repeated_ends - repeated_starts))))
    padding = max(1, int(round(0.003 * source_rate)))
    sweep_samples = min(period_samples - 1, active_samples + 2 * padding)

    # Use a sweep after the first detected repeat so any narration tail at the
    # transition cannot enter the reference. The track itself remains the
    # source of truth; no synthetic ESS is substituted.
    reference_index = int(chain[min(2, chain.size - 1)])
    sweep_start = max(0, int(starts[reference_index]) - padding)
    sweep_end = min(mono.size, sweep_start + sweep_samples)
    reference = np.asarray(mono[sweep_start:sweep_end], dtype=np.float64)
    if reference.size < int(0.1 * source_rate):
        raise ValueError("Detected sweep is too short to use as an ESS reference")

    reference = _resample(reference, source_rate, target_rate)
    peak = float(np.max(np.abs(reference)))
    if peak <= 1e-12:
        raise ValueError("Extracted ESS reference is silent")

    marker, marker_to_ess_samples = _detect_phaseeq_timing_marker(
        mono,
        np.maximum(0, repeated_starts - padding),
        source_rate,
    )
    capture_samples = period_samples - marker_to_ess_samples if marker is not None else period_samples
    silence_samples = max(1, capture_samples - sweep_samples)
    start_hz, end_hz = _estimate_sweep_frequencies(reference, target_rate)
    first_sweep_start = max(0, int(repeated_starts[0]) - padding)
    inferred_sweep_starts = tuple(max(0, int(value) - padding) for value in repeated_starts)
    inferred_marker_starts = (
        tuple(value - marker_to_ess_samples for value in inferred_sweep_starts if value >= marker_to_ess_samples)
        if marker is not None else ()
    )
    intro = _resample(np.asarray(mono[:first_sweep_start], dtype=np.float64), source_rate, target_rate)
    return KnownEssReference(
        samples=readonly(reference),
        source_sample_rate=source_rate,
        analysis_sample_rate=target_rate,
        source_channels=channels,
        sweep_start_s=sweep_start / source_rate,
        sweep_duration_s=reference.size / target_rate,
        shot_period_s=period_samples / source_rate,
        repeat_count=int(chain.size),
        source_name=str(source_name),
        start_frequency_hz=start_hz,
        end_frequency_hz=end_hz,
        source_channel=selected_channel + 1,
        ess_channels=tuple(channel + 1 for channel in ess_channels),
        variable_loop_gap=True,
        silence_duration_s=silence_samples / source_rate,
        peak_dbfs=20.0 * np.log10(max(peak, 1e-15)),
        shot_capture_duration_s=capture_samples / source_rate,
        timing_marker_samples=(
            readonly(_resample(marker, source_rate, target_rate)) if marker is not None else None
        ),
        timing_marker_to_ess_s=marker_to_ess_samples / source_rate,
        intro_samples=readonly(intro) if intro.size else None,
        intro_duration_s=first_sweep_start / source_rate,
        shot_start_samples=inferred_sweep_starts,
        reference_samples_source=int(sweep_end - sweep_start),
        timing_marker_start_samples=inferred_marker_starts,
    )


def extract_manifest_ess_reference(
    audio: np.ndarray,
    source_sample_rate: int,
    analysis_sample_rate: int,
    manifest_data: bytes | str | dict,
    *,
    source_name: str = "",
) -> KnownEssReference:
    """Load an exact PhaseEQ-generated reference from its sidecar manifest."""

    if isinstance(manifest_data, dict):
        manifest = manifest_data
    else:
        text = manifest_data.decode("utf-8") if isinstance(manifest_data, bytes) else str(manifest_data)
        manifest = json.loads(text)
    if manifest.get("generator") != "PhaseEQ ESS Generator":
        raise ValueError("Manifest is not a PhaseEQ ESS Generator manifest")
    settings = dict(manifest.get("settings", {}))
    if int(settings.get("sample_rate", 0)) != int(source_sample_rate):
        raise ValueError("Manifest and audio sample rates do not match")
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    starts = tuple(int(value) for value in manifest.get("shot_start_samples", ()))
    reference_count = int(manifest.get("reference_samples", 0))
    if not starts or reference_count < 2 or starts[0] < 0 or starts[0] + reference_count > values.shape[0]:
        raise ValueError("Manifest ESS sample positions are invalid")
    if manifest.get("pcm_sha256"):
        quantized = np.rint(np.clip(values, -1.0, (2**23 - 1) / 2**23) * 2**23).astype("<i4")
        digest = hashlib.sha256(quantized.tobytes()).hexdigest()
        if digest != str(manifest["pcm_sha256"]):
            raise ValueError("Audio PCM does not match the ESS manifest hash")
    selected_channel, ess_channels = _select_manifest_ess_channel(values, starts, reference_count)
    mono = np.asarray(values[:, selected_channel], dtype=np.float64)
    reference = mono[starts[0] : starts[0] + reference_count]
    reference = _resample(reference, int(source_sample_rate), int(analysis_sample_rate))
    period_samples = int(manifest.get("shot_period_samples", 0))
    if period_samples <= reference_count:
        raise ValueError("Manifest shot period is invalid")
    ess_output_route = str(settings.get("output_channels", "unknown")).strip().casefold()
    marker_output_route = str(
        manifest.get(
            "timing_marker_output_channels",
            settings.get("timing_marker_output_channels", "unknown"),
        )
    ).strip().casefold()
    expected_ess_channel = {"left": 0, "right": 1}.get(ess_output_route)
    expected_marker_channel = {"left": 0, "right": 1}.get(marker_output_route)
    routing_manifest_verified = bool(
        values.shape[1] >= 2
        and expected_ess_channel is not None
        and expected_marker_channel is not None
        and expected_ess_channel != expected_marker_channel
        and tuple(ess_channels) == (expected_ess_channel,)
    )
    marker_count = int(manifest.get("timing_marker_samples", 0))
    marker_to_ess = int(manifest.get("timing_marker_to_ess_samples", 0))
    marker_starts = tuple(int(value) for value in manifest.get("timing_marker_start_samples", ()))
    marker: np.ndarray | None = None
    if marker_count > 0 and marker_to_ess >= marker_count and marker_starts:
        marker_start = marker_starts[0]
        if marker_start < 0 or marker_start + marker_count > values.shape[0]:
            raise ValueError("Manifest timing marker positions are invalid")
        marker_segment = np.asarray(
            values[marker_start : marker_start + marker_count], dtype=np.float64,
        )
        route_channel = expected_marker_channel
        if route_channel is not None and route_channel < marker_segment.shape[1]:
            marker_channel = route_channel
        else:
            marker_channel = int(np.argmax(np.sum(marker_segment * marker_segment, axis=0)))
        marker = marker_segment[:, marker_channel]
        if float(np.max(np.abs(marker))) <= 1e-12:
            raise ValueError("Manifest timing marker channel is silent")
        if expected_marker_channel is not None and marker_segment.shape[1] >= 2:
            other_channel = 1 - expected_marker_channel
            if float(np.max(np.abs(marker_segment[:, other_channel]))) > 1e-8:
                routing_manifest_verified = False
    capture_samples = int(manifest.get("shot_capture_samples", 0))
    if capture_samples <= reference_count:
        capture_samples = period_samples - marker_to_ess if marker is not None else period_samples
    silence_samples = max(1, capture_samples - reference_count)
    peak = float(np.max(np.abs(reference)))
    intro_end = marker_starts[0] if marker_starts else starts[0]
    intro = _resample(
        np.asarray(mono[:intro_end], dtype=np.float64),
        int(source_sample_rate),
        int(analysis_sample_rate),
    )
    return KnownEssReference(
        samples=readonly(reference),
        source_sample_rate=int(source_sample_rate),
        analysis_sample_rate=int(analysis_sample_rate),
        source_channels=int(values.shape[1]),
        sweep_start_s=starts[0] / int(source_sample_rate),
        sweep_duration_s=reference.size / int(analysis_sample_rate),
        shot_period_s=period_samples / int(source_sample_rate),
        repeat_count=len(starts),
        source_name=str(source_name),
        start_frequency_hz=float(settings.get("start_frequency_hz", 0.0)),
        end_frequency_hz=float(settings.get("end_frequency_hz", 0.0)),
        source_channel=selected_channel + 1,
        ess_channels=tuple(channel + 1 for channel in ess_channels),
        variable_loop_gap=True,
        silence_duration_s=silence_samples / int(source_sample_rate),
        peak_dbfs=20.0 * np.log10(max(peak, 1e-15)),
        shot_capture_duration_s=capture_samples / int(source_sample_rate),
        timing_marker_samples=(
            readonly(_resample(marker, int(source_sample_rate), int(analysis_sample_rate)))
            if marker is not None else None
        ),
        timing_marker_to_ess_s=marker_to_ess / int(source_sample_rate) if marker is not None else 0.0,
        intro_samples=readonly(intro) if intro.size else None,
        intro_duration_s=intro_end / int(source_sample_rate),
        shot_start_samples=starts,
        reference_samples_source=reference_count,
        timing_marker_start_samples=marker_starts,
        ess_output_channels=ess_output_route,
        timing_marker_output_channels=marker_output_route,
        terminal_timing_marker=bool(manifest.get("terminal_timing_marker", False)),
        routing_manifest_verified=routing_manifest_verified,
    )


def build_inferred_ess_manifest(
    audio: np.ndarray,
    source_sample_rate: int,
    reference: KnownEssReference,
    *,
    source_name: str = "",
) -> bytes:
    """Build a reusable PhaseEQ manifest from an analyzed WAV/FLAC track."""

    rate = int(source_sample_rate)
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if rate <= 0 or values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Analyzed ESS audio is empty or invalid")
    reference_count = int(reference.reference_samples_source)
    if reference_count < 2:
        reference_count = int(round(reference.sweep_duration_s * rate))
    period_samples = int(round(reference.shot_period_s * rate))
    capture_samples = int(round(reference.shot_capture_duration_s * rate))
    if capture_samples <= reference_count:
        capture_samples = int(round((reference.sweep_duration_s + reference.silence_duration_s) * rate))
    starts = tuple(int(value) for value in reference.shot_start_samples)
    if not starts:
        selected_start = int(round(reference.sweep_start_s * rate))
        selected_position = min(2, max(0, reference.repeat_count - 1))
        first_start = selected_start - selected_position * period_samples
        starts = tuple(first_start + index * period_samples for index in range(reference.repeat_count))
    starts = tuple(
        value for value in starts
        if value >= 0 and value + reference_count <= values.shape[0]
    )
    if len(starts) < 3:
        raise ValueError("Analyzed ESS positions are insufficient to generate a manifest")
    marker_count = (
        int(round(len(np.asarray(reference.timing_marker_samples)) * rate / reference.analysis_sample_rate))
        if reference.has_timing_marker else 0
    )
    marker_to_ess = int(round(reference.timing_marker_to_ess_s * rate)) if marker_count else 0
    marker_starts = tuple(int(value) for value in reference.timing_marker_start_samples)
    if marker_count and not marker_starts:
        marker_starts = tuple(value - marker_to_ess for value in starts if value >= marker_to_ess)
    quantized = np.rint(np.clip(values, -1.0, (2**23 - 1) / 2**23) * 2**23).astype("<i4")
    output_channels = (
        "both" if len(reference.ess_channels) > 1 else
        "right" if reference.source_channel == 2 else
        "left"
    )
    manifest = {
        "schema_version": 1,
        "generator": "PhaseEQ ESS Generator",
        "generator_version": "analyzed-input-1",
        "manifest_origin": "analyzed_input",
        "source_name": str(source_name),
        "settings": {
            "sample_rate": rate,
            "bit_depth": 24,
            "start_frequency_hz": float(reference.start_frequency_hz),
            "end_frequency_hz": float(min(reference.end_frequency_hz, rate / 2)),
            "sweep_duration_s": reference_count / rate,
            "silence_duration_s": max(1, capture_samples - reference_count) / rate,
            "repeats": len(starts),
            "output_channels": output_channels,
            "timing_marker": bool(marker_count),
            "inferred_from_audio": True,
        },
        "channels": int(values.shape[1]),
        "actual_end_frequency_hz": float(min(reference.end_frequency_hz, rate / 2)),
        "shot_start_samples": list(starts),
        "shot_period_samples": period_samples,
        "shot_capture_samples": capture_samples,
        "reference_start_sample": starts[0],
        "reference_samples": reference_count,
        "timing_marker_start_samples": list(marker_starts),
        "timing_marker_samples": marker_count,
        "timing_marker_to_ess_samples": marker_to_ess,
        "pcm_sha256": hashlib.sha256(quantized.tobytes()).hexdigest(),
        "playback_loop_policy": {
            "file_repeat_gap": "variable",
            "measurement_timing": "start_new_segment_after_each_file_group",
        },
    }
    return json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")


def _detect_phaseeq_timing_marker(
    mono: np.ndarray,
    sweep_starts: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray | None, int]:
    """Detect the standard marker; legacy marker-free ESS returns ``None``."""

    from .generator import generate_timing_marker

    marker = np.asarray(generate_timing_marker(sample_rate), dtype=np.float64)
    expected_gap = int(round(0.120 * sample_rate))
    expected_offset = marker.size + expected_gap
    margin = max(2, int(round(0.015 * sample_rate)))
    scores: list[float] = []
    offsets: list[int] = []
    for sweep_start in sweep_starts[: min(4, sweep_starts.size)]:
        search_start = int(sweep_start) - expected_offset - margin
        search_end = int(sweep_start) - expected_offset + margin + marker.size
        if search_start < 0 or search_end > mono.size:
            continue
        window = np.asarray(mono[search_start:search_end], dtype=np.float64)
        numerator = signal.correlate(window, marker, mode="valid", method="fft")
        energy = np.sqrt(np.maximum(signal.convolve(window * window, np.ones(marker.size), mode="valid"), 0.0))
        denominator = energy * max(float(np.linalg.norm(marker)), 1e-15)
        correlation = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-15)
        best = int(np.argmax(correlation))
        scores.append(float(correlation[best]))
        offsets.append(int(sweep_start) - (search_start + best))
    if len(scores) < min(2, sweep_starts.size) or float(np.median(scores)) < 0.80:
        return None, 0
    offset = int(round(float(np.median(offsets))))
    first_start = int(sweep_starts[0]) - offset
    return np.asarray(mono[first_start : first_start + marker.size], dtype=np.float64), offset


def _estimate_sweep_frequencies(reference: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """Estimate log-sweep endpoints for legacy-file metadata and UI defaults."""

    values = np.asarray(reference, dtype=np.float64)
    if values.size < 64:
        return 0.0, 0.0
    analytic = signal.hilbert(values)
    frequency = np.diff(np.unwrap(np.angle(analytic))) * sample_rate / (2 * np.pi)
    time = (np.arange(frequency.size, dtype=np.float64) + 0.5) / sample_rate
    amplitude = np.abs(analytic[:-1])
    duration = values.size / sample_rate
    valid = (
        (time >= duration * 0.03)
        & (time <= duration * 0.94)
        & (amplitude >= float(np.max(amplitude)) * 0.08)
        & (frequency > 0)
        & (frequency < sample_rate / 2)
    )
    if np.count_nonzero(valid) < 32:
        return 0.0, 0.0
    x = time[valid]
    y = np.log(frequency[valid])
    keep = np.ones(x.size, dtype=bool)
    intercept = 0.0
    slope = 0.0
    for _ in range(4):
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        residual = y - (intercept + slope * x)
        scale = max(1.4826 * float(np.median(np.abs(residual[keep]))), 1e-4)
        keep = np.abs(residual) <= 3.0 * scale
        if np.count_nonzero(keep) < 32:
            return 0.0, 0.0
    start_hz = float(np.exp(intercept))
    end_hz = float(np.exp(intercept + slope * duration))
    end_hz = min(end_hz, sample_rate / 2)
    if end_hz <= start_hz:
        return 0.0, 0.0
    return start_hz, end_hz


def _select_repeating_ess_channel(values: np.ndarray, sample_rate: int) -> tuple[int, tuple[int, ...]]:
    candidates: list[tuple[int, int]] = []
    for channel in range(values.shape[1]):
        starts, ends = _active_runs(np.asarray(values[:, channel], dtype=np.float64), sample_rate)
        chain = _longest_repeating_chain(starts, ends, sample_rate)
        if chain.size >= 3:
            candidates.append((int(starts[int(chain[0])]), channel))
    if not candidates:
        raise ValueError("Could not find repeated ESS on any audio channel")
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1], tuple(sorted(channel for _, channel in candidates))


def _select_manifest_ess_channel(
    values: np.ndarray,
    starts: tuple[int, ...],
    reference_count: int,
) -> tuple[int, tuple[int, ...]]:
    global_peak = float(np.max(np.abs(values)))
    threshold = max(1e-8, global_peak * 1e-5)
    candidates: list[tuple[int, int]] = []
    for channel in range(values.shape[1]):
        earliest: int | None = None
        for start in starts:
            segment = np.abs(values[start : start + reference_count, channel])
            active = np.flatnonzero(segment >= threshold)
            if active.size:
                candidate = int(start + active[0])
                earliest = candidate if earliest is None else min(earliest, candidate)
        if earliest is not None:
            candidates.append((earliest, channel))
    if not candidates:
        raise ValueError("Manifest ESS is silent on every audio channel")
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1], tuple(sorted(channel for _, channel in candidates))


def _active_runs(mono: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    # A 2 ms RMS window splits low-frequency ESS around slow zero crossings.
    # 10 ms remains shorter than the sweep-to-sweep gaps in supported tracks,
    # while keeping OmniMic long and bass sweeps as one continuous active run.
    window = max(8, int(round(0.010 * sample_rate)))
    power = ndimage.uniform_filter1d(mono * mono, size=window, mode="constant")
    rms = np.sqrt(np.maximum(power, 0.0))
    peak_rms = float(np.max(rms))
    threshold = max(1e-7, peak_rms * 0.02)
    active = rms >= threshold
    transitions = np.diff(np.concatenate(([False], active, [False])).astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    duration = ends - starts
    minimum = int(round(0.12 * sample_rate))
    maximum = int(round(10.0 * sample_rate))
    keep = (duration >= minimum) & (duration <= maximum)
    return starts[keep], ends[keep]


def _longest_repeating_chain(starts: np.ndarray, ends: np.ndarray, sample_rate: int) -> np.ndarray:
    if starts.size < 3:
        return np.empty(0, dtype=int)
    durations = ends - starts
    best = np.empty(0, dtype=int)
    minimum_period = int(round(0.25 * sample_rate))
    maximum_period = int(round(15.0 * sample_rate))
    for first in range(starts.size - 2):
        initial_period = int(starts[first + 1] - starts[first])
        if not minimum_period <= initial_period <= maximum_period:
            continue
        chain = [first, first + 1]
        periods = [initial_period]
        for index in range(first + 2, starts.size):
            period = int(starts[index] - starts[chain[-1]])
            expected = float(np.median(periods))
            tolerance = max(int(round(0.012 * sample_rate)), int(round(expected * 0.025)))
            duration_expected = float(np.median(durations[chain]))
            duration_tolerance = max(int(round(0.015 * sample_rate)), int(round(duration_expected * 0.12)))
            if abs(period - expected) <= tolerance and abs(int(durations[index]) - duration_expected) <= duration_tolerance:
                chain.append(index)
                periods.append(period)
            elif period > expected + tolerance:
                break
        if len(chain) > best.size:
            best = np.asarray(chain, dtype=int)
    return best


def _resample(values: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(values, dtype=np.float64)
    divisor = gcd(source_rate, target_rate)
    return signal.resample_poly(values, target_rate // divisor, source_rate // divisor).astype(np.float64)
