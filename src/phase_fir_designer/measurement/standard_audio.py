"""Versioned external playback catalogue and deterministic, streamed PCM rendering."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import signal
import soundfile as sf

from .generator import generate_timing_marker, generate_level_check_signal
from .reference_track import KnownEssReference, _resample

SET_VERSION = "set1"
MEDIA = {"cd": (44100, 16)}
NOISE_DBFS = -41.0103
ESS_DBFS = NOISE_DBFS + 20


@dataclass(frozen=True)
class StandardTrack:
    number: int
    key: str
    title: str
    kind: str
    output: str
    sweep_s: float = 2
    tail_s: float = 2
    period_s: float = 7
    repeats: int = 17

    @property
    def start_hz(self):
        return 500.0 if self.kind == "tweeter" else 2.0

    def asset_id(self, media: str) -> str:
        if media not in MEDIA:
            raise ValueError("Unknown playback medium")
        return f"phaseeq-{SET_VERSION}-{media}-{self.key}"

    def filename(self, media: str) -> str:
        rate, bits = MEDIA[media]
        return f"{self.number:02d}_{self.title}_{rate}Hz_{bits}bit_{SET_VERSION}.wav"


def _catalogue():
    tracks = [StandardTrack(1, "level_common", "音量調整・全帯域", "noise", "both")]
    for output, suffix in (("both", "左右"), ("left", "左"), ("right", "右")):
        for kind, label, sweep, tail, period, repeats in (
            ("response", "フルレンジ", 2, 2, 7, 17),
            ("tweeter", "ツイーター", 2, 1, 5, 23),
            ("distortion", "歪み", 8, 4, 17, 10),
            ("decay", "残響", 8, 10, 23, 8),
        ):
            tracks.append(StandardTrack(len(tracks)+1, f"{kind}_{output}", f"{label}・{suffix}", kind, output, sweep, tail, period, repeats))
    tracks.extend((StandardTrack(14, "timing_left_ref", "相対測距・左基準", "timing", "right"),
                   StandardTrack(15, "timing_right_ref", "相対測距・右基準", "timing", "left")))
    return tuple(tracks)


TRACKS = _catalogue()


def track_for(kind: str, output: str) -> StandardTrack:
    return next(t for t in TRACKS if t.kind == kind and t.output == output)


def _pcm(values, bits, seed):
    values = np.asarray(values, dtype=np.float64)
    scale = 2**(bits-1)
    if bits == 16:
        rng = np.random.default_rng(seed)
        values = values + (rng.random(values.shape) - rng.random(values.shape)) / scale
    if np.max(np.abs(values)) >= 1:
        raise ValueError("Standard PCM would clip")
    pcm = (np.rint(values * scale) / scale).astype(np.float32)
    pcm[pcm == 0] = 0.0  # PCM has no signed zero.
    return pcm


def render_track(track: StandardTrack, media: str, destination: Path) -> dict:
    """Keep at most one noise channel or one ESS cycle in working memory."""
    fs, bits = MEDIA[media]
    subtype = f"PCM_{bits}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema_version=1, marker_layout="per_shot_start_end", set_version=SET_VERSION,
                    asset_id=track.asset_id(media), track_key=track.key, number=track.number,
                    media=media, sample_rate=fs, bit_depth=bits, kind=track.kind,
                    output=track.output, start_hz=track.start_hz, end_hz=22000,
                    sweep_s=track.sweep_s, tail_s=track.tail_s, period_s=track.period_s,
                    repeats=track.repeats, noise_rms_dbfs=NOISE_DBFS, ess_rms_dbfs=ESS_DBFS)
    if track.kind == "noise":
        # Disk-backed channels avoid keeping two full double-precision noise FFTs.
        with sf.SoundFile(destination, "w", samplerate=fs, channels=2, subtype=subtype) as out:
            out.write(np.zeros((fs, 2), dtype=np.float32))
            out.seek(0)
            channels = []
            for ch in range(2):
                noise = generate_level_check_signal(sample_rate=fs, duration_s=120, rms_dbfs=NOISE_DBFS,
                    start_frequency_hz=2, end_frequency_hz=22000, output_channels="left", seed=0x504551+ch)
                values = np.array(noise.stereo_samples[:, 0], copy=True)
                del noise
                fade = int(.05 * fs)
                values *= 10**(NOISE_DBFS/20) / np.sqrt(np.mean(values[fade:-fade].astype(float)**2))
                channels.append(_pcm(values, bits, 201+ch))
            for start in range(0, len(channels[0]), fs):
                out.write(np.column_stack([v[start:start+fs] for v in channels]))
            manifest["correlation"] = float(np.corrcoef(*channels)[0, 1])
            manifest["channel_rms_dbfs"] = [float(20*np.log10(np.sqrt(np.mean(v[fade:-fade].astype(float)**2)))) for v in channels]
            if abs(manifest["correlation"]) > .1:
                raise ValueError("Noise channels are too correlated")
        manifest["frames"] = 120 * fs
    else:
        count = int(track.sweep_s * fs)
        time = np.arange(count) / fs
        sweep = signal.chirp(time, track.start_hz, track.sweep_s, 22000, method="logarithmic", phi=-90)
        fade = int(.015 * fs)
        sweep[:fade] *= np.sin(np.linspace(0, np.pi/2, fade))**2
        sweep[-fade:] *= np.cos(np.linspace(0, np.pi/2, fade))**2
        sweep *= 10**(ESS_DBFS/20) / np.sqrt(np.mean(sweep[fade:-fade]**2))
        sweep = _pcm(sweep, bits, 101)
        marker = _pcm(generate_timing_marker(fs, duration_s=.25, peak_dbfs=-24), bits, 102)
        cycle = np.zeros((int(track.period_s*fs), 2), dtype=np.float32)
        start = int(1.25 * fs)
        end_marker = start + count + int(track.tail_s*fs)
        channels = (0, 1) if track.output == "both" else ((0,) if track.output == "left" else (1,))
        marker_channels = (1-channels[0],) if track.kind == "timing" else channels
        for ch in channels:
            cycle[start:start+count, ch] = sweep
        for ch in marker_channels:
            cycle[:len(marker), ch] = marker
            cycle[end_marker:end_marker+len(marker), ch] = marker
        with sf.SoundFile(destination, "w", samplerate=fs, channels=2, subtype=subtype) as out:
            out.write(np.zeros((3*fs, 2), dtype=np.float32))
            for _ in range(track.repeats):
                out.write(cycle)
            out.write(np.zeros((3*fs, 2), dtype=np.float32))
        manifest.update(frames=(6+track.repeats*track.period_s)*fs,
            reference_start=3*fs+start, reference_count=count, reference_channel=channels[0],
            reference_sha256=hashlib.sha256(sweep.tobytes()).hexdigest(),
            shots=[dict(start_marker=3*fs+i*len(cycle), ess_start=3*fs+i*len(cycle)+start,
                        capture_end=3*fs+i*len(cycle)+end_marker,
                        end_marker=3*fs+i*len(cycle)+end_marker) for i in range(track.repeats)],
            actual_ess_rms_dbfs=float(20*np.log10(np.sqrt(np.mean(sweep[fade:-fade].astype(float)**2)))))
        if abs(manifest["actual_ess_rms_dbfs"]-ESS_DBFS) > .1:
            raise ValueError("PCM ESS level is outside tolerance")
    manifest["file_sha256"] = file_hash(destination)
    return manifest


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_standard_reference(path: Path, manifest: dict, analysis_rate=48000) -> KnownEssReference:
    if manifest.get("schema_version") != 1 or manifest.get("marker_layout") != "per_shot_start_end":
        raise ValueError("Unsupported standard audio manifest")
    if manifest["kind"] == "noise" or file_hash(path) != manifest["file_sha256"]:
        raise ValueError("Standard audio pair is invalid")
    fs = manifest["sample_rate"]
    with sf.SoundFile(path) as source:
        if source.samplerate != fs or len(source) != manifest["frames"]:
            raise ValueError("Standard audio format mismatch")
        source.seek(manifest["reference_start"])
        samples = source.read(manifest["reference_count"], dtype="float32", always_2d=True)[:, manifest["reference_channel"]]
    if hashlib.sha256(samples.tobytes()).hexdigest() != manifest["reference_sha256"]:
        raise ValueError("Reference PCM mismatch")
    return KnownEssReference(samples=_resample(samples, fs, analysis_rate), source_sample_rate=fs,
        analysis_sample_rate=analysis_rate, source_channels=2, sweep_start_s=4.25,
        sweep_duration_s=manifest["sweep_s"], shot_period_s=manifest["period_s"],
        repeat_count=manifest["repeats"], source_name=path.name, start_frequency_hz=manifest["start_hz"],
        end_frequency_hz=manifest["end_hz"], source_channel=manifest["reference_channel"]+1,
        peak_dbfs=float(20*np.log10(np.max(np.abs(samples)))),
        shot_capture_duration_s=manifest["sweep_s"]+manifest["tail_s"],
        variable_loop_gap=True, ess_output_channels=manifest["output"])
