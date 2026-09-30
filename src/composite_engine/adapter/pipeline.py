from __future__ import annotations

from dataclasses import dataclass, replace
import io
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from ..complex_sum import transform_response
from ..core import MultichannelCompositeResult, result_from_responses
from ..fft import unwrapped_phase_on_reliable_branch
from .package import ChannelInput, CompositePackage, _load_frd_bytes
from ..iir_crossover import IIRCrossoverConfig, iir_crossover_response, sos_is_stable
from ..phase_alignment import AllPassSection, allpass_response
from ..target import target_response_from_magnitude_db, target_impulse_response
from target_engine import apply_target_edit_payload
from response_display.cache import content_cached


@dataclass(frozen=True)
class ResponseStage:
    name: str
    filename: str
    data: bytes


@dataclass(frozen=True)
class ChannelPipelineInput:
    name: str
    way: str
    group: str
    stages: tuple[ResponseStage, ...]
    gain_db: float = 0.0
    polarity: int = 1
    delay_samples: float = 0.0
    iir_crossover: IIRCrossoverConfig = IIRCrossoverConfig()
    dsp_additional_delay_samples: float = 0.0
    auto_alignment_delay_samples: float = 0.0
    auto_alignment_allpass: tuple[AllPassSection, ...] = ()
    iir_sos: tuple[tuple[float, ...], ...] = ()
    speaker_relative_timing_samples: float = 0.0


@dataclass(frozen=True)
class GroupTargetInput:
    group: str
    name: str
    filename: str
    data: bytes
    minimum_phase: bool | None = None
    edit_payload: dict[str, object] | None = None


@content_cached(revision="composite-channel-evaluation-v2", max_entries=4, max_bytes=64*1024*1024)
def build_channel_pipelines(
    sample_rate_hz: int,
    channels: list[ChannelPipelineInput],
    *,
    fft_size: int | None = None,
    group_targets: tuple[GroupTargetInput, ...] = (),
) -> MultichannelCompositeResult:
    """Multiply ordered Channel stages, then sum realized Channels per Group."""
    if int(sample_rate_hz) <= 0:
        raise ValueError("sample rate must be positive")
    if not channels:
        raise ValueError("at least one Channel pipeline is required")
    # A Channel may be intentionally FIR-free.  Its response starts at Unity
    # and can still contain IIR crossover/EQ, gain, polarity and delay stages.
    longest = max(
        (
            _wav_taps(stage)
            for stage in (
                *(stage for channel in channels for stage in channel.stages),
                *(ResponseStage(target.name, target.filename, target.data) for target in group_targets),
            )
        ),
        default=2,
    )
    size = (
        max(2, int(fft_size), longest)
        if fft_size is not None
        else max(2, longest, int(sample_rate_hz))
    )
    if size % 2 == 0:
        size += 1
    frequency = np.fft.rfftfreq(size, 1.0 / int(sample_rate_hz))
    targets: dict[str, tuple[GroupTargetInput, np.ndarray, bool]] = {}
    for target in group_targets:
        if target.group in targets:
            raise ValueError(f"duplicate Group Target: {target.group}")
        target_response = _realize_stage_response(
            ResponseStage(target.name, target.filename, target.data),
            int(sample_rate_hz),
            size,
            complete_boundaries=True,
        )
        phase_available = _phase_available(target.filename, target.data)
        if target.minimum_phase is not None and not phase_available:
            target_response = target_response_from_magnitude_db(
                20.0 * np.log10(np.maximum(np.abs(target_response), 1e-15)),
                minimum_phase=bool(target.minimum_phase),
            )
            phase_available = True
        if target.edit_payload:
            target_response = apply_target_edit_payload(
                target_response, frequency, int(sample_rate_hz), target.edit_payload,
            )
            phase_available = phase_available or any(
                bool(target.edit_payload.get(key))
                for key in ("phase_peq", "phase_shelf", "phase_tilt", "phase_allpass", "iir_filters")
            )
        targets[target.group] = (
            target,
            target_response,
            phase_available,
        )
    realized: dict[str, dict[str, np.ndarray]] = {}
    realized_unwrapped_phase: dict[str, dict[str, np.ndarray]] = {}
    for channel_index, channel in enumerate(channels):
        response = np.ones(frequency.shape, dtype=np.complex128)
        continuous_phase = np.zeros(frequency.shape, dtype=float)
        for stage_index, stage in enumerate(channel.stages):
            stage_response = _realize_stage_response(stage, int(sample_rate_hz), size)
            response *= stage_response
            continuous_phase += _realize_stage_unwrapped_phase(
                stage, stage_response, frequency,
            )
        speaker_timing_samples = float(channel.speaker_relative_timing_samples)
        if not np.isfinite(speaker_timing_samples):
            raise ValueError(f"{channel.name}: Speaker relative timing is not finite")
        if speaker_timing_samples:
            speaker_timing_phase = (
                -2.0 * np.pi * frequency * speaker_timing_samples
                / float(sample_rate_hz)
            )
            response *= np.exp(1j * speaker_timing_phase)
            continuous_phase += speaker_timing_phase
        if channel.iir_sos:
            sos = np.asarray(channel.iir_sos, dtype=float)
            if (
                sos.ndim != 2 or sos.shape[1] != 6
                or not np.isfinite(sos).all() or not sos_is_stable(sos)
            ):
                raise ValueError(f"{channel.name}: Channel IIR SOS is invalid or unstable")
            _frequency, iir_response = signal.sosfreqz(
                sos, worN=frequency, fs=int(sample_rate_hz),
            )
            response *= np.asarray(iir_response, dtype=complex)
            continuous_phase += unwrapped_phase_on_reliable_branch(iir_response)
        crossover_response = iir_crossover_response(
            channel.iir_crossover, frequency, int(sample_rate_hz),
        )
        response *= crossover_response
        continuous_phase += unwrapped_phase_on_reliable_branch(crossover_response)
        response *= int(channel.iir_crossover.lr2_polarity)
        if int(channel.iir_crossover.lr2_polarity) < 0:
            continuous_phase += np.pi
        alignment_allpass_response = allpass_response(
            channel.auto_alignment_allpass, frequency, int(sample_rate_hz),
        )
        response *= alignment_allpass_response
        continuous_phase += unwrapped_phase_on_reliable_branch(
            alignment_allpass_response
        )
        total_delay_samples = (
            float(channel.dsp_additional_delay_samples)
            + float(channel.delay_samples)
            + float(channel.auto_alignment_delay_samples)
        )
        if int(channel.polarity) < 0:
            continuous_phase += np.pi
        continuous_phase -= (
            2.0 * np.pi * frequency * total_delay_samples
            / float(sample_rate_hz)
        )
        response = transform_response(
            response, frequency, int(sample_rate_hz), gain_db=channel.gain_db,
            polarity=channel.polarity,
            delay_samples=total_delay_samples,
        )
        group_responses = realized.setdefault(channel.group, {})
        group_phases = realized_unwrapped_phase.setdefault(channel.group, {})
        if channel.name in group_responses:
            raise ValueError(f"duplicate Channel name in Group {channel.group}: {channel.name}")
        group_responses[channel.name] = response
        group_phases[channel.name] = np.rad2deg(continuous_phase)
    _align_channel_phase_branches(
        channels, realized_unwrapped_phase, frequency,
    )
    unknown_target_groups = set(targets).difference(realized)
    if unknown_target_groups:
        raise ValueError(
            "Group Target has no matching Channel Group: "
            + ", ".join(sorted(unknown_target_groups))
        )
    return MultichannelCompositeResult(
        sample_rate_hz=int(sample_rate_hz),
        groups={
            group: result_from_responses(
                group=group, sample_rate_hz=int(sample_rate_hz), fft_size=size,
                frequency_hz=frequency, channel_responses=responses,
                target_response=targets[group][1] if group in targets else None,
                target_name=targets[group][0].name if group in targets else None,
                target_phase_available=targets[group][2] if group in targets else False,
                channel_unwrapped_phase_deg=realized_unwrapped_phase[group],
            )
            for group, responses in realized.items()
        },
    )


def attach_group_targets(
    multichannel: MultichannelCompositeResult,
    group_targets: tuple[GroupTargetInput, ...],
) -> MultichannelCompositeResult:
    """Attach display-only targets without rebuilding any Channel response."""
    targets = {target.group: target for target in group_targets}
    if len(targets) != len(group_targets):
        raise ValueError("duplicate Group Target")
    unknown_groups = set(targets).difference(multichannel.groups)
    if unknown_groups:
        raise ValueError(
            "Group Target has no matching Channel Group: "
            + ", ".join(sorted(unknown_groups))
        )
    groups = {}
    for group, base in multichannel.groups.items():
        target = targets.get(group)
        if target is None:
            groups[group] = base
            continue
        response = _realize_stage_response(
            ResponseStage(target.name, target.filename, target.data),
            base.sample_rate_hz,
            base.fft_size,
            complete_boundaries=True,
        )
        phase_available = _phase_available(target.filename, target.data)
        if target.minimum_phase is not None and not phase_available:
            response = target_response_from_magnitude_db(
                20.0 * np.log10(np.maximum(np.abs(response), 1e-15)),
                minimum_phase=bool(target.minimum_phase),
            )
            phase_available = True
        if target.edit_payload:
            response = apply_target_edit_payload(
                response, base.frequency_hz, base.sample_rate_hz,
                target.edit_payload,
            )
            phase_available = phase_available or any(
                bool(target.edit_payload.get(key))
                for key in (
                    "phase_peq", "phase_shelf", "phase_tilt",
                    "phase_allpass", "iir_filters",
                )
            )
        groups[group] = replace(
            base,
            target_response=response,
            target_name=target.name,
            target_phase_available=phase_available,
            target_impulse_response=target_impulse_response(response, base.fft_size),
        )
    return MultichannelCompositeResult(
        sample_rate_hz=multichannel.sample_rate_hz,
        groups=groups,
    )


def _wav_taps(stage: ResponseStage) -> int:
    if not stage.filename.casefold().endswith(".wav"):
        return 2
    _sample_rate, values = wavfile.read(io.BytesIO(stage.data))
    return int(values.shape[0])


def _realize_stage_response(stage: ResponseStage, sample_rate_hz: int, fft_size: int, *, complete_boundaries=False) -> np.ndarray:
    if stage.name in {"PhaseEQ Response Processing", "Speaker Package", "Manual Speaker"}:
        if not _phase_available(stage.filename, stage.data):
            raise ValueError(f"{stage.name}: 位相なしのSpeaker応答はチャンネル間の複素合成・時間評価に使用できません。GainはPhaseEQで確認できます。")
    stage_name = "pipeline_stage"
    package = CompositePackage.from_channel_inputs(sample_rate_hz, [ChannelInput(
        name=stage_name,
        way=stage.name,
        group="Pipeline",
        filename=stage.filename,
        data=stage.data,
    )])
    frequency, responses, _ = package.evaluate_responses("Pipeline", fft_size=fft_size)
    response = responses[stage_name]
    if (complete_boundaries or stage.name in {"PhaseEQ Response Processing", "Speaker Package", "Manual Speaker"}) and Path(stage.filename).suffix.casefold() in {".frd", ".csv", ".txt"}:
        from response_completion import complete_response
        values = _load_frd_bytes(stage.data, stage.name)
        gain, phase, _ = complete_response(values[:, 0], values[:, 1],
            values[:, 2] if values.shape[1] >= 3 else None, frequency, phase_is_continuous=True)
        outside = (frequency < values[0, 0]) | (frequency > values[-1, 0])
        completed = 10**(gain / 20) * np.exp(1j * np.deg2rad(0.0 if phase is None else phase))
        response = np.where(outside, completed, response)
    return response


def _realize_stage_unwrapped_phase(
    stage: ResponseStage,
    response: np.ndarray,
    frequency_hz: np.ndarray,
) -> np.ndarray:
    """Preserve an FRD's continuous phase instead of reconstructing its branch."""
    if Path(stage.filename).suffix.casefold() not in {".frd", ".csv", ".txt"}:
        return unwrapped_phase_on_reliable_branch(response)
    values = _load_frd_bytes(stage.data, stage.name)
    if values.shape[1] < 3:
        return np.zeros(np.asarray(frequency_hz).shape, dtype=float)
    source_phase = np.deg2rad(values[:, 2])
    if stage.name in {"PhaseEQ Response Processing", "Speaker Package", "Manual Speaker"}:
        from response_completion import complete_response
        _, phase, _ = complete_response(values[:, 0], values[:, 1], values[:, 2],
            frequency_hz, phase_is_continuous=True)
        return np.deg2rad(phase)
    return np.interp(
        np.asarray(frequency_hz, dtype=float), values[:, 0], source_phase,
        left=source_phase[0], right=source_phase[-1],
    )


def _align_channel_phase_branches(
    channels: list[ChannelPipelineInput],
    phases_by_group: dict[str, dict[str, np.ndarray]],
    frequency_hz: np.ndarray,
) -> None:
    """Join display branches downward while keeping the highest Way stable."""
    way_rank = {
        "SUB": 0,
        "Low": 1,
        "Low-Mid": 2,
        "Mid": 2,
        "High-Mid": 3,
        "High": 4,
    }
    frequency = np.asarray(frequency_hz, dtype=float)
    for group, phases in phases_by_group.items():
        ordered = sorted(
            (
                channel for channel in channels
                if channel.group == group
                and channel.name in phases
                and channel.way in way_rank
            ),
            key=lambda channel: way_rank[channel.way],
        )
        for lower, upper in reversed(tuple(zip(ordered, ordered[1:]))):
            lower_fc = float(
                lower.iir_crossover.lowpass_hz
                or lower.iir_crossover.common_lowpass_hz
            )
            upper_fc = float(
                upper.iir_crossover.highpass_hz
                or upper.iir_crossover.common_highpass_hz
            )
            crossover = lower_fc if lower_fc > 0.0 else upper_fc
            if crossover <= 0.0:
                continue
            index = int(np.argmin(np.abs(frequency - crossover)))
            lower_phase = float(phases[lower.name][index])
            upper_phase = float(phases[upper.name][index])
            branch_shift = 360.0 * round((upper_phase - lower_phase) / 360.0)
            phases[lower.name] = np.asarray(phases[lower.name], dtype=float) + branch_shift


def _phase_available(filename: str, data: bytes) -> bool:
    if filename.casefold().endswith(".wav"):
        return True
    delimiter = "," if filename.casefold().endswith(".csv") else None
    values = np.genfromtxt(io.BytesIO(data), comments="#", delimiter=delimiter)
    return bool(values.ndim == 2 and values.shape[1] >= 3)
