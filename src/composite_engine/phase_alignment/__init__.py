from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy import signal

from composite_engine.distance_timing import external_distance_timing

from timing_provenance import (
    causal_delays_from_relative_arrivals,
    normalize_timing_provenance,
    trusted_relative_arrivals,
)


FIR_CROSSOVER_METHODS = frozenset({"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR"})
IIR_CROSSOVER_METHODS = frozenset({"LR2", "LR4"})


from crossover_engine.alignment import AllPassSection, allpass_sos, allpass_response


@dataclass(frozen=True)
class BoundaryAssessment:
    lower: str
    upper: str
    crossover_hz: float
    lower_level_db: float
    upper_level_db: float
    level_difference_db: float
    valid_fraction: float
    caution: str
    allpass_decision: str = ""
    allpass_improvement_db: float = 0.0
    analysis_low_hz: float = 0.0
    analysis_high_hz: float = 0.0
    lower_phase_slope_ms: float = 0.0
    upper_phase_slope_ms: float = 0.0
    aligned_group_delay_difference_ms: float = 0.0


@dataclass(frozen=True)
class ChannelAlignment:
    channel: str
    relative_delay_samples: float
    dsp_delay_samples: float
    allpass_sections: tuple[AllPassSection, ...] = ()


@dataclass(frozen=True)
class PhaseAlignmentProposal:
    reference_channel: str
    common_offset_samples: float
    channels: tuple[ChannelAlignment, ...]
    boundaries: tuple[BoundaryAssessment, ...]
    confidence: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AlignmentBranch:
    name: str
    ordered_channels: tuple[str, ...]


@dataclass(frozen=True)
class SystemPhaseAlignmentProposal:
    reference_channel: str
    common_offset_samples: float
    channels: tuple[ChannelAlignment, ...]
    boundaries: tuple[BoundaryAssessment, ...]
    confidence: str
    warnings: tuple[str, ...]
    branches: tuple[str, ...]
    timing_source: str = "phase_fit"


def accumulate_alignment_proposal(
    proposal: PhaseAlignmentProposal,
    previous_delay_samples: dict[str, float],
    previous_allpass: dict[str, tuple[AllPassSection, ...]],
) -> PhaseAlignmentProposal:
    """Convert a residual analysis proposal into stable absolute DSP settings."""
    channels = []
    for item in proposal.channels:
        old_delay = float(previous_delay_samples.get(item.channel, 0.0))
        old_sections = tuple(previous_allpass.get(item.channel, ()))
        channels.append(replace(
            item,
            relative_delay_samples=old_delay + float(item.relative_delay_samples),
            dsp_delay_samples=old_delay + float(item.dsp_delay_samples),
            allpass_sections=old_sections + tuple(item.allpass_sections),
        ))
    return replace(proposal, channels=tuple(channels))


def analyze_system_phase_alignment(
    frequency_hz: np.ndarray,
    channel_responses: dict[str, np.ndarray],
    branches: tuple[AlignmentBranch, ...],
    crossover_frequencies_hz: tuple[float, ...],
    sample_rate_hz: int,
    *,
    window_oct: float = 0.5,
    allpass_limit: int = 1,
    crossover_methods: tuple[str, ...] | None = None,
    timing_provenance_by_channel: dict[str, object] | None = None,
    external_distances_m_by_channel: dict[str, float] | None = None,
) -> SystemPhaseAlignmentProposal:
    """Analyze all branches, then causalize every physical Channel once."""
    if not branches:
        raise ValueError("at least one alignment branch is required")
    channel_order = list(dict.fromkeys(
        channel for branch in branches for channel in branch.ordered_channels
    ))
    trusted_arrivals = trusted_relative_arrivals(
        tuple(channel_order), timing_provenance_by_channel or {},
        sample_rate_hz=int(sample_rate_hz),
    )
    distance_timing = None
    if external_distances_m_by_channel is not None:
        missing = set(channel_order).difference(external_distances_m_by_channel)
        extra = set(external_distances_m_by_channel).difference(channel_order)
        if missing or extra:
            raise ValueError(
                "External distance Channels do not match the alignment System"
            )
        distance_timing = external_distance_timing(
            external_distances_m_by_channel,
            sample_rate_hz=int(sample_rate_hz),
        )
    authoritative_timing = distance_timing is not None or trusted_arrivals is not None
    proposals: list[tuple[AlignmentBranch, PhaseAlignmentProposal]] = []
    for branch in branches:
        if len(branch.ordered_channels) < 2:
            raise ValueError(f"{branch.name}: alignment branch requires two or more Channels")
        proposals.append((
            branch,
            analyze_phase_alignment(
                frequency_hz,
                channel_responses,
                branch.ordered_channels,
                crossover_frequencies_hz,
                sample_rate_hz,
                window_oct=window_oct,
                allpass_limit=allpass_limit,
                crossover_methods=crossover_methods,
                fit_delays=not authoritative_timing,
            ),
        ))

    relative_by_channel: dict[str, float] = {}
    sections_by_channel: dict[str, tuple[AllPassSection, ...]] = {}
    proposal_channel_order: list[str] = []
    boundaries: list[BoundaryAssessment] = []
    warnings: list[str] = []
    confidence_rank = {"Low": 0, "Medium": 1, "High": 2}
    confidence = "High"
    for branch, proposal in proposals:
        confidence = min(
            (confidence, proposal.confidence),
            key=lambda value: confidence_rank.get(value, 0),
        )
        warnings.extend(f"{branch.name}: {warning}" for warning in proposal.warnings)
        boundaries.extend(proposal.boundaries)
        for item in proposal.channels:
            relative = float(item.relative_delay_samples)
            sections = tuple(item.allpass_sections)
            if item.channel in relative_by_channel:
                if not np.isclose(relative_by_channel[item.channel], relative, atol=1e-9):
                    raise ValueError(
                        f"{item.channel}: shared Channel delay proposal is inconsistent"
                    )
                if sections_by_channel[item.channel] != sections:
                    raise ValueError(
                        f"{item.channel}: shared Channel All-pass proposal is inconsistent"
                    )
                continue
            proposal_channel_order.append(item.channel)
            relative_by_channel[item.channel] = relative
            sections_by_channel[item.channel] = sections

    timing_source = "phase_fit"
    reference_channel = " / ".join(
        branch.ordered_channels[0] for branch in branches
    )
    if proposal_channel_order != channel_order:
        raise ValueError("Alignment proposal Channel order is inconsistent")
    if distance_timing is not None:
        dsp_delays = distance_timing.delay_samples
        quantized_offset = 0.0
        timing_source = "external_distance"
        reference_channel = distance_timing.reference_channel
        warnings.append(
            "外部測定距離を最優先し、最も遠いChannelへ到達時間を揃えました "
            f"(公称音速 {distance_timing.speed_of_sound_m_s:.2f} m/s)"
        )
        confidence = "High"
    elif trusted_arrivals is not None:
        dsp_delays = causal_delays_from_relative_arrivals(trusted_arrivals)
        quantized_offset = 0.0
        timing_source = "phaseeq_tweeter_reference"
        first_provenance = normalize_timing_provenance(
            (timing_provenance_by_channel or {}).get(channel_order[0])
        )
        if first_provenance is not None:
            reference_channel = first_provenance.reference_tweeter_channel_id
        warnings.append(
            "同一SessionのPhaseEQ Tweeter基準・相対到達時間を最優先で適用しました"
        )
        confidence = "High"
    else:
        if timing_provenance_by_channel and any(
            value is not None for value in timing_provenance_by_channel.values()
        ):
            warnings.append(
                "Tweeter基準が全Channelで同一Session／有効状態ではないため、"
                "Crossover位相推定へ全体fallbackしました"
            )
        common_offset = max(0.0, -min(relative_by_channel.values(), default=0.0))
        dsp_delays = {
            channel: float(np.floor(relative + common_offset + 0.5))
            for channel, relative in relative_by_channel.items()
        }
        quantized_offset = float(np.floor(common_offset + 0.5))
    applied_responses = {
        channel: (
            np.asarray(channel_responses[channel], dtype=complex)
            * allpass_response(
                sections_by_channel[channel], frequency_hz, int(sample_rate_hz),
            )
            * np.exp(
                -1j * 2.0 * np.pi * np.asarray(frequency_hz, dtype=float)
                * float(dsp_delays[channel]) / float(sample_rate_hz)
            )
        )
        for channel in channel_order
    }
    verified_boundaries: list[BoundaryAssessment] = []
    assessment_index = 0
    for branch in branches:
        for boundary_index, crossover in enumerate(crossover_frequencies_hz):
            assessment = boundaries[assessment_index]
            assessment_index += 1
            lower = branch.ordered_channels[boundary_index]
            upper = branch.ordered_channels[boundary_index + 1]
            mask, weights = _reliable_mask(
                np.asarray(frequency_hz, dtype=float),
                applied_responses[lower], applied_responses[upper],
                float(crossover), float(window_oct),
            )
            if np.count_nonzero(mask) >= 8:
                lower_delay = _robust_phase_delay_samples(
                    np.asarray(frequency_hz, dtype=float)[mask],
                    applied_responses[lower][mask], weights[mask], int(sample_rate_hz),
                )
                upper_delay = _robust_phase_delay_samples(
                    np.asarray(frequency_hz, dtype=float)[mask],
                    applied_responses[upper][mask], weights[mask], int(sample_rate_hz),
                )
                assessment = replace(
                    assessment,
                    aligned_group_delay_difference_ms=(
                        float(upper_delay - lower_delay) * 1_000.0
                        / float(sample_rate_hz)
                    ),
                )
            verified_boundaries.append(assessment)
    return SystemPhaseAlignmentProposal(
        reference_channel=reference_channel,
        common_offset_samples=quantized_offset,
        channels=tuple(
            ChannelAlignment(
                channel=channel,
                relative_delay_samples=float(
                    dsp_delays[channel] - quantized_offset
                ),
                dsp_delay_samples=dsp_delays[channel],
                allpass_sections=sections_by_channel[channel],
            )
            for channel in channel_order
        ),
        boundaries=tuple(verified_boundaries),
        confidence=confidence,
        warnings=tuple(warnings),
        branches=tuple(branch.name for branch in branches),
        timing_source=timing_source,
    )


def analyze_phase_alignment(
    frequency_hz: np.ndarray,
    channel_responses: dict[str, np.ndarray],
    ordered_channels: tuple[str, ...],
    crossover_frequencies_hz: tuple[float, ...],
    sample_rate_hz: int,
    *,
    window_oct: float = 0.5,
    allpass_limit: int = 1,
    crossover_methods: tuple[str, ...] | None = None,
    fit_delays: bool = True,
) -> PhaseAlignmentProposal:
    """Fit upward from the lowest-frequency Channel, adding delay to higher Ways."""
    if len(ordered_channels) < 2 or len(crossover_frequencies_hz) != len(ordered_channels) - 1:
        raise ValueError("Channel and crossover boundary counts do not match")
    if any(name not in channel_responses for name in ordered_channels):
        raise ValueError("all ordered Channels require a realized Complex Response")
    if allpass_limit not in range(max(2, len(ordered_channels) - 2) + 1):
        raise ValueError("All-pass limit exceeds the channel topology")
    methods = crossover_methods or tuple("LR4" for _ in crossover_frequencies_hz)
    if len(methods) != len(crossover_frequencies_hz):
        raise ValueError("Crossover method count does not match boundaries")
    if any(method not in FIR_CROSSOVER_METHODS | IIR_CROSSOVER_METHODS | {"Through"} for method in methods):
        raise ValueError("Unsupported crossover method for phase alignment")
    frequency = np.asarray(frequency_hz, dtype=float)
    sections, compensation_channels = _structural_allpass_topology(
        ordered_channels, crossover_frequencies_hz, methods, allpass_limit,
    )
    assessments: list[BoundaryAssessment] = []
    warnings: list[str] = []
    valid_fractions: list[float] = []
    # Assess boundaries in the same low-to-high order used by the solver.
    for boundary_index in range(len(crossover_frequencies_hz)):
        lower = ordered_channels[boundary_index]
        upper = ordered_channels[boundary_index + 1]
        crossover = float(crossover_frequencies_hz[boundary_index])
        lower_response = np.asarray(channel_responses[lower], dtype=complex)
        upper_response = np.asarray(channel_responses[upper], dtype=complex)
        mask, weights = _reliable_mask(
            frequency, lower_response, upper_response, crossover, float(window_oct),
        )
        valid_fraction = float(np.count_nonzero(mask)) / max(1, int(np.count_nonzero(
            (frequency >= crossover / 2.0 ** window_oct)
            & (frequency <= crossover * 2.0 ** window_oct)
        )))
        valid_fractions.append(valid_fraction)
        level_mask = (
            (frequency >= crossover / 2.0 ** (1.0 / 12.0))
            & (frequency <= crossover * 2.0 ** (1.0 / 12.0))
        )
        lower_level = _median_db(lower_response[level_mask])
        upper_level = _median_db(upper_response[level_mask])
        level_difference = abs(lower_level - upper_level)
        caution = ""
        if level_difference > 6.0:
            caution = "Fc周辺音圧差が6 dBを超えています"
        elif level_difference > 3.0:
            caution = "Fc周辺音圧差が3 dBを超えています"
        elif level_difference > 1.5:
            caution = "Fc周辺音圧差が1.5 dBを超えています"
        if caution:
            warnings.append(f"{lower}/{upper}: {caution} ({level_difference:.2f} dB)")
        analysis_low = crossover / 2.0 ** float(window_oct)
        analysis_high = crossover * 2.0 ** float(window_oct)
        lower_slope_ms = 0.0
        upper_slope_ms = 0.0
        if np.count_nonzero(mask) >= 8:
            lower_slope_ms = (
                _robust_phase_delay_samples(
                    frequency[mask], lower_response[mask], weights[mask], int(sample_rate_hz),
                ) * 1_000.0 / float(sample_rate_hz)
            )
            upper_slope_ms = (
                _robust_phase_delay_samples(
                    frequency[mask], upper_response[mask], weights[mask], int(sample_rate_hz),
                ) * 1_000.0 / float(sample_rate_hz)
            )
        if allpass_limit == 0:
            allpass_decision = "未使用: All-pass能力が使用不可"
        elif methods[boundary_index] == "Through":
            allpass_decision = "未使用: スルー境界の外部ネットワーク方式は推定しません"
        elif methods[boundary_index] in FIR_CROSSOVER_METHODS:
            allpass_decision = f"不要: {methods[boundary_index]}境界"
        elif compensation_channels[boundary_index]:
            recipients = " / ".join(compensation_channels[boundary_index])
            allpass_decision = (
                f"理論配置: {recipients}へ{methods[boundary_index]}境界補償"
            )
        else:
            allpass_decision = "不要: 全Channelがこの境界位相を通過済み"
        assessments.append(BoundaryAssessment(
            lower, upper, crossover, lower_level, upper_level,
            level_difference, valid_fraction, caution,
            allpass_decision=allpass_decision,
            analysis_low_hz=analysis_low,
            analysis_high_hz=analysis_high,
            lower_phase_slope_ms=lower_slope_ms,
            upper_phase_slope_ms=upper_slope_ms,
        ))
        if np.count_nonzero(mask) < 8:
            warnings.append(f"{lower}/{upper}: 有効な位相点が不足しています")
            assessments[-1] = replace(
                assessments[-1], caution=(
                    f"{caution} / 有効な位相点が不足" if caution else "有効な位相点が不足"
                ),
            )
            continue
    # Keep the lowest-frequency Channel at zero. Align the next higher Channel
    # to the already-corrected lower Channel, then use that corrected result as
    # the reference for the following boundary (Low→Mid→High for 3Way).
    relative = {ordered_channels[0]: 0.0}
    corrected_responses = {
        ordered_channels[0]: np.asarray(
            channel_responses[ordered_channels[0]], dtype=complex,
        ) * allpass_response(
            sections[ordered_channels[0]], frequency, int(sample_rate_hz),
        )
    }
    for boundary_index in range(len(crossover_frequencies_hz)):
        lower = ordered_channels[boundary_index]
        upper = ordered_channels[boundary_index + 1]
        crossover = float(crossover_frequencies_hz[boundary_index])
        lower_corrected = corrected_responses[lower]
        upper_response = np.asarray(channel_responses[upper], dtype=complex) * allpass_response(
            sections[upper], frequency, int(sample_rate_hz),
        )
        mask, weights = _reliable_mask(
            frequency, lower_corrected, upper_response, crossover, float(window_oct),
        )
        if not fit_delays or np.count_nonzero(mask) < 8:
            relative[upper] = 0.0
        else:
            fitted_delay = -_robust_relative_delay(
                frequency[mask], lower_corrected[mask], upper_response[mask],
                weights[mask], int(sample_rate_hz),
            )
            optimized_delay = float(_optimize_pair_delay(
                frequency[mask], upper_response[mask], lower_corrected[mask],
                weights[mask], int(sample_rate_hz), crossover, fitted_delay,
            ))
            delay_limit_samples = float(sample_rate_hz) * 0.02
            if not np.isfinite(optimized_delay):
                raise ValueError(f"{lower}/{upper}: 相対Delayを有限値として推定できません")
            if abs(optimized_delay) > delay_limit_samples:
                estimated_ms = optimized_delay * 1_000.0 / float(sample_rate_hz)
                raise ValueError(
                    f"{lower}/{upper}: 相対Delay推定 {estimated_ms:.2f} ms が"
                    "解析範囲 ±20 ms を超えました。Speaker入力と時間基準を確認してください"
                )
            relative[upper] = optimized_delay
        corrected_responses[upper] = upper_response * np.exp(
            -1j * 2.0 * np.pi * frequency * relative[upper] / float(sample_rate_hz)
        )
    # Sequential fitting may require advancing a higher-frequency Way. DSP
    # delay is causal, so translate every signed relative delay by one shared
    # offset. This preserves all pairwise phase/group-delay relationships while
    # making every applied delay non-negative.
    common_offset = max(0.0, -min(relative.values(), default=0.0))
    # Downstream DSPs accept only whole-sample Delay. Quantize after causal
    # translation so rounding can never create a negative setting, then derive
    # the displayed low-reference values from the actual applied integers.
    dsp_delays = {
        name: float(np.floor(relative[name] + common_offset + 0.5))
        for name in ordered_channels
    }
    quantized_offset = dsp_delays[ordered_channels[0]]
    # Verify the response that will actually be applied.  The solver uses
    # fractional candidates, but the DSP contract is whole-sample Delay; a
    # pre-quantization diagnostic can incorrectly report a perfect result.
    applied_responses = {
        name: (
            np.asarray(channel_responses[name], dtype=complex)
            * allpass_response(sections[name], frequency, int(sample_rate_hz))
            * np.exp(
                -1j * 2.0 * np.pi * frequency * dsp_delays[name]
                / float(sample_rate_hz)
            )
        )
        for name in ordered_channels
    }
    for boundary_index, assessment in enumerate(assessments):
        lower = ordered_channels[boundary_index]
        upper = ordered_channels[boundary_index + 1]
        crossover = float(crossover_frequencies_hz[boundary_index])
        mask, weights = _reliable_mask(
            frequency, applied_responses[lower], applied_responses[upper],
            crossover, float(window_oct),
        )
        if np.count_nonzero(mask) < 8:
            continue
        lower_delay = _robust_phase_delay_samples(
            frequency[mask], applied_responses[lower][mask], weights[mask],
            int(sample_rate_hz),
        )
        upper_delay = _robust_phase_delay_samples(
            frequency[mask], applied_responses[upper][mask], weights[mask],
            int(sample_rate_hz),
        )
        assessments[boundary_index] = replace(
            assessment,
            aligned_group_delay_difference_ms=(
                float(upper_delay - lower_delay) * 1_000.0
                / float(sample_rate_hz)
            ),
        )
    confidence = (
        "Low" if min(valid_fractions, default=0.0) < 0.4 or any(a.level_difference_db > 6.0 for a in assessments)
        else "Medium" if warnings else "High"
    )
    return PhaseAlignmentProposal(
        reference_channel=ordered_channels[0],
        common_offset_samples=quantized_offset,
        channels=tuple(
            ChannelAlignment(
                channel=name,
                relative_delay_samples=float(dsp_delays[name] - quantized_offset),
                dsp_delay_samples=dsp_delays[name],
                allpass_sections=sections[name],
            )
            for name in ordered_channels
        ),
        boundaries=tuple(assessments),
        confidence=confidence,
        warnings=tuple(warnings),
    )


def structural_allpass_sections(ordered_channels, crossover_frequencies_hz, methods):
    """Calculate theoretical phase compensation without measurements or delay fitting."""
    if len(crossover_frequencies_hz) != len(ordered_channels) - 1 or len(methods) != len(crossover_frequencies_hz):
        raise ValueError("Channel and crossover boundary counts do not match")
    if any(method not in FIR_CROSSOVER_METHODS | IIR_CROSSOVER_METHODS | {"Through"} for method in methods):
        raise ValueError("Unsupported crossover method for phase alignment")
    if any(not np.isfinite(fc) or fc <= 0 for fc in crossover_frequencies_hz):
        raise ValueError("Crossover frequencies must be positive and finite")
    sections, _ = _structural_allpass_topology(
        ordered_channels, crossover_frequencies_hz, methods, max(0, len(ordered_channels) - 2),
    )
    return sections


def _structural_allpass_topology(
    ordered_channels: tuple[str, ...],
    crossover_frequencies_hz: tuple[float, ...],
    methods: tuple[str, ...],
    block_limit: int,
) -> tuple[dict[str, tuple[AllPassSection, ...]], tuple[tuple[str, ...], ...]]:
    sections: dict[str, list[AllPassSection]] = {name: [] for name in ordered_channels}
    blocks_used = {name: 0 for name in ordered_channels}
    recipients: list[tuple[str, ...]] = []
    for boundary_index, (frequency, method) in enumerate(zip(crossover_frequencies_hz, methods)):
        boundary_recipients: list[str] = []
        if method not in IIR_CROSSOVER_METHODS or block_limit == 0:
            recipients.append(())
            continue
        for channel_index, channel in enumerate(ordered_channels):
            if channel_index in {boundary_index, boundary_index + 1}:
                continue
            if blocks_used[channel] >= block_limit:
                continue
            if method == "LR2":
                # Above-boundary recipients require the LR2 constant 180° sign
                # that is otherwise supplied by the adjacent Channel polarity.
                polarity = -1 if channel_index > boundary_index + 1 else 1
                block = (AllPassSection(
                    float(frequency), 0.0, order=1, polarity=polarity,
                ),)
            else:
                block = (AllPassSection(
                    float(frequency), 1.0 / np.sqrt(2.0), order=2,
                ),)
            sections[channel].extend(block)
            blocks_used[channel] += 1
            boundary_recipients.append(channel)
        recipients.append(tuple(boundary_recipients))
    return (
        {name: tuple(values) for name, values in sections.items()},
        tuple(recipients),
    )
def _reliable_mask(frequency, lower, upper, crossover, window_oct):
    band = (
        (frequency >= crossover / 2.0 ** window_oct)
        & (frequency <= crossover * 2.0 ** window_oct)
    )
    lower_db = 20.0 * np.log10(np.maximum(np.abs(lower), 1e-15))
    upper_db = 20.0 * np.log10(np.maximum(np.abs(upper), 1e-15))
    threshold = (
        (lower_db >= np.max(lower_db[band]) - 24.0)
        & (upper_db >= np.max(upper_db[band]) - 24.0)
    ) if np.any(band) else np.zeros(frequency.shape, dtype=bool)
    mask = band & threshold & np.isfinite(lower_db) & np.isfinite(upper_db)
    weights = np.zeros(frequency.shape, dtype=float)
    if np.any(mask):
        pair_level = np.minimum(lower_db, upper_db)
        weights[mask] = np.clip(10.0 ** ((pair_level[mask] - np.max(pair_level[mask])) / 20.0), 0.05, 1.0)
    return mask, weights


def _robust_relative_delay(frequency, lower, upper, weights, sample_rate):
    phase = np.unwrap(np.angle(lower * np.conj(upper)))
    x = np.asarray(frequency, dtype=float)
    w = np.asarray(weights, dtype=float)
    slope = 0.0
    for _iteration in range(5):
        design = np.column_stack((x, np.ones_like(x)))
        solution = np.linalg.lstsq(design * np.sqrt(w[:, None]), phase * np.sqrt(w), rcond=None)[0]
        slope = float(solution[0])
        residual = phase - design @ solution
        scale = max(float(np.median(np.abs(residual - np.median(residual)))) * 1.4826, 1e-6)
        w = weights * np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-12))
    return slope * float(sample_rate) / (2.0 * np.pi)


def _robust_phase_delay_samples(frequency, response, weights, sample_rate):
    phase = np.unwrap(np.angle(response))
    x = np.asarray(frequency, dtype=float)
    base_weights = np.asarray(weights, dtype=float)
    fitted_weights = base_weights.copy()
    slope = 0.0
    for _iteration in range(5):
        design = np.column_stack((x, np.ones_like(x)))
        solution = np.linalg.lstsq(
            design * np.sqrt(fitted_weights[:, None]),
            phase * np.sqrt(fitted_weights),
            rcond=None,
        )[0]
        slope = float(solution[0])
        residual = phase - design @ solution
        scale = max(
            float(np.median(np.abs(residual - np.median(residual)))) * 1.4826,
            1e-6,
        )
        fitted_weights = base_weights * np.minimum(
            1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-12),
        )
    return -slope * float(sample_rate) / (2.0 * np.pi)


def _optimize_pair_delay(frequency, lower, upper, weights, sample_rate, crossover, fitted_delay):
    # Search one half crossover period around the phase-slope estimate. In a
    # sequential Low->Mid->High solve, the upper boundary needs an absolute
    # delay relative to an already-corrected Mid; centering the search at zero
    # can truncate the valid solution at higher crossovers.
    span = float(sample_rate) / (2.0 * float(crossover))
    candidates = np.unique(np.concatenate((
        np.linspace(float(fitted_delay) - span, float(fitted_delay) + span, 801),
        np.linspace(float(fitted_delay) - span / 8.0, float(fitted_delay) + span / 8.0, 201),
        np.array([0.0, float(fitted_delay)]),
    )))
    normalized_weight = np.asarray(weights, dtype=float) / max(float(np.sum(weights)), 1e-12)
    denominator = np.maximum(np.abs(lower) + np.abs(upper), 1e-15)
    best_delay = 0.0
    best_score = -np.inf
    for delay in candidates:
        shifted = lower * np.exp(
            -1j * 2.0 * np.pi * frequency * float(delay) / float(sample_rate)
        )
        coherence = np.sum(normalized_weight * np.abs(shifted + upper) / denominator)
        score = float(coherence) - 1e-7 * abs(float(delay))
        if score > best_score:
            best_score = score
            best_delay = float(delay)
    return best_delay


def _median_db(response):
    return float(np.median(20.0 * np.log10(np.maximum(np.abs(response), 1e-15)))) if response.size else -300.0
