from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..config import DesignConfig, IIRFilter, LinearFIRFilter, SpeakerResponse
from ..analysis_policy import AnalysisPolicy, normalize_analysis_fft_size
from ..correction_pipeline import CorrectionResult, correction_from_design_result
from ..export_pipeline import realized_filter_complex_response
from ..project_runtime import ProjectRuntimeSnapshot
from ..response_axis import finite_response_samples, finite_unwrapped_phase_samples
from ..linear_fir import linear_fir_overlay
from ..iir import iir_frequency_response
from .crossover_bank import CrossoverWayResult
from .latency import WayLatencyEstimate, estimate_way_latency
from .models import DSPDevice, DSPWay
from .way_fir_synthesis import WayFIRSynthesisResult, WayFIRTargetStage, synthesize_way_fir


FLAT_SOURCE_REVISION = "flat-source-v1"


@dataclass(frozen=True)
class DSPWayResult:
    way: DSPWay
    device: DSPDevice
    project_revision: str
    original_response: SpeakerResponse | None
    transient_config: DesignConfig
    correction: CorrectionResult
    frequency: np.ndarray
    iir_response: np.ndarray
    fir: np.ndarray
    fir_response: np.ndarray
    original_complex_response: np.ndarray
    acoustic_response: np.ndarray
    fir_latency_ms: float
    crossover_fir: np.ndarray | None
    crossover_iir_filters: tuple[IIRFilter, ...]
    crossover_description: str
    crossover_natural_fir_taps: int
    alignment_fir: np.ndarray | None
    alignment_iir_filters: tuple[IIRFilter, ...]
    latency: WayLatencyEstimate
    fir_synthesis: WayFIRSynthesisResult
    analysis_fft_size: int
    original_impulse: np.ndarray | None = None
    original_impulse_sample_rate: int | None = None


def regenerate_way(
    project: ProjectRuntimeSnapshot | None,
    device: DSPDevice,
    way: DSPWay,
    *,
    compute_group_delay: bool = True,
    speaker_response: SpeakerResponse | None = None,
    target_response: SpeakerResponse | None = None,
    crossover: CrossoverWayResult | None = None,
    crossover_mask: tuple[LinearFIRFilter, ...] = (),
    retained_fir: np.ndarray | None = None,
    input_iir_filters: tuple[IIRFilter, ...] = (),
    fir_master_enabled: bool = True,
    analysis_fft_size: int = 16_384,
) -> DSPWayResult:
    device.validate()
    way.validate()
    if way.dsp_id != device.id:
        raise ValueError("way does not belong to the supplied DSP device")
    common_analysis_fft_size = normalize_analysis_fft_size(analysis_fft_size)
    effective_analysis_fft_size = AnalysisPolicy(
        common_analysis_fft_size
    ).effective_fft_size(required_length=int(device.taps))
    if (
        project is not None
        and way.source_package_id
        and project.project_id
        and way.source_package_id != project.project_id
    ):
        raise ValueError("Way Speaker Package id does not match the supplied source")
    if project is None:
        nyquist = float(device.sample_rate) / 2.0
        flat_response = SpeakerResponse(
            [0.0, nyquist / 2.0, nyquist],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )
        base_config = DesignConfig(
            sample_rate=int(device.sample_rate),
            taps=int(device.taps),
            analysis_fft_size=effective_analysis_fft_size,
            speaker_response=flat_response,
            target_response=flat_response,
        )
        project_revision = FLAT_SOURCE_REVISION
    else:
        base_config = way.correction_config or project.config
        project_revision = project.revision
    effective_speaker = speaker_response or base_config.speaker_response
    if effective_speaker is None:
        raise ValueError("Project does not contain a processed Speaker/Input response for FIR regeneration")
    design_speaker = _response_after_iir(
        effective_speaker,
        tuple(item for item in input_iir_filters if item.enabled),
        int(device.sample_rate),
    )

    # Crossover Plan is the sole source of the Way band mask.  Output Linear
    # FIR is a separate correction stage and is never inferred from DSPWay.
    active_mask = tuple(crossover_mask)
    output_fir_config = replace(
        base_config,
        sample_rate=int(device.sample_rate),
        taps=int(device.taps),
        analysis_fft_size=effective_analysis_fft_size,
    ).normalized()
    output_fir_overlay = linear_fir_overlay(output_fir_config)
    transient = replace(
        base_config,
        sample_rate=int(device.sample_rate),
        taps=int(device.taps),
        analysis_fft_size=effective_analysis_fft_size,
        linear_fir_filters=list(active_mask),
        linear_fir_eq_mask_smoothing_oct=float(way.eq_mask_smoothing_oct),
        # Input IIR is part of the source seen by the Way FIR, but remains a
        # separate physical stage in output/export.  Do not add it to
        # transient.iir_filters or it would be applied twice.
        speaker_response=design_speaker,
        target_response=target_response if target_response is not None else base_config.target_response,
        iir_filters=(
            list(base_config.iir_filters)
            + list(way.manual_iir_filters)
            + list(way.auto_iir_filters)
        ),
    ).normalized()
    crossover_fir = None if crossover is None or crossover.fir is None else np.asarray(crossover.fir, dtype=float)
    stages: list[WayFIRTargetStage] = []
    if output_fir_overlay is not None:
        stages.append(WayFIRTargetStage("Linear FIR", output_fir_overlay))
    if crossover_fir is not None:
        stages.append(WayFIRTargetStage("Crossover", crossover_fir))
    alignment_fir = np.asarray(way.alignment_fir, dtype=float) if way.alignment_fir else None
    if alignment_fir is not None:
        stages.append(WayFIRTargetStage("Alignment", alignment_fir))
    fir_active = bool(fir_master_enabled and device.fir_supported and way.fir_enabled)
    # FIR OFF is always a true bypass: retain the editor settings, but remove
    # the FIR transfer and its latency from the active signal path.
    fir_synthesis = synthesize_way_fir(
        transient,
        stages,
        enabled=fir_active,
        compute_group_delay=compute_group_delay,
    )
    correction = correction_from_design_result(
        transient,
        fir_synthesis.design_result,
        apply_linear_fir_mask=bool(active_mask),
    )
    combined_fir = np.asarray(fir_synthesis.fir, dtype=float)
    frequency, fir_response = realized_filter_complex_response(
        combined_fir,
        int(device.sample_rate),
        effective_analysis_fft_size,
    )
    source = speaker_response or (
        project.processed_speaker_response or project.raw_speaker_response
        if project is not None
        else effective_speaker
    )
    original_complex = response_complex_on_axis(source, frequency)
    project_iir_response = _complex_on_axis(correction.frequency, correction.iir_response, frequency)
    crossover_iir_filters = () if crossover is None else tuple(crossover.iir_filters)
    alignment_iir_filters = tuple(item for item in way.alignment_iir_filters if item.enabled)
    crossover_iir_response = iir_frequency_response(
        crossover_iir_filters + alignment_iir_filters,
        int(device.sample_rate),
        frequency,
    )
    iir_response = project_iir_response * crossover_iir_response
    acoustic = original_complex * iir_response * fir_response
    all_iir_filters = (
        tuple(item for item in transient.iir_filters if item.enabled)
        + crossover_iir_filters
        + alignment_iir_filters
    )
    latency = estimate_way_latency(
        combined_fir,
        all_iir_filters,
        int(device.sample_rate),
        frequency,
        active_response=acoustic,
        fixed_processing_delay_ms=float(device.latency_offset_ms),
        common_processing_cancels=True,
    )
    return DSPWayResult(
        way=way,
        device=device,
        project_revision=project_revision,
        original_response=source,
        transient_config=transient,
        correction=correction,
        frequency=frequency,
        iir_response=iir_response,
        fir=combined_fir.copy(),
        fir_response=fir_response,
        original_complex_response=original_complex,
        acoustic_response=acoustic,
        fir_latency_ms=float(latency.fir.scalar_delay_ms),
        crossover_fir=None if crossover_fir is None else crossover_fir.copy(),
        crossover_iir_filters=crossover_iir_filters,
        crossover_description="Full range" if crossover is None else crossover.description,
        crossover_natural_fir_taps=0 if crossover is None else int(crossover.natural_fir_taps),
        alignment_fir=None if alignment_fir is None else alignment_fir.copy(),
        alignment_iir_filters=alignment_iir_filters,
        latency=latency,
        fir_synthesis=fir_synthesis,
        analysis_fft_size=effective_analysis_fft_size,
        original_impulse=(
            None
            if project is None or project.original_speaker_impulse is None
            else np.asarray(project.original_speaker_impulse, dtype=float)
        ),
        original_impulse_sample_rate=(
            None if project is None else project.original_speaker_impulse_sample_rate
        ),
    )


def response_complex_on_axis(response: SpeakerResponse | None, frequency: np.ndarray) -> np.ndarray:
    frequency = np.asarray(frequency, dtype=float)
    if response is None:
        return np.ones_like(frequency, dtype=complex)
    source_frequency, source_gain = finite_response_samples(
        response.frequency,
        response.gain_db,
        min_frequency=0.0,
    )
    if source_frequency.size < 2:
        return np.ones_like(frequency, dtype=complex)
    from response_completion import complete_response
    phase = None
    if response.phase_deg is not None:
        phase_frequency, source_phase = finite_unwrapped_phase_samples(
            response.frequency, response.phase_deg, min_frequency=0.0)
        if phase_frequency.size:
            phase = np.interp(source_frequency, phase_frequency, source_phase)
    gain, phase, _ = complete_response(
        source_frequency, source_gain, phase, frequency, phase_is_continuous=True)
    return 10.0 ** (gain / 20.0) * np.exp(1j * np.deg2rad(0.0 if phase is None else phase))


def _response_after_iir(
    response: SpeakerResponse,
    filters: tuple[IIRFilter, ...],
    sample_rate: int,
) -> SpeakerResponse:
    if not filters:
        return response
    frequency = np.asarray(response.frequency, dtype=float)
    original = response_complex_on_axis(response, frequency)
    transfer = iir_frequency_response(filters, int(sample_rate), frequency)
    combined = original * transfer
    gain_db = 20.0 * np.log10(np.maximum(np.abs(combined), 1e-12))
    phase_deg = np.rad2deg(np.unwrap(np.angle(combined)))
    return SpeakerResponse(
        frequency.tolist(),
        gain_db.tolist(),
        phase_deg.tolist(),
    )


def _complex_on_axis(source_frequency: np.ndarray, values: np.ndarray, frequency: np.ndarray) -> np.ndarray:
    source_frequency = np.asarray(source_frequency, dtype=float)
    values = np.asarray(values, dtype=complex)
    frequency = np.asarray(frequency, dtype=float)
    magnitude = np.abs(values)
    phase = np.unwrap(np.angle(values))
    interpolated_magnitude = np.interp(frequency, source_frequency, magnitude, left=magnitude[0], right=magnitude[-1])
    interpolated_phase = np.interp(frequency, source_frequency, phase, left=phase[0], right=phase[-1])
    return interpolated_magnitude * np.exp(1j * interpolated_phase)
