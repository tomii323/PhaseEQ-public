from __future__ import annotations

from dataclasses import dataclass, replace
import uuid
from typing import Callable

import numpy as np

from ..config import SpeakerResponse
from ..input_pipeline import InputProcessingSettings, InputSource, process_input, shift_response_gain
from ..project_runtime import ProjectRuntimeSnapshot
from ..speaker import smooth_speaker_response
from .models import DSPDevice, DSPSystem, DSPSystemProcessing, DSPSystemTarget, DSPWay
from .routing import common_input_iir_filters, initialize_default_routing, unconnected_way_ids
from .output_pipeline import DSPSystemOutput, prepare_system_output
from .regeneration import DSPWayResult, regenerate_way
from .topology import resize_system_topology
from .crossover_system import (
    crossover_mask_filters,
    ensure_system_crossover_plan,
    prepare_system_crossovers,
)
from .crossover_phase_alignment import (
    PhaseAlignmentRecommendation,
)
from .design_signature import (
    alignment_design_signature,
    fir_design_signatures,
    system_design_signature,
)
from .delay_guide import fixed_delay_guides
from .output_assignment import ensure_output_channel_assignments


@dataclass(frozen=True)
class DSPSystemBuild:
    system: DSPSystem
    way_results: tuple[DSPWayResult, ...]
    sums: dict[str, object]
    output: DSPSystemOutput
    warnings: tuple[str, ...]
    stale_way_ids: tuple[str, ...]
    design_signature: str = ""
    fir_signatures: dict[str, str] | None = None
    alignment_signature: str = ""
    target_responses: dict[str, SpeakerResponse] | None = None


def system_template(name: str, mode: str = "3Way") -> DSPSystem:
    main_way_count = next((count for count in (1, 2, 3, 4) if mode.startswith(str(count))), 3)
    sub_enabled = "+ Sub" in mode
    devices: tuple[DSPDevice, ...]
    if mode == "Upper 96k / Lower 48k":
        devices = (
            DSPDevice(
                "dsp-upper", "DSP-Upper", "miniDSP",
                sample_rate=96_000, taps=4097,
            ),
            DSPDevice(
                "dsp-lower", "DSP-Lower", "miniDSP",
                sample_rate=48_000, taps=4097,
            ),
        )
        main_way_count = 4
        sub_enabled = False
    else:
        devices = (DSPDevice("dsp-1", "DSP-1", "miniDSP"),)
    system = DSPSystem(
        id=str(uuid.uuid4()), name=name, devices=devices,
        targets={"Main": DSPSystemTarget()},
    )
    return ensure_output_channel_assignments(
        initialize_default_routing(resize_system_topology(system, main_way_count, sub_enabled))
    )


def build_dsp_system(
    system: DSPSystem,
    project_loader: Callable[[str], ProjectRuntimeSnapshot | None],
    *,
    fir_artifact_loader: Callable[[str], np.ndarray | None] | None = None,
    analysis_fft_size: int = 16_384,
) -> DSPSystemBuild:
    system = ensure_system_crossover_plan(
        ensure_output_channel_assignments(system)
    )
    system.validate()
    expected_fir_signatures = fir_design_signatures(
        system,
        project_loader,
        analysis_fft_size=analysis_fft_size,
    )
    alignment_signature = alignment_design_signature(system, expected_fir_signatures)
    design_signature = system_design_signature(
        system,
        project_loader,
        analysis_fft_size=analysis_fft_size,
    )
    warnings: list[str] = []
    stale: list[str] = []
    results: list[DSPWayResult] = []
    target_responses: dict[str, SpeakerResponse] = {}
    crossover = prepare_system_crossovers(system)
    warnings.extend(item.message for item in crossover.issues)
    for way in system.ways:
        source_id = way.source_package_id
        project = project_loader(source_id) if source_id else None
        if source_id and project is None:
            warnings.append(
                f"{way.name}: 選択中のSpeaker Packageが見つからないため、Flat sourceで生成しました。"
            )
        if project is not None:
            source_input = project.processed_speaker_response or project.raw_speaker_response
            if source_input is None:
                warnings.append(f"{way.name}: Speaker PackageにINPUT処理後応答がありません。")
                continue
            # Speaker Package INPUT is immutable here. DSP System processing is
            # applied only after that stored result.
            processed = process_system_input(
                source_input,
                system.processing,
                system.device(way.dsp_id).sample_rate,
            )
            target = system_target_response(
                way.output_target_override or system.targets.get(way.acoustic_group),
                project,
                processed,
            )
        else:
            device = system.device(way.dsp_id)
            flat = SpeakerResponse(
                [0.0, float(device.sample_rate) / 4.0, float(device.sample_rate) / 2.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            )
            processed = process_system_input(flat, system.processing, device.sample_rate)
            target = system_target_response(
                way.output_target_override or system.targets.get(way.acoustic_group),
                None,
                processed,
            )
        if target is not None:
            previous_target = target_responses.get(way.acoustic_group)
            if previous_target is None or len(target.frequency) > len(previous_target.frequency):
                target_responses[way.acoustic_group] = target
        try:
            result = regenerate_way(
                project,
                system.device(way.dsp_id),
                way,
                speaker_response=processed,
                target_response=target,
                crossover=crossover.ways[way.id],
                crossover_mask=crossover_mask_filters(crossover.plan, way.id),
                retained_fir=None,
                input_iir_filters=common_input_iir_filters(system, way.id),
                fir_master_enabled=bool(system.processing.fir_enabled),
                analysis_fft_size=analysis_fft_size,
            )
            results.append(result)
            saved_way_signature = system.way_design_signatures.get(way.id, "")
            expected_way_signature = expected_fir_signatures.get(way.id, "")
            if saved_way_signature and saved_way_signature != expected_way_signature:
                stale.append(way.id)
            device = system.device(way.dsp_id)
            if not system.processing.fir_enabled:
                warnings.append(f"{way.name}: System FIRがOFFのためIIR Only／FIR遅延0 msで評価しています。")
            elif not system.device(way.dsp_id).fir_supported:
                warnings.append(f"{way.name}: DeviceがWay FIR非対応のためIIR Onlyで評価しています。")
            elif not way.fir_enabled:
                warnings.append(f"{way.name}: Way FIRがOFFのためIIR Onlyで評価しています。")
            if result.fir_synthesis.bypassed and result.crossover_fir is not None:
                warnings.append(
                    f"{way.name}: FIR帯域分割もBypassされています。実運用にはIIR crossoverを選択してください。"
                )
        except ValueError as exc:
            warnings.append(f"{way.name}: {exc}")
    if not results:
        detail = " / ".join(warnings)
        message = "再生成可能なWayがありません。接続Speaker Packageの入力データを確認してください。"
        raise ValueError(f"{message} {detail}" if detail else message)
    output = prepare_system_output(
        results, auto_gain=system.processing.auto_gain,
        remove_nyquist=system.processing.remove_nyquist,
        remove_nyquist_strength=system.processing.remove_nyquist_strength,
        inputs=system.inputs,
        input_bindings=system.input_bindings,
        routes=system.routes,
        design_signature=design_signature,
        fir_signatures=expected_fir_signatures,
        alignment_signature=alignment_signature,
    )
    sums: dict[str, object] = {}
    if system.processing.auto_gain and any(result.fir_synthesis.bypassed for result in results):
        warnings.append("FIR OFF／非対応Wayがあるため、Common FIR Auto Gainは適用していません。")
    if len({result.device.sample_rate for result in results}) > 1:
        warnings.append("異なるSample RateのDSPを合成しています。実機間のClock／Latency同期を確認してください。")
    if output.output_overall_peak_db > 0.0:
        warnings.append(f"出力Peakが0 dBFSを超えています: {output.output_overall_peak_db:+.2f} dB")
    disconnected = set(unconnected_way_ids(system))
    for way in system.ways:
        if way.id in disconnected:
            warnings.append(f"{way.name}: Routingが未接続です。")
    return DSPSystemBuild(
        system=system,
        way_results=tuple(results),
        sums=sums,
        output=output,
        warnings=tuple(warnings),
        stale_way_ids=tuple(stale),
        design_signature=design_signature,
        fir_signatures=expected_fir_signatures,
        alignment_signature=alignment_signature,
        target_responses=target_responses,
    )


def process_system_input(response: SpeakerResponse, processing: DSPSystemProcessing, sample_rate: int) -> SpeakerResponse:
    result = process_input(
        InputSource(response=response),
        InputProcessingSettings(
            sample_rate=sample_rate, phase_center_enabled=False, band_extension_enabled=False,
            smoothing_fraction=processing.smoothing_fraction,
            smoothing_mode=processing.smoothing_mode, gain_shift_mode="Manual",
            manual_gain_shift_db=processing.gain_shift_db,
        ),
    )
    assert result.processed_response is not None
    return result.processed_response


def system_target_response(
    target: DSPSystemTarget | None,
    project: ProjectRuntimeSnapshot | None,
    speaker: SpeakerResponse,
) -> SpeakerResponse | None:
    if target is None or target.mode == "Project":
        base = (
            None
            if project is None
            else project.raw_target_response or project.config.target_response
        )
    elif target.mode == "Custom":
        base = target.response
    else:
        base = SpeakerResponse(list(speaker.frequency), np.zeros(len(speaker.frequency)).tolist(), None)
    if base is None:
        return None
    smoothed = smooth_speaker_response(base, target.smoothing_fraction, mode="gain_phase")
    return shift_response_gain(smoothed, target.gain_shift_db)


def auto_level(system: DSPSystem, build: DSPSystemBuild, group: str) -> DSPSystem:
    selected = [result for result in build.way_results if result.way.acoustic_group == group]
    if not selected:
        return system
    peaks = {result.way.id: float(np.nanmax(20.0 * np.log10(np.maximum(np.abs(result.acoustic_response), 1e-12)))) for result in selected}
    reference = min(peaks.values())
    ways = tuple(replace(way, gain_db=way.gain_db + reference - peaks.get(way.id, reference)) for way in system.ways)
    return replace(system, ways=ways, revision=system.revision + 1)


def auto_delay(system: DSPSystem, build: DSPSystemBuild, group: str) -> DSPSystem:
    """Apply deterministic fixed-delay guidance without acoustic optimisation."""

    selected = [result for result in build.way_results if result.way.acoustic_group == group]
    if not selected:
        return system
    guides = {item.way_id: item for item in fixed_delay_guides(selected)}
    ways = tuple(
        replace(
            way,
            delay_ms=(
                round(guides[way.id].suggested_total_ms / system.device(way.dsp_id).delay_resolution_ms)
                * system.device(way.dsp_id).delay_resolution_ms
            ),
        )
        if way.id in guides and guides[way.id].state in {"eligible", "reference"}
        else way
        for way in system.ways
    )
    return replace(system, ways=ways, revision=system.revision + 1)






def apply_phase_alignment_recommendation(
    system: DSPSystem,
    boundary_id: str,
    recommendation: PhaseAlignmentRecommendation,
) -> DSPSystem:
    plan = system.crossover_plan
    if plan is None:
        raise ValueError("DSP System does not contain a crossover plan")
    boundary = next((item for item in plan.boundaries if item.id == boundary_id), None)
    if boundary is None:
        raise ValueError(f"unknown crossover boundary: {boundary_id}")
    if recommendation.method == "none" or recommendation.compensate == "none":
        return system
    way_id = (
        boundary.lower_way_id
        if recommendation.compensate == "lower"
        else boundary.upper_way_id
    )
    ways: list[DSPWay] = []
    for way in system.ways:
        if way.id != way_id:
            ways.append(way)
            continue
        replacing_same_boundary = way.alignment_source == boundary_id
        base_delay_ms = (
            float(way.delay_ms) - float(way.alignment_delay_ms)
            if replacing_same_boundary else float(way.delay_ms)
        )
        base_polarity = (
            bool(way.polarity_invert) ^ bool(way.alignment_polarity_invert)
            if replacing_same_boundary else bool(way.polarity_invert)
        )
        alignment_fir = () if replacing_same_boundary else tuple(way.alignment_fir)
        if recommendation.method == "fir_phase" and recommendation.fir_correction:
            correction = np.asarray(recommendation.fir_correction, dtype=float)
            if alignment_fir:
                correction = np.convolve(np.asarray(alignment_fir, dtype=float), correction)
            alignment_fir = tuple(float(value) for value in correction)
        filters = () if replacing_same_boundary else tuple(way.alignment_iir_filters)
        if recommendation.method == "iir_allpass":
            filters += tuple(recommendation.allpass_filters)
        ways.append(replace(
            way,
            delay_ms=base_delay_ms + float(recommendation.delay_ms),
            polarity_invert=base_polarity ^ bool(recommendation.polarity_invert),
            alignment_iir_filters=filters,
            alignment_fir=alignment_fir,
            alignment_source=boundary_id,
            alignment_delay_ms=(
                0.0 if replacing_same_boundary else float(way.alignment_delay_ms)
            ) + float(recommendation.delay_ms),
            alignment_polarity_invert=(
                False if replacing_same_boundary else bool(way.alignment_polarity_invert)
            ) ^ bool(recommendation.polarity_invert),
        ))
    return replace(system, ways=tuple(ways), revision=system.revision + 1)


def clear_phase_alignment(system: DSPSystem, way_id: str | None = None) -> DSPSystem:
    ways = tuple(
        replace(
            way,
            delay_ms=float(way.delay_ms) - float(way.alignment_delay_ms),
            polarity_invert=bool(way.polarity_invert) ^ bool(way.alignment_polarity_invert),
            alignment_iir_filters=(),
            alignment_fir=(),
            alignment_source="",
            alignment_delay_ms=0.0,
            alignment_polarity_invert=False,
        )
        if way_id is None or way.id == way_id
        else way
        for way in system.ways
    )
    return replace(system, ways=ways, revision=system.revision + 1)




def stamp_project_revisions(
    system: DSPSystem,
    project_loader: Callable[[str], ProjectRuntimeSnapshot | None],
) -> DSPSystem:
    revisions = dict(system.project_revisions)
    for way in system.ways:
        source_id = way.source_package_id
        project = project_loader(source_id) if source_id else None
        if project is not None:
            revisions[source_id] = project.revision
    return replace(system, project_revisions=revisions)


def stamp_design_signature(
    system: DSPSystem,
    build: DSPSystemBuild,
    project_loader: Callable[[str], ProjectRuntimeSnapshot | None],
) -> DSPSystem:
    """Record the generated signature without changing the design request."""

    stamped = stamp_project_revisions(system, project_loader)
    return replace(
        stamped,
        last_design_signature=build.design_signature,
        last_alignment_signature=build.alignment_signature,
        applied_design_signature=build.design_signature,
        live_result_signature=build.design_signature,
        way_design_signatures={
            result.way.id: (build.fir_signatures or {}).get(
                result.way.id, result.fir_synthesis.design_signature
            )
            for result in build.way_results
        },
    )
