from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
import hashlib
import json
from typing import Any

from ..analysis_policy import normalize_analysis_fft_size
from ..project_runtime import ProjectRuntimeSnapshot
from .crossover_system import ensure_system_crossover_plan
from .models import DSPSystem
from .routing import common_input_iir_filters
from .output_assignment import ensure_output_channel_assignments
from .way_fir_synthesis import WAY_FIR_SYNTHESIS_ENGINE


DESIGN_SIGNATURE_SCHEMA_VERSION = 4
DSP_SYSTEM_DESIGN_ALGORITHM_VERSION = "dsp-system-output-way-matrix-v3"
FLAT_PROJECT_DESIGN_SIGNATURE = "flat-source-v1"


def project_design_signature(project: ProjectRuntimeSnapshot) -> str:
    """Hash only Project data that can change a DSP System result.

    Project name, database id, save timestamp and whole-payload revision are
    deliberately excluded.  Equivalent Input/IIR/FIR/Auto EQ settings and
    response data therefore keep the same signature after a metadata-only save.
    """

    config_payload = asdict(project.config.normalized())
    # DSP System owns these three design conditions and replaces the Project
    # values before generation.
    for device_owned_key in ("sample_rate", "taps", "analysis_fft_size"):
        config_payload.pop(device_owned_key, None)
    payload = {
        "schema": DESIGN_SIGNATURE_SCHEMA_VERSION,
        "config": config_payload,
        "fallback_raw_speaker_response": (
            _response_payload(project.raw_speaker_response)
            if project.config.speaker_response is None
            else None
        ),
        "raw_target_response": _response_payload(project.raw_target_response),
    }
    return _digest(payload)


def system_design_signature(
    system: DSPSystem,
    project_loader: Callable[[str], ProjectRuntimeSnapshot | None],
    *,
    analysis_fft_size: int = 16_384,
) -> str:
    """Return the Export signature for the current end-to-end DSP design.

    Kept under the original public name so saved systems and export callers stay
    compatible.  FIR regeneration and alignment now have narrower signatures.
    """

    fir_signatures = fir_design_signatures(
        system,
        project_loader,
        analysis_fft_size=analysis_fft_size,
    )
    alignment = alignment_design_signature(system, fir_signatures)
    return export_design_signature(
        system,
        fir_signatures,
        alignment,
        analysis_fft_size=analysis_fft_size,
    )


def fir_design_signatures(
    system: DSPSystem,
    project_loader: Callable[[str], ProjectRuntimeSnapshot | None],
    *,
    analysis_fft_size: int = 16_384,
) -> dict[str, str]:
    normalized = ensure_system_crossover_plan(
        ensure_output_channel_assignments(system)
    )
    signatures: dict[str, str] = {}
    common_analysis_fft_size = normalize_analysis_fft_size(analysis_fft_size)
    for way in normalized.ways:
        source_id = way.source_package_id
        project = project_loader(source_id) if source_id else None
        project_signature = (
            FLAT_PROJECT_DESIGN_SIGNATURE
            if project is None
            else project_design_signature(project)
        )
        device = normalized.device(way.dsp_id)
        target = way.output_target_override or normalized.targets.get(way.acoustic_group)
        crossover_payload: dict[str, Any] = {}
        if normalized.crossover_plan is not None:
            crossover_payload = {
                "overlap_convention": normalized.crossover_plan.overlap_convention,
                "boundaries": [
                    asdict(item)
                    for item in normalized.crossover_plan.boundaries
                    if way.id in {item.lower_way_id, item.upper_way_id}
                ],
                "sub_crossovers": [
                    asdict(item)
                    for item in normalized.crossover_plan.sub_crossovers
                    if way.id == item.way_id
                ],
            }
        signatures[way.id] = _digest({
            "schema": DESIGN_SIGNATURE_SCHEMA_VERSION,
            "algorithm": DSP_SYSTEM_DESIGN_ALGORITHM_VERSION,
            "way_fir_engine": WAY_FIR_SYNTHESIS_ENGINE,
            "source": {"id": source_id, "signature": project_signature},
            "device": {
                "sample_rate": device.sample_rate,
                "taps": device.taps,
                "fir_supported": device.fir_supported,
                "fir_off_behavior": "removed",
            },
            "application_analysis_policy": {
                "analysis_fft_size": common_analysis_fft_size,
            },
            "processing": asdict(normalized.processing),
            "input_iir_filters": [
                asdict(item) for item in common_input_iir_filters(normalized, way.id)
            ],
            "target": None if target is None else asdict(target),
            "way": {
                "linear_fir_filters": [asdict(item) for item in way.linear_fir_filters],
                "eq_mask_smoothing_oct": way.eq_mask_smoothing_oct,
                "manual_iir_filters": [asdict(item) for item in way.manual_iir_filters],
                "auto_iir_filters": [asdict(item) for item in way.auto_iir_filters],
                "correction_config": (
                    None if way.correction_config is None else asdict(way.correction_config)
                ),
                "fir_enabled": way.fir_enabled,
                "retained_fir_artifact_id": "",
            },
            "crossover": crossover_payload,
        })
    return signatures


def alignment_design_signature(system: DSPSystem, fir_signatures: Mapping[str, str]) -> str:
    normalized = ensure_system_crossover_plan(
        ensure_output_channel_assignments(system)
    )
    return _digest({
        "schema": DESIGN_SIGNATURE_SCHEMA_VERSION,
        "fir": dict(sorted(fir_signatures.items())),
        "devices": [
            {
                "id": item.id,
                "latency_offset_ms": item.latency_offset_ms,
                "delay_resolution_ms": item.delay_resolution_ms,
            }
            for item in normalized.devices
        ],
        "ways": [
            {
                "id": item.id,
                "alignment_iir_filters": [asdict(value) for value in item.alignment_iir_filters],
                "alignment_fir": list(item.alignment_fir),
                "alignment_source": item.alignment_source,
                "alignment_delay_ms": item.alignment_delay_ms,
                "alignment_polarity_invert": item.alignment_polarity_invert,
                "delay_ms": item.delay_ms,
            }
            for item in normalized.ways
        ],
    })


def export_design_signature(
    system: DSPSystem,
    fir_signatures: Mapping[str, str],
    alignment_signature: str,
    *,
    analysis_fft_size: int = 16_384,
) -> str:
    normalized = ensure_system_crossover_plan(
        ensure_output_channel_assignments(system)
    )
    return _digest({
        "schema": DESIGN_SIGNATURE_SCHEMA_VERSION,
        "alignment": alignment_signature,
        "fir": dict(sorted(fir_signatures.items())),
        "application_analysis_policy": {
            "analysis_fft_size": normalize_analysis_fft_size(analysis_fft_size),
        },
        "devices": [
            {
                "id": item.id,
                "target": item.target,
                "sample_rate": item.sample_rate,
                "max_input_channels": item.max_input_channels,
                "max_output_channels": item.max_output_channels,
                "fir_off_behavior": "removed",
            }
            for item in normalized.devices
        ],
        "inputs": [asdict(item) for item in normalized.inputs],
        "input_bindings": [asdict(item) for item in normalized.input_bindings],
        "routes": [asdict(item) for item in normalized.routes],
        "ways": [
            {
                "id": item.id,
                "output_id": item.output_id,
                "dsp_id": item.dsp_id,
                "output_channel": item.output_channel,
                "gain_db": item.gain_db,
                "polarity_invert": item.polarity_invert,
                "muted": item.muted,
                "routing_role": item.routing_role,
            }
            for item in normalized.ways
        ],
    })


def design_signature_matches(generated: str, expected: str) -> bool:
    return bool(generated) and bool(expected) and generated == expected


def short_design_signature(signature: str, length: int = 10) -> str:
    value = str(signature).strip()
    return value[: max(4, int(length))] if value else "Not generated"


def _response_payload(response: Any) -> Mapping[str, Any] | None:
    return None if response is None else asdict(response)


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
