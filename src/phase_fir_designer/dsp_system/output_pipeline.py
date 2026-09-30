from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Mapping

import numpy as np

from ..config import IIRFilter
from ..fir_auto_gain import apply_fixed_multiway_fir_auto_gain, measure_fir_peak_db
from ..fir_postprocess import remove_nyquist_component
from .regeneration import DSPWayResult
from .models import DSPInput, DSPInputBinding, DSPRoute


@dataclass(frozen=True)
class DSPWayOutput:
    way_id: str
    way_name: str
    dsp_id: str
    device_name: str
    target: str
    output_channel: int
    sample_rate: int
    coefficients: np.ndarray
    peak_db: float
    delay_ms: float
    project_revision: str
    way_gain_db: float = 0.0
    polarity_invert: bool = False
    max_output_channels: int = 8
    max_input_channels: int = 2
    iir_filters: tuple[IIRFilter, ...] = ()
    fir_enabled: bool = True
    design_signature: str = ""
    composite_group: str = "Main"
    composite_channel_name: str = ""


@dataclass(frozen=True)
class DSPSystemOutput:
    ways: tuple[DSPWayOutput, ...]
    common_auto_gain_db: float
    original_overall_peak_db: float
    output_overall_peak_db: float
    revision: str
    inputs: tuple[DSPInput, ...] = ()
    input_bindings: tuple[DSPInputBinding, ...] = ()
    routes: tuple[DSPRoute, ...] = ()
    design_signature: str = ""
    fir_signatures: dict[str, str] | None = None
    alignment_signature: str = ""


def prepare_system_output(
    way_results: list[DSPWayResult] | tuple[DSPWayResult, ...],
    *,
    auto_gain: bool = True,
    remove_nyquist: bool = False,
    remove_nyquist_strength: float = 1.0,
    inputs: tuple[DSPInput, ...] = (),
    input_bindings: tuple[DSPInputBinding, ...] = (),
    routes: tuple[DSPRoute, ...] = (),
    design_signature: str = "",
    fir_signatures: Mapping[str, str] | None = None,
    alignment_signature: str = "",
) -> DSPSystemOutput:
    active = [result for result in way_results if not result.way.muted]
    if not active:
        raise ValueError("at least one active DSP way is required")
    prepared: list[np.ndarray] = []
    level_adjusted: list[np.ndarray] = []
    for result in active:
        coefficients = np.asarray(result.fir, dtype=float).copy()
        if remove_nyquist and float(remove_nyquist_strength) > 0.0:
            coefficients, _removed = remove_nyquist_component(
                coefficients,
                strength=float(remove_nyquist_strength),
            )
        prepared.append(coefficients)
        # Output trim and polarity are explicit DSP stages.  They must remain
        # available when FIR is OFF and must never change the FIR artifact.
        level_adjusted.append(
            coefficients * 10.0 ** (float(result.way.gain_db) / 20.0)
        )

    original_peaks = tuple(float(measure_fir_peak_db(item)[0]) for item in level_adjusted)
    all_fir_enabled = all(not result.fir_synthesis.bypassed for result in active)
    if auto_gain and all_fir_enabled:
        common = apply_fixed_multiway_fir_auto_gain(level_adjusted)
        # Only the common scalar is retained.  FIR coefficients stay pure;
        # adapters implement the common output gain as an explicit stage.
        coefficients_list = prepared
        applied_gain_db = float(common.applied_gain_db)
        original_overall_peak_db = float(common.original_overall_peak_db)
        output_overall_peak_db = float(common.output_overall_peak_db)
    else:
        coefficients_list = prepared
        applied_gain_db = 0.0
        original_overall_peak_db = max(original_peaks)
        output_overall_peak_db = original_overall_peak_db

    outputs: list[DSPWayOutput] = []
    revision_hash = hashlib.sha256()
    revision_hash.update(str(design_signature).encode("ascii"))
    for result, coefficients in zip(active, coefficients_list, strict=True):
        values = np.asarray(coefficients, dtype=float)
        revision_hash.update(result.way.id.encode("utf-8"))
        revision_hash.update(result.project_revision.encode("ascii"))
        revision_hash.update(int(result.device.max_output_channels).to_bytes(2, "little"))
        revision_hash.update(int(result.device.max_input_channels).to_bytes(2, "little"))
        revision_hash.update(values.astype("<f8", copy=False).tobytes())
        outputs.append(
            DSPWayOutput(
                way_id=result.way.id,
                way_name=result.way.name,
                dsp_id=result.device.id,
                device_name=result.device.name,
                target=result.device.target,
                output_channel=int(result.way.output_channel),
                sample_rate=int(result.device.sample_rate),
                coefficients=values,
                peak_db=float(measure_fir_peak_db(values)[0]),
                delay_ms=float(result.way.delay_ms),
                project_revision=result.project_revision,
                way_gain_db=float(result.way.gain_db),
                polarity_invert=bool(result.way.polarity_invert),
                max_output_channels=int(result.device.max_output_channels),
                max_input_channels=int(result.device.max_input_channels),
                iir_filters=(
                    tuple(item for item in result.transient_config.iir_filters if item.enabled)
                    + tuple(item for item in result.crossover_iir_filters if item.enabled)
                    + tuple(item for item in result.alignment_iir_filters if item.enabled)
                ),
                fir_enabled=not result.fir_synthesis.bypassed,
                design_signature=(fir_signatures or {}).get(
                    result.way.id, result.fir_synthesis.design_signature,
                ),
                composite_group=result.way.acoustic_group,
                composite_channel_name=(
                    f"{result.way.routing_role} {result.way.name}"
                    if result.way.routing_role not in {"", "Custom", "Other"}
                    else result.way.name
                ),
            )
        )
    return DSPSystemOutput(
        ways=tuple(outputs),
        common_auto_gain_db=applied_gain_db,
        original_overall_peak_db=original_overall_peak_db,
        output_overall_peak_db=output_overall_peak_db,
        revision=revision_hash.hexdigest(),
        inputs=inputs,
        input_bindings=input_bindings,
        routes=routes,
        design_signature=str(design_signature),
        fir_signatures=dict(fir_signatures or {}),
        alignment_signature=str(alignment_signature),
    )
