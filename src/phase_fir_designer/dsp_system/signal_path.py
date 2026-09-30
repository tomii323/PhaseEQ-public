from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..export_pipeline import realized_filter_complex_response
from ..iir import biquad_rows, iir_frequency_response
from ..speaker import speaker_response_wavelet_impulse_source
from .output_pipeline import DSPSystemOutput, DSPWayOutput
from .regeneration import DSPWayResult


@dataclass(frozen=True)
class FinalSignalPathTrace:
    """Serializable fixed-stage trace for one physical Input-to-Output path."""

    input_id: str
    input_name: str
    dsp_id: str
    input_channel: int
    way_id: str
    way_name: str
    output_channel: int
    input_gain_db: float
    route_gain_db: float
    way_gain_db: float
    common_output_gain_db: float
    total_gain_db: float
    input_polarity_invert: bool
    way_polarity_invert: bool
    total_polarity_invert: bool
    way_delay_ms: float
    total_explicit_delay_ms: float
    input_iir_sections: int
    way_iir_sections: int
    fir_enabled: bool


@dataclass(frozen=True)
class SignalPathStageResponse:
    """Analysis-only response stages for one routed Input-to-Output path."""

    frequency: np.ndarray
    stages: dict[str, np.ndarray]
    analysis_complex_response: np.ndarray
    impulse_response: np.ndarray
    impulse_provenance: str
    impulse_note: str


def build_final_signal_path_traces(output: DSPSystemOutput) -> tuple[FinalSignalPathTrace, ...]:
    inputs = {item.id: item for item in output.inputs}
    ways = {item.way_id: item for item in output.ways}
    bindings = {
        (item.input_id, item.dsp_id): int(item.input_channel)
        for item in output.input_bindings
    }
    traces: list[FinalSignalPathTrace] = []
    for route in output.routes:
        way = ways.get(route.way_id)
        source = inputs.get(route.input_id)
        if way is None or source is None or not route.enabled or route.muted or source.muted:
            continue
        channel = bindings.get((source.id, way.dsp_id))
        if channel is None:
            continue
        input_gain = float(source.gain_db)
        route_gain = float(route.gain_db)
        way_gain = float(way.way_gain_db)
        common_gain = float(output.common_auto_gain_db)
        input_invert = bool(source.polarity_invert)
        way_invert = bool(way.polarity_invert)
        traces.append(FinalSignalPathTrace(
            input_id=source.id,
            input_name=source.name,
            dsp_id=way.dsp_id,
            input_channel=channel,
            way_id=way.way_id,
            way_name=way.way_name,
            output_channel=int(way.output_channel),
            input_gain_db=input_gain,
            route_gain_db=route_gain,
            way_gain_db=way_gain,
            common_output_gain_db=common_gain,
            total_gain_db=input_gain + route_gain + way_gain + common_gain,
            input_polarity_invert=input_invert,
            way_polarity_invert=way_invert,
            total_polarity_invert=input_invert ^ way_invert,
            way_delay_ms=float(way.delay_ms),
            total_explicit_delay_ms=float(way.delay_ms),
            input_iir_sections=len(biquad_rows(
                tuple(item for item in source.iir_filters if item.enabled),
                int(way.sample_rate),
            )),
            way_iir_sections=len(biquad_rows(
                tuple(item for item in way.iir_filters if item.enabled),
                int(way.sample_rate),
            )),
            fir_enabled=bool(way.fir_enabled),
        ))
    return tuple(traces)


def final_way_complex_response(
    result: DSPWayResult,
    output: DSPSystemOutput,
    frequency: np.ndarray,
    *,
    input_id: str | None = None,
    include_device_latency: bool = True,
) -> np.ndarray:
    """Evaluate exactly the stages represented by the DSP export package.

    Signal order is Input trim/IIR -> Route -> Way IIR -> FIR -> Way output
    trim/polarity/delay -> Common output gain.  Linear stages commute for the
    frequency-domain calculation, but adapters emit them in this order.
    """

    frequency = np.asarray(frequency, dtype=float)
    way = next((item for item in output.ways if item.way_id == result.way.id), None)
    if way is None:
        return np.zeros_like(frequency, dtype=complex)
    route_sum = _route_sum_transfer(output, way, frequency, input_id=input_id)
    if not np.any(np.abs(route_sum) > 0.0):
        return np.zeros_like(frequency, dtype=complex)
    source = _project_complex(
        np.asarray(result.frequency, dtype=float),
        np.asarray(result.original_complex_response, dtype=complex),
        frequency,
        max_frequency=float(result.device.sample_rate) / 2.0,
    )
    way_iir = iir_frequency_response(
        tuple(item for item in way.iir_filters if item.enabled),
        int(way.sample_rate),
        frequency,
    )
    if way.fir_enabled:
        fir_frequency, fir_values = realized_filter_complex_response(
            np.asarray(way.coefficients, dtype=float),
            int(way.sample_rate),
            max(2, (len(frequency) - 1) * 2),
        )
        fir = _project_complex(
            fir_frequency,
            fir_values,
            frequency,
            max_frequency=float(way.sample_rate) / 2.0,
        )
    else:
        fir = np.ones_like(frequency, dtype=complex)
    gain = 10.0 ** (
        (float(way.way_gain_db) + float(output.common_auto_gain_db)) / 20.0
    )
    polarity = -1.0 if way.polarity_invert else 1.0
    delay_ms = float(way.delay_ms)
    if include_device_latency:
        delay_ms += float(result.device.latency_offset_ms)
    delay = np.exp(-1j * 2.0 * np.pi * frequency * delay_ms / 1000.0)
    return source * route_sum * way_iir * fir * gain * polarity * delay


def signal_path_stage_response(
    result: DSPWayResult,
    output: DSPSystemOutput,
    frequency: np.ndarray,
    *,
    input_id: str,
    include_device_latency: bool = True,
) -> SignalPathStageResponse:
    """Evaluate named stages without changing the export or DSP data model."""

    frequency = np.asarray(frequency, dtype=float)
    way = next((item for item in output.ways if item.way_id == result.way.id), None)
    source_input = next((item for item in output.inputs if item.id == input_id), None)
    if way is None or source_input is None:
        raise ValueError("selected signal path is no longer available")
    if (source_input.id, way.dsp_id) not in {
        (item.input_id, item.dsp_id) for item in output.input_bindings
    }:
        raise ValueError("selected input is not bound to the output DSP")

    source = _project_complex(
        np.asarray(result.frequency, dtype=float),
        np.asarray(result.original_complex_response, dtype=complex),
        frequency,
        max_frequency=float(result.device.sample_rate) / 2.0,
    )
    input_iir = iir_frequency_response(
        tuple(item for item in source_input.iir_filters if item.enabled),
        int(way.sample_rate),
        frequency,
    )
    input_trim = 10.0 ** (float(source_input.gain_db) / 20.0)
    input_polarity = -1.0 if source_input.polarity_invert else 1.0
    after_input_iir = source * input_iir * input_trim * input_polarity

    route_transfer = np.zeros_like(frequency, dtype=complex)
    for route in output.routes:
        if (
            route.input_id == input_id
            and route.way_id == way.way_id
            and route.enabled
            and not route.muted
        ):
            route_transfer += 10.0 ** (float(route.gain_db) / 20.0)
    if not np.any(np.abs(route_transfer) > 0.0):
        raise ValueError("selected signal path is muted or not routed")
    after_routing = after_input_iir * route_transfer

    output_iir = iir_frequency_response(
        tuple(item for item in result.transient_config.iir_filters if item.enabled),
        int(way.sample_rate),
        frequency,
    )
    after_output_iir = after_routing * output_iir
    crossover_alignment_iir = iir_frequency_response(
        tuple(item for item in (*result.crossover_iir_filters, *result.alignment_iir_filters) if item.enabled),
        int(way.sample_rate),
        frequency,
    )
    after_crossover_alignment = after_output_iir * crossover_alignment_iir

    if way.fir_enabled:
        fir_frequency, fir_values = realized_filter_complex_response(
            np.asarray(way.coefficients, dtype=float),
            int(way.sample_rate),
            max(2, (len(frequency) - 1) * 2),
        )
        fir = _project_complex(
            fir_frequency,
            fir_values,
            frequency,
            max_frequency=float(way.sample_rate) / 2.0,
        )
    else:
        fir = np.ones_like(frequency, dtype=complex)
    after_fir = after_crossover_alignment * fir

    final_gain = 10.0 ** (
        (float(way.way_gain_db) + float(output.common_auto_gain_db)) / 20.0
    )
    final_polarity = -1.0 if way.polarity_invert else 1.0
    delay_ms = float(way.delay_ms)
    if include_device_latency:
        delay_ms += float(result.device.latency_offset_ms)
    final_delay = np.exp(-1j * 2.0 * np.pi * frequency * delay_ms / 1000.0)
    final_output = after_fir * final_gain * final_polarity * final_delay

    provenance = "Reconstructed IR"
    note = "Gain＋Phaseの複素応答から解析用IRを再構成しています。"
    analysis_final = final_output
    if result.original_impulse is not None and result.original_impulse_sample_rate:
        provenance = "Original IR"
        note = "保存された実IRへ信号経路を適用しています。"
        original_frequency = np.fft.rfftfreq(
            len(result.original_impulse),
            d=1.0 / float(result.original_impulse_sample_rate),
        )
        original_complex = np.fft.rfft(result.original_impulse)
        measured_source = _project_complex(
            original_frequency,
            original_complex,
            frequency,
            max_frequency=min(
                float(result.original_impulse_sample_rate) / 2.0,
                float(way.sample_rate) / 2.0,
            ),
        )
        downstream = np.divide(
            final_output,
            source,
            out=np.zeros_like(final_output),
            where=np.abs(source) > 1e-12,
        )
        analysis_final = measured_source * downstream
    elif result.project_revision == "flat-source-v1":
        provenance = "Synthetic flat impulse"
        note = "Flat Responseの単位インパルスへ信号経路を適用しています。"
    elif result.original_response is not None and result.original_response.phase_deg is None:
        provenance = "Estimated IR · Minimum phase"
        note = "Gainのみから最小位相IRを推定しています。絶対時間と反射の評価には使用できません。"
        reconstructed = speaker_response_wavelet_impulse_source(
            result.original_response,
            int(way.sample_rate),
        )
        if reconstructed is not None:
            reconstructed_frequency = np.fft.rfftfreq(
                len(reconstructed.impulse),
                d=1.0 / float(way.sample_rate),
            )
            reconstructed_complex = np.fft.rfft(reconstructed.impulse)
            estimated_source = _project_complex(
                reconstructed_frequency,
                reconstructed_complex,
                frequency,
                max_frequency=float(way.sample_rate) / 2.0,
            )
            downstream = np.divide(
                final_output,
                source,
                out=np.zeros_like(final_output),
                where=np.abs(source) > 1e-12,
            )
            analysis_final = estimated_source * downstream
    fft_size = max(2, (len(frequency) - 1) * 2)
    impulse = np.fft.irfft(analysis_final, n=fft_size)
    return SignalPathStageResponse(
        frequency=frequency,
        stages={
            "Input Response": source,
            "After Input IIR": after_input_iir,
            "After Routing": after_routing,
            "After Output IIR": after_output_iir,
            "After Crossover / Alignment": after_crossover_alignment,
            "After FIR": after_fir,
            "Final Output": final_output,
        },
        analysis_complex_response=analysis_final,
        impulse_response=impulse,
        impulse_provenance=provenance,
        impulse_note=note,
    )


def _route_sum_transfer(
    output: DSPSystemOutput,
    way: DSPWayOutput,
    frequency: np.ndarray,
    *,
    input_id: str | None,
) -> np.ndarray:
    inputs = {item.id: item for item in output.inputs}
    bindings = {(item.input_id, item.dsp_id) for item in output.input_bindings}
    combined = np.zeros_like(frequency, dtype=complex)
    for route in output.routes:
        if route.way_id != way.way_id or (input_id is not None and route.input_id != input_id):
            continue
        source = inputs.get(route.input_id)
        if (
            source is None
            or source.muted
            or route.muted
            or not route.enabled
            or (source.id, way.dsp_id) not in bindings
        ):
            continue
        gain = 10.0 ** ((float(source.gain_db) + float(route.gain_db)) / 20.0)
        polarity = -1.0 if source.polarity_invert else 1.0
        transfer = np.full_like(frequency, gain * polarity, dtype=complex)
        transfer = transfer * iir_frequency_response(
            tuple(item for item in source.iir_filters if item.enabled),
            int(way.sample_rate),
            frequency,
        )
        combined += transfer
    return combined


def _project_complex(
    source_frequency: np.ndarray,
    values: np.ndarray,
    frequency: np.ndarray,
    *,
    max_frequency: float,
) -> np.ndarray:
    source_frequency = np.asarray(source_frequency, dtype=float)
    values = np.asarray(values, dtype=complex)
    frequency = np.asarray(frequency, dtype=float)
    valid = (
        np.isfinite(source_frequency)
        & np.isfinite(values.real)
        & np.isfinite(values.imag)
    )
    if np.count_nonzero(valid) < 2:
        return np.zeros_like(frequency, dtype=complex)
    source_frequency = source_frequency[valid]
    values = values[valid]
    magnitude = np.abs(values)
    phase = np.unwrap(np.angle(values))
    supported = frequency <= float(max_frequency)
    projected = np.zeros_like(frequency, dtype=complex)
    projected_magnitude = np.interp(
        frequency[supported], source_frequency, magnitude,
        left=magnitude[0], right=magnitude[-1],
    )
    projected_phase = np.interp(
        frequency[supported], source_frequency, phase,
        left=phase[0], right=phase[-1],
    )
    projected[supported] = projected_magnitude * np.exp(1j * projected_phase)
    return projected
