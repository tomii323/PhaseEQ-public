from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from fir_output_window import apply_output_window

from .fir_auto_gain import (
    FFT_OVERSAMPLING,
    TARGET_FIR_PEAK_DB,
    FIRAutoGainResult,
    apply_fixed_fir_auto_gain,
    measure_fir_peak_db,
)
from .fir_postprocess import remove_nyquist_component
from .phase_curves import frequency_axis, phase_error_deg, wrapped_phase_deg


@dataclass(frozen=True)
class ExportFIRPostprocessResult:
    fir: np.ndarray
    removed_nyquist_amplitude: float
    auto_gain: FIRAutoGainResult
    auto_gain_enabled: bool


def _bypassed_fir_auto_gain(fir: np.ndarray) -> FIRAutoGainResult:
    peak_db, fft_size = measure_fir_peak_db(fir, oversampling=FFT_OVERSAMPLING)
    return FIRAutoGainResult(
        coefficients=np.asarray(fir, dtype=float).copy(),
        original_peak_db=float(peak_db),
        output_peak_db=float(peak_db),
        target_peak_db=TARGET_FIR_PEAK_DB,
        applied_gain_db=0.0,
        linear_gain=1.0,
        fft_size=int(fft_size),
        oversampling=FFT_OVERSAMPLING,
        changed=False,
    )


def realized_filter_complex_response(
    fir: np.ndarray,
    sample_rate: int,
    fft_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    frequency = frequency_axis(sample_rate, fft_size)
    response = np.fft.rfft(np.asarray(fir, dtype=float), n=fft_size)
    delay_samples = (len(fir) - 1) / 2
    response = response * np.exp(1j * 2.0 * np.pi * frequency * delay_samples / sample_rate)
    return frequency, response


def result_for_export_fir_stage(
    result: Any,
    sample_rate: int,
    output_fir_polarity_invert: bool,
    output_fir_remove_nyquist_enabled: bool,
    output_fir_remove_nyquist_strength: float,
    fir_auto_gain_enabled: bool = True,
    output_fir_cosine_taper_enabled: bool = False,
) -> Any:
    export_result = output_fir_for_export_report(
        result.fir,
        output_fir_polarity_invert=output_fir_polarity_invert,
        output_fir_remove_nyquist_enabled=output_fir_remove_nyquist_enabled,
        output_fir_remove_nyquist_strength=output_fir_remove_nyquist_strength,
        fir_auto_gain_enabled=fir_auto_gain_enabled,
        output_fir_cosine_taper_enabled=output_fir_cosine_taper_enabled,
    )
    output_fir = export_result.fir

    analysis_fft_size = max((len(result.frequency) - 1) * 2, len(output_fir))
    _, realized_complex = realized_filter_complex_response(
        output_fir,
        sample_rate,
        analysis_fft_size,
    )
    realized_gain_db = 20.0 * np.log10(np.maximum(np.abs(realized_complex), 1e-12))
    realized_phase_unwrapped_deg = np.rad2deg(np.unwrap(np.angle(realized_complex)))
    realized_phase_wrapped_deg = wrapped_phase_deg(realized_phase_unwrapped_deg)
    gain_error_db = realized_gain_db - result.target_gain_db

    return replace(
        result,
        fir=output_fir,
        realized_gain_db=realized_gain_db,
        gain_error_db=gain_error_db,
        realized_phase_wrapped_deg=realized_phase_wrapped_deg,
        realized_phase_unwrapped_deg=realized_phase_unwrapped_deg,
        phase_error_deg=phase_error_deg(
            result.target_phase_unwrapped_deg,
            realized_phase_unwrapped_deg,
            np.minimum(realized_gain_db, result.target_gain_db),
        ),
        impulse_response=output_fir.copy(),
        step_response=np.cumsum(output_fir),
    )


def output_fir_for_export(
    fir: np.ndarray,
    *,
    output_fir_polarity_invert: bool,
    output_fir_remove_nyquist_enabled: bool,
    output_fir_remove_nyquist_strength: float,
    fir_auto_gain_enabled: bool = True,
    output_fir_cosine_taper_enabled: bool = False,
) -> tuple[np.ndarray, float]:
    result = output_fir_for_export_report(
        fir,
        output_fir_polarity_invert=output_fir_polarity_invert,
        output_fir_remove_nyquist_enabled=output_fir_remove_nyquist_enabled,
        output_fir_remove_nyquist_strength=output_fir_remove_nyquist_strength,
        fir_auto_gain_enabled=fir_auto_gain_enabled,
        output_fir_cosine_taper_enabled=output_fir_cosine_taper_enabled,
    )
    return result.fir, result.removed_nyquist_amplitude


def output_fir_for_export_report(
    fir: np.ndarray,
    *,
    output_fir_polarity_invert: bool,
    output_fir_remove_nyquist_enabled: bool,
    output_fir_remove_nyquist_strength: float,
    fir_auto_gain_enabled: bool = True,
    output_fir_cosine_taper_enabled: bool = False,
) -> ExportFIRPostprocessResult:
    output_fir = np.asarray(fir, dtype=float).copy()
    removed_amplitude = 0.0
    if output_fir_remove_nyquist_enabled and float(output_fir_remove_nyquist_strength) > 0.0:
        output_fir, removed_amplitude = remove_nyquist_component(
            output_fir,
            strength=output_fir_remove_nyquist_strength,
        )
    if output_fir_polarity_invert:
        output_fir = -output_fir
        removed_amplitude = -removed_amplitude
    # Last shape-changing operation; Auto Gain afterwards is a scalar only.
    output_fir = apply_output_window(output_fir, output_fir_cosine_taper_enabled)
    auto_gain = (
        apply_fixed_fir_auto_gain(output_fir)
        if fir_auto_gain_enabled else _bypassed_fir_auto_gain(output_fir)
    )
    return ExportFIRPostprocessResult(
        fir=auto_gain.coefficients,
        removed_nyquist_amplitude=float(removed_amplitude),
        auto_gain=auto_gain,
        auto_gain_enabled=bool(fir_auto_gain_enabled),
    )
