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
from .phase_curves import frequency_axis, phase_error_deg, wrapped_phase_deg
from .design import calculate_group_delay_ms
from .output_tap_policy import (
    StandaloneOutputTapResult,
    apply_standalone_output_tap_policy,
)


@dataclass(frozen=True)
class ExportFIRPostprocessResult:
    fir: np.ndarray
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
    *, crop_delay_samples: float = 0.0, original_taps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    frequency = frequency_axis(sample_rate, fft_size)
    response = np.fft.rfft(np.asarray(fir, dtype=float), n=fft_size)
    delay_samples = ((original_taps or len(fir)) - 1) / 2 - crop_delay_samples
    response = response * np.exp(1j * 2.0 * np.pi * frequency * delay_samples / sample_rate)
    return frequency, response


def result_for_export_fir_stage(
    result: Any,
    sample_rate: int,
    output_fir_polarity_invert: bool,
    fir_auto_gain_enabled: bool = True,
    output_fir_cosine_taper_enabled: bool = False,
) -> Any:
    export_result = output_fir_for_export_report(
        result.fir,
        output_fir_polarity_invert=output_fir_polarity_invert,
        fir_auto_gain_enabled=fir_auto_gain_enabled,
        output_fir_cosine_taper_enabled=(
            output_fir_cosine_taper_enabled and not result.output_window_applied
        ),
        output_fir_cosine_taper_protected_center_taps=(
            result.output_window_protected_center_taps
        ),
    )
    output_fir = export_result.fir

    analysis_fft_size = max((len(result.frequency) - 1) * 2, len(output_fir))
    _, realized_complex = realized_filter_complex_response(
        output_fir,
        sample_rate,
        analysis_fft_size,
        crop_delay_samples=result.crop_delay_samples,
        original_taps=getattr(result, "fir_original_taps", None),
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
        group_delay_ms=(calculate_group_delay_ms(output_fir, sample_rate, result.frequency)
                        + result.crop_delay_samples * 1000.0 / sample_rate),
    )


def result_for_standalone_output_taps(
    result: Any,
    sample_rate: int,
    configured_taps: int,
    *,
    auto_crop: bool = False,
    output_window_enabled: bool = False,
    kaiser_band_split_active: bool = False,
) -> tuple[Any, StandaloneOutputTapResult]:
    """Apply the standalone frame policy and rebuild coefficient-derived data."""
    if getattr(result, "output_window_applied", False):
        raise ValueError("a finalized FIR must not be cropped or tapered again")
    tap_result = apply_standalone_output_tap_policy(
        result.fir,
        configured_taps,
        auto_crop=auto_crop,
        minimum_taps=getattr(result, "basic_fir_taps", 1),
        output_window_enabled=output_window_enabled,
        kaiser_band_split_active=kaiser_band_split_active,
    )
    output_fir = tap_result.coefficients
    analysis_fft_size = max((len(result.frequency) - 1) * 2, len(output_fir))
    _, realized_complex = realized_filter_complex_response(
        output_fir,
        sample_rate,
        analysis_fft_size,
        crop_delay_samples=tap_result.crop_delay_samples,
        original_taps=tap_result.input_taps,
    )
    realized_gain_db = 20.0 * np.log10(np.maximum(np.abs(realized_complex), 1e-12))
    realized_phase_unwrapped_deg = np.rad2deg(np.unwrap(np.angle(realized_complex)))
    realized_phase_wrapped_deg = wrapped_phase_deg(realized_phase_unwrapped_deg)
    updated = replace(
        result,
        fir=output_fir,
        realized_gain_db=realized_gain_db,
        gain_error_db=realized_gain_db - result.target_gain_db,
        realized_phase_wrapped_deg=realized_phase_wrapped_deg,
        realized_phase_unwrapped_deg=realized_phase_unwrapped_deg,
        phase_error_deg=phase_error_deg(
            result.target_phase_unwrapped_deg,
            realized_phase_unwrapped_deg,
            np.minimum(realized_gain_db, result.target_gain_db),
        ),
        impulse_response=output_fir.copy(),
        group_delay_ms=(calculate_group_delay_ms(output_fir, sample_rate, result.frequency)
                        + tap_result.crop_delay_samples * 1000.0 / sample_rate),
        step_response=np.cumsum(output_fir),
        output_window_protected_center_taps=tap_result.taper_reference_taps,
        output_window_applied=tap_result.window_applied,
        crop_delay_samples=tap_result.crop_delay_samples,
        fir_original_taps=tap_result.input_taps,
    )
    return updated, tap_result


def output_fir_for_export(
    fir: np.ndarray,
    *,
    output_fir_polarity_invert: bool,
    fir_auto_gain_enabled: bool = True,
    output_fir_cosine_taper_enabled: bool = False,
    output_fir_cosine_taper_protected_center_taps: int | None = None,
) -> np.ndarray:
    result = output_fir_for_export_report(
        fir,
        output_fir_polarity_invert=output_fir_polarity_invert,
        fir_auto_gain_enabled=fir_auto_gain_enabled,
        output_fir_cosine_taper_enabled=output_fir_cosine_taper_enabled,
        output_fir_cosine_taper_protected_center_taps=(
            output_fir_cosine_taper_protected_center_taps
        ),
    )
    return result.fir


def output_fir_for_export_report(
    fir: np.ndarray,
    *,
    output_fir_polarity_invert: bool,
    fir_auto_gain_enabled: bool = True,
    output_fir_cosine_taper_enabled: bool = False,
    output_fir_cosine_taper_protected_center_taps: int | None = None,
) -> ExportFIRPostprocessResult:
    output_fir = np.asarray(fir, dtype=float).copy()
    if output_fir_polarity_invert:
        output_fir = -output_fir
    # Legacy raw-FIR API only. Finalized callers disable this branch; OFF
    # must not invoke another window operation at the output boundary.
    if output_fir_cosine_taper_enabled:
        output_fir = apply_output_window(
            output_fir, True,
            protected_center_taps=output_fir_cosine_taper_protected_center_taps,
        )
    auto_gain = (
        apply_fixed_fir_auto_gain(output_fir)
        if fir_auto_gain_enabled else _bypassed_fir_auto_gain(output_fir)
    )
    return ExportFIRPostprocessResult(
        fir=auto_gain.coefficients,
        auto_gain=auto_gain,
        auto_gain_enabled=bool(fir_auto_gain_enabled),
    )
