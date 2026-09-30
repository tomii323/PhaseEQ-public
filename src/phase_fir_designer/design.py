from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft
from scipy import signal
from scipy.sparse.linalg import LinearOperator, lsqr
from fir_design_common import generation_fft_size, generation_frequency_axis, tap_center, center_phase
from fir_design_common import normalize_dc_gain as _normalize_dc_gain

from .config import DesignConfig
from .linear_fir import add_linear_fir_target_response, apply_linear_fir_filters
from .phase_curves import (
    build_target_curves,
    frequency_axis,
    phase_error_deg,
    wrapped_phase_deg,
)


@dataclass(frozen=True)
class DesignResult:
    fir: np.ndarray
    frequency: np.ndarray
    target_gain_db: np.ndarray
    realized_gain_db: np.ndarray
    gain_error_db: np.ndarray
    target_phase_wrapped_deg: np.ndarray
    realized_phase_wrapped_deg: np.ndarray
    realized_phase_unwrapped_deg: np.ndarray
    phase_error_deg: np.ndarray
    impulse_response: np.ndarray
    group_delay_ms: np.ndarray
    step_response: np.ndarray
    target_phase_unwrapped_deg: np.ndarray


def design_phase_fir(
    config: DesignConfig,
    *,
    compute_group_delay: bool = True,
    apply_linear_fir_overlay: bool = True,
) -> DesignResult:
    normalized = config.normalized()
    normalized.validate()

    design_fft_size = _weighted_ls_grid_size(normalized)
    design_freq = generation_frequency_axis(normalized.sample_rate, design_fft_size)
    design_target_gain_db, design_target_phase_unwrapped_deg = build_target_curves(normalized, design_freq)
    design_target_gain_db, design_target_phase_unwrapped_deg = _hold_nyquist_endpoint_from_previous(
        normalized,
        design_freq,
        design_target_gain_db,
        design_target_phase_unwrapped_deg,
    )
    fir = _design_fir_with_gain_correction(
        normalized,
        design_freq,
        design_target_gain_db,
        design_target_phase_unwrapped_deg,
    )
    if apply_linear_fir_overlay:
        fir = apply_linear_fir_filters(normalized, fir)
    fir = fir.astype(np.float64, copy=False)

    analysis_fft_size = _analysis_fft_size_for_taps(normalized.analysis_fft_size, fir.size)
    analysis_freq = frequency_axis(normalized.sample_rate, analysis_fft_size)
    target_gain_db, target_phase_unwrapped_deg = build_target_curves(normalized, analysis_freq)
    target_gain_db, target_phase_unwrapped_deg = _hold_nyquist_endpoint_from_previous(
        normalized,
        analysis_freq,
        target_gain_db,
        target_phase_unwrapped_deg,
    )
    if apply_linear_fir_overlay:
        target_gain_db, target_phase_unwrapped_deg = add_linear_fir_target_response(
            normalized,
            target_gain_db,
            target_phase_unwrapped_deg,
            analysis_fft_size,
            _realized_response,
        )
    realized_complex = _realized_response(fir, normalized.sample_rate, analysis_fft_size)
    realized_gain_db = 20.0 * np.log10(np.maximum(np.abs(realized_complex), 1e-12))
    realized_phase_unwrapped = np.rad2deg(np.unwrap(np.angle(realized_complex)))
    realized_phase_wrapped = wrapped_phase_deg(realized_phase_unwrapped)
    target_phase_wrapped = wrapped_phase_deg(target_phase_unwrapped_deg)
    gain_error = realized_gain_db - target_gain_db
    phase_error = phase_error_deg(
        target_phase_unwrapped_deg,
        realized_phase_unwrapped,
        np.minimum(realized_gain_db, target_gain_db),
    )

    if compute_group_delay:
        group_delay_ms = calculate_group_delay_ms(fir, normalized.sample_rate, analysis_freq)
    else:
        group_delay_ms = np.full_like(analysis_freq, np.nan, dtype=float)

    return DesignResult(
        fir=fir,
        frequency=analysis_freq,
        target_gain_db=target_gain_db,
        realized_gain_db=realized_gain_db,
        gain_error_db=gain_error,
        target_phase_wrapped_deg=target_phase_wrapped,
        realized_phase_wrapped_deg=realized_phase_wrapped,
        realized_phase_unwrapped_deg=realized_phase_unwrapped,
        phase_error_deg=phase_error,
        impulse_response=fir.copy(),
        group_delay_ms=group_delay_ms,
        step_response=np.cumsum(fir),
        target_phase_unwrapped_deg=target_phase_unwrapped_deg,
    )


def _realized_response(fir: np.ndarray, sample_rate: int, fft_size: int) -> np.ndarray:
    response = fft.rfft(fir, n=fft_size)
    freq = frequency_axis(sample_rate, fft_size)
    return response * center_phase(freq, sample_rate, tap_center(fir.size))


def calculate_group_delay_ms(fir: np.ndarray, sample_rate: int, frequency: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(fir, dtype=float).ravel()
    frequency = np.asarray(frequency, dtype=float)
    fft_size = _matching_rfft_size(frequency, int(sample_rate))
    if coefficients.size and fft_size is not None:
        response = fft.rfft(coefficients, n=fft_size)
        weighted_response = fft.rfft(
            np.arange(coefficients.size, dtype=float) * coefficients,
            n=fft_size,
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            gd_samples = np.real(weighted_response / response)
        return np.nan_to_num(gd_samples / sample_rate * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)

    _, gd_samples = signal.group_delay((fir, [1.0]), fs=sample_rate, w=frequency)
    return np.nan_to_num(gd_samples / sample_rate * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)


def _matching_rfft_size(frequency: np.ndarray, sample_rate: int) -> int | None:
    """Return the FFT size when *frequency* is a complete real-FFT axis."""

    axis = np.asarray(frequency, dtype=float)
    if axis.ndim != 1 or axis.size < 2 or sample_rate <= 0:
        return None
    fft_size = (int(axis.size) - 1) * 2
    expected = fft.rfftfreq(fft_size, d=1.0 / float(sample_rate))
    tolerance = max(float(sample_rate), 1.0) * np.finfo(float).eps * 16.0
    if axis.shape == expected.shape and np.allclose(axis, expected, rtol=1e-12, atol=tolerance):
        return fft_size
    return None


def _design_fir_with_gain_correction(
    config: DesignConfig,
    frequency: np.ndarray,
    target_gain_db: np.ndarray,
    target_phase_unwrapped_deg: np.ndarray,
) -> np.ndarray:
    magnitude_gain_db = np.asarray(target_gain_db, dtype=float).copy()
    correction_mask = _gain_correction_update_mask(config, frequency)
    correction_floor_db, correction_ceiling_db = _gain_correction_bounds_db(target_gain_db, correction_mask)
    fit_fft_size = _fft_size_from_rfft_frequency(frequency)
    fir = _fir_from_target(config, frequency, magnitude_gain_db, target_phase_unwrapped_deg)
    for _ in range(5):
        realized = _realized_response(fir, config.sample_rate, fit_fft_size)
        realized_gain_db = 20.0 * np.log10(np.maximum(np.abs(realized), 1e-9))
        gain_error_db = realized_gain_db - target_gain_db
        evaluated_error = gain_error_db[correction_mask]
        if evaluated_error.size == 0 or np.nanmax(np.abs(evaluated_error)) < 0.05:
            break
        correction_db = np.clip(-gain_error_db, -12.0, 12.0)
        next_magnitude_gain_db = magnitude_gain_db.copy()
        next_magnitude_gain_db[correction_mask] = np.clip(
            magnitude_gain_db[correction_mask] + correction_db[correction_mask],
            correction_floor_db,
            correction_ceiling_db,
        )
        magnitude_gain_db = next_magnitude_gain_db
        magnitude_gain_db[0] = target_gain_db[0]
        if np.isclose(frequency[-1], config.sample_rate / 2):
            magnitude_gain_db[-1] = target_gain_db[-1]
        fir = _fir_from_target(config, frequency, magnitude_gain_db, target_phase_unwrapped_deg)
    return fir


def _gain_correction_bounds_db(target_gain_db: np.ndarray, correction_mask: np.ndarray) -> tuple[float, float]:
    target = np.asarray(target_gain_db, dtype=float)
    mask = np.asarray(correction_mask, dtype=bool) & np.isfinite(target)
    if not np.any(mask):
        return -48.0, 48.0
    floor = min(-48.0, float(np.nanmin(target[mask])) - 12.0)
    ceiling = max(48.0, float(np.nanmax(target[mask])) + 12.0)
    return max(floor, -180.0), min(ceiling, 180.0)


def _fir_from_target(
    config: DesignConfig,
    frequency: np.ndarray,
    magnitude_gain_db: np.ndarray,
    phase_unwrapped_deg: np.ndarray,
) -> np.ndarray:
    magnitude = 10.0 ** (np.asarray(magnitude_gain_db, dtype=float) / 20.0)
    response = magnitude * np.exp(1j * np.deg2rad(phase_unwrapped_deg))
    response[0] = _real_boundary_value(magnitude[0], phase_unwrapped_deg[0]) + 0.0j
    if _has_nyquist_bin(config.taps) and response.size > 1:
        # An even-tap, half-sample-centred FIR cannot realize a non-zero real
        # response at Nyquist. A one-bin discontinuity creates a narrow peak
        # immediately below Nyquist, so taper across one tap-resolution band
        # to the realizable Type-II boundary instead.
        fft_size = max(2, (response.size - 1) * 2)
        transition_bins = max(2, 32 * int(np.ceil(fft_size / int(config.taps))))
        transition_bins = min(transition_bins, response.size - 1)
        taper = np.cos(np.linspace(0.0, np.pi / 2.0, transition_bins + 1))
        response[-(transition_bins + 1):] *= taper
    fir = _weighted_complex_ls_fir(config, frequency, response)
    if config.dc_gain_normalize:
        return _normalize_dc_gain(fir, target_dc_abs=magnitude[0])
    return fir


def _weighted_complex_ls_fir(
    config: DesignConfig,
    source_frequency: np.ndarray,
    target_response: np.ndarray,
) -> np.ndarray:
    taps = int(config.taps)
    grid_size = _weighted_ls_grid_size(config)
    target_frequency = generation_frequency_axis(config.sample_rate, grid_size)
    target = _interpolate_complex_response(source_frequency, target_response, target_frequency)
    weights = _weighted_ls_frequency_weights(config, target_frequency)
    sqrt_weights = np.sqrt(weights)
    delay = center_phase(target_frequency, config.sample_rate, tap_center(taps))
    weighted_target = sqrt_weights * target
    rhs = np.concatenate([np.real(weighted_target), np.imag(weighted_target)])

    def matvec(coefficients: np.ndarray) -> np.ndarray:
        response = fft.rfft(np.asarray(coefficients, dtype=float), n=grid_size) * delay
        weighted = sqrt_weights * response
        return np.concatenate([np.real(weighted), np.imag(weighted)])

    def rmatvec(residual: np.ndarray) -> np.ndarray:
        residual = np.asarray(residual, dtype=float)
        size = target_frequency.size
        weighted_residual = sqrt_weights * (residual[:size] + 1j * residual[size:])
        spectrum = np.conj(delay) * weighted_residual
        adjoint_spectrum = spectrum.copy()
        if adjoint_spectrum.size > 2:
            adjoint_spectrum[1:-1] *= 0.5
        adjoint = fft.irfft(adjoint_spectrum, n=grid_size) * grid_size
        return np.asarray(adjoint[:taps], dtype=float)

    operator = LinearOperator(
        shape=(rhs.size, taps),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=float,
    )
    solution = lsqr(
        operator,
        rhs,
        damp=1e-6,
        atol=1e-7,
        btol=1e-7,
        iter_lim=300,
        show=False,
    )[0]
    return np.asarray(solution, dtype=np.float64)


def _weighted_ls_grid_size(config: DesignConfig) -> int:
    if config.fir_generation_grid == "multiway-v1":
        return generation_fft_size(config.sample_rate, config.taps)
    minimum = int(config.taps) * 8
    return fft.next_fast_len(minimum, real=True)


def _fft_size_from_rfft_frequency(frequency: np.ndarray) -> int:
    frequency = np.asarray(frequency, dtype=float)
    return max(1, (int(frequency.size) - 1) * 2)


def _weighted_ls_frequency_weights(config: DesignConfig, frequency: np.ndarray) -> np.ndarray:
    frequency = np.asarray(frequency, dtype=float)
    return np.ones_like(frequency, dtype=float)


def _interpolate_complex_response(
    source_frequency: np.ndarray,
    source_response: np.ndarray,
    target_frequency: np.ndarray,
) -> np.ndarray:
    source_frequency = np.asarray(source_frequency, dtype=float)
    source_response = np.asarray(source_response, dtype=complex)
    target_frequency = np.asarray(target_frequency, dtype=float)
    gain_db = 20.0 * np.log10(np.maximum(np.abs(source_response), 1e-12))
    phase_deg = np.rad2deg(np.unwrap(np.angle(source_response)))
    interpolated_gain = np.interp(target_frequency, source_frequency, gain_db)
    interpolated_phase = np.interp(target_frequency, source_frequency, phase_deg)
    return _complex_from_gain_phase(interpolated_gain, interpolated_phase)


def _analysis_fft_size_for_taps(selected_fft_size: int, taps: int) -> int:
    fft_size = max(int(selected_fft_size), int(taps))
    return fft.next_fast_len(fft_size, real=True)


def _hold_nyquist_endpoint_from_previous(
    config: DesignConfig,
    frequency: np.ndarray,
    gain_db: np.ndarray,
    phase_unwrapped_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frequency = np.asarray(frequency, dtype=float)
    gain = np.asarray(gain_db, dtype=float).copy()
    phase = np.asarray(phase_unwrapped_deg, dtype=float).copy()
    if frequency.size < 2:
        return gain, phase
    if np.isclose(float(frequency[-1]), float(config.sample_rate) / 2.0):
        gain[-1] = gain[-2]
        phase[-1] = phase[-2]
    return gain, phase


def _complex_from_gain_phase(gain_db: np.ndarray, phase_unwrapped_deg: np.ndarray) -> np.ndarray:
    magnitude = 10.0 ** (np.asarray(gain_db, dtype=float) / 20.0)
    phase = np.deg2rad(np.asarray(phase_unwrapped_deg, dtype=float))
    return magnitude * np.exp(1j * phase)


def _gain_correction_update_mask(config: DesignConfig, frequency: np.ndarray) -> np.ndarray:
    return np.ones_like(np.asarray(frequency, dtype=float), dtype=bool)


def _real_boundary_value(magnitude: float, phase_deg: float) -> float:
    sign = -1.0 if np.cos(np.deg2rad(phase_deg)) < 0.0 else 1.0
    return float(magnitude) * sign


def _has_nyquist_bin(taps: int) -> bool:
    return int(taps) % 2 == 0
