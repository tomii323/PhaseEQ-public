from __future__ import annotations

import numpy as np

from ..alignment import remove_center_time_reference, tap_center


def fft_axis(sample_rate_hz: int, fft_size: int) -> np.ndarray:
    return np.fft.rfftfreq(int(fft_size), 1.0 / int(sample_rate_hz))


def fir_complex_response(coefficients: np.ndarray, sample_rate_hz: int, fft_size: int, *, center_position: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(coefficients, dtype=float)
    if fft_size < values.size:
        raise ValueError("fft_size must not be shorter than the FIR")
    frequency = fft_axis(sample_rate_hz, fft_size)
    response = np.fft.rfft(values, n=fft_size)
    center = tap_center(values.size) if center_position is None else float(center_position)
    return frequency, remove_center_time_reference(response, frequency, sample_rate_hz, center)


def group_delay_ms(response: np.ndarray, frequency_hz: np.ndarray) -> np.ndarray:
    phase = np.unwrap(np.angle(np.asarray(response, dtype=complex)))
    frequency = np.asarray(frequency_hz, dtype=float)
    if phase.size < 2:
        return np.zeros_like(frequency)
    delay = -np.gradient(phase, 2.0 * np.pi * frequency, edge_order=1) * 1000.0
    return np.nan_to_num(delay, nan=0.0, posinf=0.0, neginf=0.0)


def unwrapped_phase_on_reliable_branch(response: np.ndarray) -> np.ndarray:
    """Unwrap phase and choose its integer-2pi branch at peak response gain.

    A high-pass or band-pass response has an undefined phase in its deep
    stopband.  Starting ``unwrap`` at DC can therefore accumulate several
    artificial turns before reaching the useful passband.  The continuous
    slope is retained, while only the physically equivalent constant branch
    is moved so the peak-gain reference lies within [-pi, pi].
    """
    values = np.asarray(response, dtype=complex)
    phase = np.unwrap(np.angle(values))
    if phase.size == 0:
        return phase
    magnitude = np.abs(values)
    finite = np.isfinite(phase) & np.isfinite(magnitude)
    if not np.any(finite):
        return np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)
    candidates = np.flatnonzero(finite)
    reference_index = int(candidates[np.argmax(magnitude[finite])])
    branch_turns = round(float(phase[reference_index]) / (2.0 * np.pi))
    return phase - 2.0 * np.pi * branch_turns
