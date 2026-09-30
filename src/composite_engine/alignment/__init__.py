from __future__ import annotations

import numpy as np
from fir_design_common import tap_center, center_phase


def center_pad(coefficients: np.ndarray, output_taps: int) -> tuple[np.ndarray, float]:
    """Pad around the tap center without cropping or changing coefficients."""
    values = np.asarray(coefficients)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("coefficients must be a non-empty vector")
    if output_taps < values.size:
        raise ValueError("center padding cannot crop coefficients")
    before = (int(output_taps) - values.size) // 2
    after = int(output_taps) - values.size - before
    padded = np.pad(values, (before, after))
    error = before + tap_center(values.size) - tap_center(output_taps)
    return padded, float(error)


def remove_center_time_reference(response: np.ndarray, frequency_hz: np.ndarray, sample_rate_hz: int, center_position: float) -> np.ndarray:
    return np.asarray(response, dtype=complex) * center_phase(frequency_hz, sample_rate_hz, center_position)


def apply_delay(response: np.ndarray, frequency_hz: np.ndarray, sample_rate_hz: int, delay_samples: float) -> np.ndarray:
    return np.asarray(response, dtype=complex) * np.exp(
        -1j * 2.0 * np.pi * np.asarray(frequency_hz) * float(delay_samples) / float(sample_rate_hz)
    )


def dsp_additional_delay(output_taps: int, *, has_fir: bool) -> float:
    """Move an IIR-only path to the common FIR tap center."""
    return 0.0 if has_fir else tap_center(output_taps)
