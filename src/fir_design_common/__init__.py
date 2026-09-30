"""Application-independent generation grid and real FIR center conventions."""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def compose_fir_stages(stages, taps: int) -> np.ndarray:
    """Full linear convolution followed by one central crop/pad, without a window.

    An odd unmatched sample stays on the right, matching PhaseEQ. Different
    support parity quantizes the geometric center by half a sample.
    """
    taps = int(taps)
    if taps < 1:
        raise ValueError("tap count must be positive")
    result = None
    for stage in stages:
        if np.iscomplexobj(stage):
            raise ValueError("FIR coefficients must be real")
        values = np.asarray(stage, dtype=float)
        if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
            raise ValueError("FIR stages must be nonempty finite vectors")
        result = values.copy() if result is None else fftconvolve(result, values, mode="full")
    if result is None:
        raise ValueError("at least one FIR stage is required")
    if result.size < taps:
        left = (taps - result.size) // 2
        return np.pad(result, (left, taps - result.size - left))
    start = (result.size - taps) // 2
    return result[start:start + taps].copy()


def generation_fft_size(sample_rate: int, *tap_counts: int) -> int:
    if int(sample_rate) <= 0 or any(int(t) < 0 for t in tap_counts):
        raise ValueError("sample rate must be positive and tap counts non-negative")
    size = max(2, int(sample_rate), *(int(t) for t in tap_counts))
    return size + size % 2


def generation_frequency_axis(sample_rate: int, fft_size: int) -> np.ndarray:
    if int(sample_rate) <= 0 or int(fft_size) < 2 or int(fft_size) % 2:
        raise ValueError("generation grid requires positive sample rate and even FFT size")
    return np.fft.rfftfreq(int(fft_size), 1.0 / int(sample_rate))


def tap_center(tap_count: int) -> float:
    if int(tap_count) < 1:
        raise ValueError("tap count must be positive")
    return (int(tap_count) - 1) / 2.0


def center_phase(frequency: np.ndarray, sample_rate: int, center: float) -> np.ndarray:
    return np.exp(1j * 2.0 * np.pi * np.asarray(frequency) * float(center) / int(sample_rate))


def project_centered_fir(response: np.ndarray, taps: int) -> np.ndarray:
    """Project an even-rFFT grid onto integer or half-sample centered support.

    The spectrum is center-referenced, not causal. Real irFFT projects DC and
    Nyquist to real values. Padding beyond the grid does not add resolution.
    """
    taps = int(taps)
    center = tap_center(taps)
    values = np.asarray(response, dtype=complex)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("response must be a finite even-rFFT spectrum")
    size = 2 * (values.size - 1)
    anchor = taps // 2
    advance = anchor - center
    if advance:
        values = values * center_phase(np.fft.rfftfreq(size), 1, advance)
    centered = np.roll(np.fft.irfft(values, n=size), size // 2)
    start = size // 2 - anchor
    stop = start + taps
    return np.pad(centered[max(start, 0):min(stop, size)],
                  (max(-start, 0), max(stop-size, 0))).astype(float)


def normalize_dc_gain(fir: np.ndarray, target_dc_abs: float) -> np.ndarray:
    dc = float(np.sum(fir))
    if abs(dc) < 1e-12:
        return fir
    return fir * (target_dc_abs / abs(dc))
