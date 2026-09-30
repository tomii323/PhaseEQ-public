from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class AlignmentResult:
    aligned: np.ndarray
    delay_samples: float
    delay_ms: float
    method: str
    confidence: float
    fallback_used: bool = False


def apply_fractional_delay(values: np.ndarray, delay_samples: float) -> np.ndarray:
    """Delay a real impulse by a fractional number of samples."""
    x = _finite_signal(values)
    if x.size == 0 or np.isclose(delay_samples, 0.0):
        return x.copy()
    spectrum = np.fft.rfft(x)
    bins = np.arange(spectrum.size, dtype=float)
    phase = np.exp(-1j * 2.0 * np.pi * bins * float(delay_samples) / float(x.size))
    return np.fft.irfft(spectrum * phase, n=x.size).astype(float, copy=False)


def align_impulse_to_reference(
    source: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    *,
    method: str = "cross_correlation",
    band: tuple[float, float] = (200.0, 10_000.0),
    confidence_threshold: float = 0.15,
) -> AlignmentResult:
    """Align source IR to reference IR and return the aligned source.

    delay_samples is the measured source delay relative to reference.  The
    returned signal has the opposite delay applied.
    """
    src = _finite_signal(source)
    ref = _finite_signal(reference)
    if src.size == 0:
        return AlignmentResult(src.copy(), 0.0, 0.0, "none", 0.0, fallback_used=True)
    if ref.size == 0:
        return AlignmentResult(src.copy(), 0.0, 0.0, "none", 0.0, fallback_used=True)
    src, ref = _same_length(src, ref)

    selected_method = str(method).lower()
    if selected_method in {"peak", "ir_peak", "ir peak abs", "peak_abs"}:
        delay = _peak_abs_delay(src, ref)
        confidence = 1.0
        fallback = False
        method_name = "IR Peak Abs"
    else:
        delay, confidence = _cross_correlation_delay(
            _band_limited(src, sample_rate, band),
            _band_limited(ref, sample_rate, band),
        )
        fallback = confidence < confidence_threshold or not np.isfinite(delay)
        if fallback:
            delay = _peak_abs_delay(src, ref)
            method_name = "IR Peak Abs"
        else:
            method_name = "Band-limited Cross Correlation"

    aligned = apply_fractional_delay(src, -float(delay))
    return AlignmentResult(
        aligned=aligned,
        delay_samples=float(delay),
        delay_ms=float(delay) / float(sample_rate) * 1000.0,
        method=method_name,
        confidence=float(np.nan_to_num(confidence)),
        fallback_used=fallback,
    )


def _same_length(source: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = max(source.size, reference.size)
    return _pad_center(source, n), _pad_center(reference, n)


def _pad_center(values: np.ndarray, size: int) -> np.ndarray:
    if values.size == size:
        return values.copy()
    if values.size > size:
        start = (values.size - size) // 2
        return values[start : start + size].copy()
    before = (size - values.size) // 2
    after = size - values.size - before
    return np.pad(values, (before, after))


def _band_limited(values: np.ndarray, sample_rate: int, band: tuple[float, float]) -> np.ndarray:
    x = _finite_signal(values)
    low_hz, high_hz = sorted((float(band[0]), float(band[1])))
    nyquist = float(sample_rate) / 2.0
    low = max(low_hz, 1.0)
    high = min(high_hz, nyquist * 0.98)
    if not (0.0 < low < high < nyquist):
        return x
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    padlen = min(x.size - 1, 3 * (2 * sos.shape[0] + 1))
    if padlen <= 1:
        return x
    return signal.sosfiltfilt(sos, x, padlen=padlen)


def _cross_correlation_delay(source: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    src = _zero_mean_unit_norm(source)
    ref = _zero_mean_unit_norm(reference)
    if src.size == 0 or ref.size == 0:
        return 0.0, 0.0
    corr = signal.correlate(src, ref, mode="full", method="fft")
    lags = signal.correlation_lags(src.size, ref.size, mode="full").astype(float)
    idx = int(np.argmax(np.abs(corr)))
    lag = lags[idx]
    if 0 < idx < corr.size - 1:
        y0, y1, y2 = np.abs(corr[idx - 1]), np.abs(corr[idx]), np.abs(corr[idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if not np.isclose(denom, 0.0):
            lag += 0.5 * (y0 - y2) / denom
    confidence = float(np.max(np.abs(corr)))
    return float(lag), confidence


def _zero_mean_unit_norm(values: np.ndarray) -> np.ndarray:
    x = _finite_signal(values)
    if x.size == 0:
        return x
    x = x - float(np.mean(x))
    norm = float(np.linalg.norm(x))
    if norm <= 1e-12:
        return np.zeros_like(x)
    return x / norm


def _finite_signal(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size == 0 or np.all(np.isfinite(x)):
        return x.copy()
    return np.where(np.isfinite(x), x, 0.0).astype(float, copy=False)


def _peak_abs_delay(source: np.ndarray, reference: np.ndarray) -> float:
    src_idx = int(np.argmax(np.abs(source))) if source.size else 0
    ref_idx = int(np.argmax(np.abs(reference))) if reference.size else 0
    return float(src_idx - ref_idx)
