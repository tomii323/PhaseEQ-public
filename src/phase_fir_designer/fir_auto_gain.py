from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

TARGET_FIR_PEAK_DB = -0.3
FFT_OVERSAMPLING = 64
ATTENUATION_ONLY = True
DB_EPSILON = 1e-300


@dataclass(frozen=True)
class FIRAutoGainResult:
    """Result of fixed FIR Auto Gain for one FIR.

    Parameters
    ----------
    coefficients
        Output FIR coefficients after applying the common scalar gain.
    original_peak_db
        Frequency-response peak of the input FIR in dB.
    output_peak_db
        Frequency-response peak after the scalar gain in dB.
    target_peak_db
        Fixed target peak in dB.
    applied_gain_db
        Applied scalar gain in dB. This value is never positive.
    linear_gain
        Linear gain applied to all coefficients.
    fft_size
        FFT size used for peak analysis.
    oversampling
        Oversampling factor used to choose the FFT size.
    changed
        True when attenuation was applied.
    """

    coefficients: NDArray[np.float64]
    original_peak_db: float
    output_peak_db: float
    target_peak_db: float
    applied_gain_db: float
    linear_gain: float
    fft_size: int
    oversampling: int
    changed: bool


@dataclass(frozen=True)
class MultiwayFIRAutoGainResult:
    """Result of fixed FIR Auto Gain for multiple FIR ways.

    The same scalar gain is applied to all ways so relative levels and
    crossover balance are preserved.
    """

    coefficients: tuple[NDArray[np.float64], ...]
    original_channel_peak_db: tuple[float, ...]
    output_channel_peak_db: tuple[float, ...]
    original_overall_peak_db: float
    output_overall_peak_db: float
    target_peak_db: float
    applied_gain_db: float
    linear_gain: float
    fft_size: int
    oversampling: int
    changed: bool


def next_power_of_two(value: int) -> int:
    """Return the smallest power of two greater than or equal to ``value``.

    Parameters
    ----------
    value
        Positive integer lower bound.

    Returns
    -------
    int
        Power-of-two FFT size.

    Raises
    ------
    ValueError
        If ``value`` is less than one.
    """

    if value <= 0:
        raise ValueError("value must be greater than zero")
    return 1 << (int(value) - 1).bit_length()


def validate_fir_coefficients(coefficients: ArrayLike, *, name: str = "coefficients") -> NDArray[np.float64]:
    """Validate FIR coefficients as a finite one-dimensional float64 array.

    Parameters
    ----------
    coefficients
        Input FIR coefficients.
    name
        Name used in validation errors.

    Returns
    -------
    numpy.ndarray
        Float64 one-dimensional view or copy.

    Raises
    ------
    ValueError
        If the input is empty, non-finite, or not one-dimensional.
    """

    fir = np.asarray(coefficients, dtype=np.float64)
    if fir.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if fir.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(fir)):
        raise ValueError(f"{name} contains NaN or infinity")
    return fir


def calculate_fir_auto_gain_fft_size(tap_count: int, *, oversampling: int = FFT_OVERSAMPLING) -> int:
    """Calculate the fixed FIR Auto Gain FFT size.

    Parameters
    ----------
    tap_count
        FIR tap count.
    oversampling
        FFT oversampling factor. The fixed specification uses 64.

    Returns
    -------
    int
        Power-of-two FFT size for response peak analysis.
    """

    if tap_count <= 0:
        raise ValueError("tap_count must be greater than zero")
    if oversampling < 1:
        raise ValueError("oversampling must be at least 1")
    return next_power_of_two(int(tap_count) * int(oversampling))


def measure_fir_peak_db(
    coefficients: ArrayLike,
    *,
    fft_size: int | None = None,
    oversampling: int = FFT_OVERSAMPLING,
    epsilon: float = DB_EPSILON,
) -> tuple[float, int]:
    """Measure the maximum FIR frequency-response magnitude in dB.

    This measures the filter response only. Input sample peaks, true peak,
    inter-sample peak, downstream mixer summing, and DAC/DSP headroom are out
    of scope.

    Parameters
    ----------
    coefficients
        FIR coefficients.
    fft_size
        Optional FFT size. When omitted, ``tap_count * oversampling`` rounded
        up to a power of two is used.
    oversampling
        Oversampling factor for FFT-size selection.
    epsilon
        Positive lower bound for stable dB conversion.

    Returns
    -------
    tuple[float, int]
        Peak response in dB and FFT size used.
    """

    fir = validate_fir_coefficients(coefficients)
    if fft_size is None:
        fft_size = calculate_fir_auto_gain_fft_size(fir.size, oversampling=oversampling)
    if fft_size < fir.size:
        raise ValueError("fft_size must be greater than or equal to FIR tap count")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be greater than zero")
    spectrum = np.fft.rfft(fir, n=int(fft_size))
    peak_linear = max(float(np.max(np.abs(spectrum))), float(epsilon))
    peak_db = 20.0 * np.log10(peak_linear)
    return float(peak_db), int(fft_size)


def calculate_fixed_fir_auto_gain_db(detected_peak_db: float, *, target_peak_db: float = TARGET_FIR_PEAK_DB) -> float:
    """Calculate fixed attenuation-only FIR Auto Gain.

    Parameters
    ----------
    detected_peak_db
        Detected maximum FIR response in dB.
    target_peak_db
        Target maximum response in dB. The fixed specification uses -0.3 dB.

    Returns
    -------
    float
        Gain in dB. This value is never positive.
    """

    if not np.isfinite(detected_peak_db):
        raise ValueError("detected_peak_db must be finite")
    if not np.isfinite(target_peak_db):
        raise ValueError("target_peak_db must be finite")
    required_gain_db = float(target_peak_db) - float(detected_peak_db)
    return min(0.0, required_gain_db)


def apply_fixed_fir_auto_gain(coefficients: ArrayLike) -> FIRAutoGainResult:
    """Apply fixed attenuation-only Auto Gain to final FIR coefficients.

    The same scalar gain is applied to all coefficients. Tap count, peak
    position, coefficient ratios, phase, group delay, and time placement are
    not changed.

    Parameters
    ----------
    coefficients
        Final FIR coefficients after all FIR processing and tap adjustment.

    Returns
    -------
    FIRAutoGainResult
        Auto Gain result and scaled coefficients.
    """

    fir = validate_fir_coefficients(coefficients)
    original_peak_db, fft_size = measure_fir_peak_db(fir, oversampling=FFT_OVERSAMPLING)
    applied_gain_db = calculate_fixed_fir_auto_gain_db(original_peak_db)
    linear_gain = float(10.0 ** (applied_gain_db / 20.0))
    output = fir * linear_gain
    output_peak_db, _ = measure_fir_peak_db(output, fft_size=fft_size, oversampling=FFT_OVERSAMPLING)
    return FIRAutoGainResult(
        coefficients=output,
        original_peak_db=original_peak_db,
        output_peak_db=output_peak_db,
        target_peak_db=TARGET_FIR_PEAK_DB,
        applied_gain_db=applied_gain_db,
        linear_gain=linear_gain,
        fft_size=fft_size,
        oversampling=FFT_OVERSAMPLING,
        changed=not np.isclose(applied_gain_db, 0.0, atol=1e-12),
    )


def apply_fixed_multiway_fir_auto_gain(filters: Sequence[ArrayLike]) -> MultiwayFIRAutoGainResult:
    """Apply one common fixed Auto Gain to multiple FIR ways.

    Parameters
    ----------
    filters
        FIR coefficient arrays for all ways.

    Returns
    -------
    MultiwayFIRAutoGainResult
        Multiway result with one common scalar gain applied to every way.
    """

    if not filters:
        raise ValueError("filters must not be empty")
    fir_list = tuple(
        validate_fir_coefficients(coefficients, name=f"filters[{index}]")
        for index, coefficients in enumerate(filters)
    )
    max_tap_count = max(fir.size for fir in fir_list)
    fft_size = calculate_fir_auto_gain_fft_size(max_tap_count, oversampling=FFT_OVERSAMPLING)
    original_peaks = tuple(
        measure_fir_peak_db(fir, fft_size=fft_size, oversampling=FFT_OVERSAMPLING)[0]
        for fir in fir_list
    )
    original_overall_peak_db = float(max(original_peaks))
    applied_gain_db = calculate_fixed_fir_auto_gain_db(original_overall_peak_db)
    linear_gain = float(10.0 ** (applied_gain_db / 20.0))
    output_filters = tuple(fir * linear_gain for fir in fir_list)
    output_peaks = tuple(
        measure_fir_peak_db(fir, fft_size=fft_size, oversampling=FFT_OVERSAMPLING)[0]
        for fir in output_filters
    )
    output_overall_peak_db = float(max(output_peaks))
    return MultiwayFIRAutoGainResult(
        coefficients=output_filters,
        original_channel_peak_db=original_peaks,
        output_channel_peak_db=output_peaks,
        original_overall_peak_db=original_overall_peak_db,
        output_overall_peak_db=output_overall_peak_db,
        target_peak_db=TARGET_FIR_PEAK_DB,
        applied_gain_db=applied_gain_db,
        linear_gain=linear_gain,
        fft_size=fft_size,
        oversampling=FFT_OVERSAMPLING,
        changed=not np.isclose(applied_gain_db, 0.0, atol=1e-12),
    )
