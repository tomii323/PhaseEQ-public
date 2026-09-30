from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class AutoIIRObjectiveWeights:
    one_octave_abs: float = 0.60
    third_octave_abs: float = 0.40
    perceptual_peak: float = 0.45
    positive_peak_area: float = 0.25
    center_shift: float = 0.10


@dataclass(frozen=True)
class AutoIIRObjectiveComponents:
    one_octave_abs_db: float
    third_octave_abs_db: float
    contour_abs_db: float
    perceptual_peak_db: float
    positive_peak_area_db_oct: float
    under_target_area_db_oct: float
    center_shift_oct: float
    score: float


@dataclass(frozen=True)
class AutoIIRObjectiveReference:
    one_octave_abs_db: float
    third_octave_abs_db: float
    contour_abs_db: float
    perceptual_peak_db: float
    positive_peak_area_db_oct: float
    under_target_area_db_oct: float


@dataclass(frozen=True)
class PositivePeakMetrics:
    """Audible protrusions above Target, measured after 1/3-oct smoothing."""

    height_db: float
    area_db_oct: float
    under_target_area_db_oct: float


def fractional_octave_smooth(
    values: np.ndarray,
    frequency_hz: np.ndarray,
    smoothing_oct: float,
) -> np.ndarray:
    """Gaussian smooth values on an approximately uniform log-frequency axis."""

    source = np.asarray(values, dtype=float)
    frequency = np.asarray(frequency_hz, dtype=float)
    if source.shape != frequency.shape or source.ndim != 1 or source.size < 3:
        raise ValueError("values and frequency must be matching one-dimensional arrays")
    octave_span = max(float(np.log2(frequency[-1] / frequency[0])), 1e-9)
    points_per_octave = max((frequency.size - 1) / octave_span, 1.0)
    sigma = max(float(smoothing_oct) * points_per_octave / 2.355, 0.25)
    radius = max(2, int(np.ceil(3.0 * sigma)))
    axis = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * np.square(axis / sigma))
    kernel /= np.sum(kernel)
    return np.convolve(np.pad(source, radius, mode="edge"), kernel, mode="valid")


def weighted_abs_error_mean(
    error_db: np.ndarray,
    frequency_hz: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Return weighted absolute dB error averaged over log2 frequency."""

    error = np.asarray(error_db, dtype=float)
    frequency = np.asarray(frequency_hz, dtype=float)
    weight = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if error.shape != frequency.shape or error.shape != weight.shape:
        raise ValueError("error, frequency, and weights must have the same shape")
    x = np.log2(frequency)
    denominator = max(float(np.trapezoid(weight, x=x)), 1e-12)
    return float(np.trapezoid(np.abs(error) * weight, x=x) / denominator)


def _weighted_log_area(
    values: np.ndarray,
    frequency_hz: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Integrate a non-negative curve in dB-oct without weight-scale drift."""

    value = np.maximum(np.asarray(values, dtype=float), 0.0)
    frequency = np.asarray(frequency_hz, dtype=float)
    weight = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if value.shape != frequency.shape or value.shape != weight.shape:
        raise ValueError("values, frequency, and weights must have the same shape")
    x = np.log2(frequency)
    span = max(float(x[-1] - x[0]), 1e-12)
    weight_mean = max(float(np.trapezoid(weight, x=x)) / span, 1e-12)
    return float(np.trapezoid(value * weight, x=x) / weight_mean)


def positive_peak_metrics(
    error_db: np.ndarray,
    frequency_hz: np.ndarray,
    weights: np.ndarray,
    *,
    residual_prominence_db: float = 0.0,
) -> PositivePeakMetrics:
    """Measure only positive response protrusions; dips never enter peak terms.

    Auto IIR uses ``Target - response`` error.  A response above Target is
    therefore negative.  The 1/3-oct response is compared with its 1-octave
    neighbourhood so gentle tonal balance remains a separate objective.
    """

    error = np.asarray(error_db, dtype=float)
    frequency = np.asarray(frequency_hz, dtype=float)
    weight = np.maximum(np.asarray(weights, dtype=float), 0.0)
    third = fractional_octave_smooth(error, frequency, 1.0 / 3.0)
    one = fractional_octave_smooth(error, frequency, 1.0)
    prominence = np.minimum(
        np.maximum(one - third - max(float(residual_prominence_db), 0.0), 0.0),
        np.maximum(-third, 0.0),
    )
    denominator = max(float(np.sum(weight)), 1e-12)
    height = float(
        np.power(
            np.sum(weight * np.power(prominence, 8.0)) / denominator,
            1.0 / 8.0,
        )
    )
    # Positive error means the corrected response has fallen below Target.
    # A 0.25 dB deadband prevents harmless shoulder ripple from being counted
    # as over-correction; only the new audible dip area is guarded.
    under_target = np.maximum(third - 0.25, 0.0)
    return PositivePeakMetrics(
        height_db=height,
        area_db_oct=_weighted_log_area(prominence, frequency, weight),
        under_target_area_db_oct=_weighted_log_area(under_target, frequency, weight),
    )


def auto_iir_objective_reference(
    error_db: np.ndarray,
    frequency_hz: np.ndarray,
    weights: np.ndarray,
) -> AutoIIRObjectiveReference:
    one_octave = fractional_octave_smooth(error_db, frequency_hz, 1.0)
    third_octave = fractional_octave_smooth(error_db, frequency_hz, 1.0 / 3.0)
    contour = third_octave - one_octave
    peaks = positive_peak_metrics(error_db, frequency_hz, weights)
    return AutoIIRObjectiveReference(
        one_octave_abs_db=weighted_abs_error_mean(one_octave, frequency_hz, weights),
        third_octave_abs_db=weighted_abs_error_mean(third_octave, frequency_hz, weights),
        contour_abs_db=weighted_abs_error_mean(contour, frequency_hz, weights),
        perceptual_peak_db=peaks.height_db,
        positive_peak_area_db_oct=peaks.area_db_oct,
        under_target_area_db_oct=peaks.under_target_area_db_oct,
    )


def evaluate_auto_iir_objective(
    error_db: np.ndarray,
    frequency_hz: np.ndarray,
    weights: np.ndarray,
    *,
    reference: AutoIIRObjectiveReference,
    center_shift_oct: float = 0.0,
    objective_weights: AutoIIRObjectiveWeights = AutoIIRObjectiveWeights(),
) -> AutoIIRObjectiveComponents:
    """Evaluate tonal, contour, positive-peak, and center terms."""

    one_octave = fractional_octave_smooth(error_db, frequency_hz, 1.0)
    third_octave = fractional_octave_smooth(error_db, frequency_hz, 1.0 / 3.0)
    one_abs = weighted_abs_error_mean(one_octave, frequency_hz, weights)
    third_abs = weighted_abs_error_mean(third_octave, frequency_hz, weights)
    contour_abs = weighted_abs_error_mean(third_octave - one_octave, frequency_hz, weights)
    peaks = positive_peak_metrics(error_db, frequency_hz, weights)
    center = max(float(center_shift_oct), 0.0)
    one_normalized = one_abs / max(float(reference.one_octave_abs_db), 0.25)
    contour_normalized = contour_abs / max(float(reference.contour_abs_db), 0.10)
    peak_normalized = peaks.height_db / max(float(reference.perceptual_peak_db), 0.25)
    peak_area_normalized = peaks.area_db_oct / max(
        float(reference.positive_peak_area_db_oct), 0.04
    )
    center_normalized = max(center - 1.0 / 12.0, 0.0) / (1.0 / 3.0)
    score = (
        float(objective_weights.one_octave_abs) * one_normalized
        + float(objective_weights.third_octave_abs) * contour_normalized
        + float(objective_weights.perceptual_peak) * peak_normalized
        + float(objective_weights.positive_peak_area) * peak_area_normalized
        + float(objective_weights.center_shift) * center_normalized**2
    )
    return AutoIIRObjectiveComponents(
        one_octave_abs_db=one_abs,
        third_octave_abs_db=third_abs,
        contour_abs_db=contour_abs,
        perceptual_peak_db=peaks.height_db,
        positive_peak_area_db_oct=peaks.area_db_oct,
        under_target_area_db_oct=peaks.under_target_area_db_oct,
        center_shift_oct=center,
        score=float(score),
    )
