"""UI-independent response interpolation; callers own physical boundary policy."""
from __future__ import annotations

from typing import Literal

import numpy as np


def interpolate_phase(source_frequency, continuous_phase, target_frequency) -> np.ndarray:
    """Interpolate continuous phase on linear Hz, preserving its unit and branch.

    Accept degrees or radians consistently. No implicit unwrap, phase-center
    removal or DC policy: these belong to input preparation or the consumer.
    Linear Hz interpolation preserves a constant time-delay phase slope.
    """
    return interpolate_values(source_frequency, continuous_phase, target_frequency, axis="linear")


def interpolate_values(
    source_frequency: np.ndarray,
    source_values: np.ndarray,
    target_frequency: np.ndarray,
    *,
    axis: Literal["linear", "log", "log10"],
    frequency_floor: float = 1e-9,
) -> np.ndarray:
    """Interpolate real values, holding endpoints outside the measured range.

    Source points must already be finite, nonempty and ordered. The caller
    chooses dB versus linear amplitude and wrapped versus continuous phase;
    no sorting, unwrapping, delay removal or DC zeroing is implicit here.
    ``log`` and ``log10`` preserve existing numerical conventions on migration.
    """
    source = np.asarray(source_frequency, dtype=float)
    values = np.asarray(source_values, dtype=float)
    target = np.asarray(target_frequency, dtype=float)
    if source.ndim != 1 or source.size == 0 or values.shape != source.shape:
        raise ValueError("source frequency and values must be matching nonempty vectors")
    if axis not in {"linear", "log", "log10"}:
        raise ValueError(f"unsupported interpolation axis: {axis}")
    if axis != "linear":
        if not np.isfinite(frequency_floor) or frequency_floor <= 0:
            raise ValueError("log frequency floor must be finite and positive")
        logarithm = np.log if axis == "log" else np.log10
        source = logarithm(np.maximum(source, frequency_floor))
        target = logarithm(np.maximum(target, frequency_floor))
    return np.interp(target, source, values, left=values[0], right=values[-1])
