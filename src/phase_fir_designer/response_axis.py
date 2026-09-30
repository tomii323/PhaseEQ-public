from __future__ import annotations

import numpy as np


def finite_response_samples(
    frequency: np.ndarray,
    values: np.ndarray,
    *,
    min_frequency: float | None = None,
    max_frequency: float | None = None,
    include_max: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, sorted, unique response samples with aligned lengths."""

    source_frequency = np.asarray(frequency, dtype=float).ravel()
    source_values = np.asarray(values, dtype=float).ravel()
    length = min(source_frequency.size, source_values.size)
    if length <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    source_frequency = source_frequency[:length]
    source_values = source_values[:length]
    valid = np.isfinite(source_frequency) & np.isfinite(source_values)
    if min_frequency is not None:
        valid &= source_frequency >= float(min_frequency)
    if max_frequency is not None:
        if include_max:
            valid &= source_frequency <= float(max_frequency)
        else:
            valid &= source_frequency < float(max_frequency)
    if not np.any(valid):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    source_frequency = source_frequency[valid]
    source_values = source_values[valid]
    order = np.argsort(source_frequency, kind="stable")
    source_frequency = source_frequency[order]
    source_values = source_values[order]
    unique_frequency, unique_indices = np.unique(source_frequency, return_index=True)
    return unique_frequency, source_values[unique_indices]


def finite_unwrapped_phase_samples(
    frequency: np.ndarray,
    phase_deg: np.ndarray,
    *,
    min_frequency: float | None = None,
    max_frequency: float | None = None,
    include_max: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter invalid phase samples before unwrapping to avoid NaN propagation."""

    source_frequency, source_phase = finite_response_samples(
        frequency,
        phase_deg,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        include_max=include_max,
    )
    if source_phase.size:
        source_phase = np.rad2deg(np.unwrap(np.deg2rad(source_phase)))
    return source_frequency, source_phase
