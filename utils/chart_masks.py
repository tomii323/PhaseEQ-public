from __future__ import annotations

import numpy as np


def hold_nyquist_phase_from_previous(phase_values: np.ndarray) -> np.ndarray:
    """Keep the displayed Nyquist phase continuous with the preceding bin."""
    phase = np.asarray(phase_values, dtype=float).copy()
    if phase.size >= 2:
        phase[-1] = phase[-2]
    return phase


def mask_values_by_gain_threshold(
    values: np.ndarray,
    gain_values_db: np.ndarray,
    threshold_db: float | None,
) -> np.ndarray:
    masked_values = np.asarray(values, dtype=float).copy()
    if threshold_db is None:
        return masked_values
    gain = np.asarray(gain_values_db, dtype=float)
    if masked_values.shape != gain.shape:
        return masked_values
    masked_values[~(np.isfinite(gain) & (gain >= float(threshold_db)))] = np.nan
    return masked_values


def mask_phase_by_gain_threshold(
    phase_values: np.ndarray,
    gain_values_db: np.ndarray,
    threshold_db: float | None,
) -> np.ndarray:
    return mask_values_by_gain_threshold(phase_values, gain_values_db, threshold_db)


def mask_phase_series_by_gain_series(
    phase_series: dict[str, np.ndarray],
    gain_series: dict[str, np.ndarray],
    threshold_db: float | None,
) -> dict[str, np.ndarray]:
    if threshold_db is None:
        return phase_series
    masked: dict[str, np.ndarray] = {}
    for name, phase in phase_series.items():
        if name == "frequency":
            masked[name] = phase
            continue
        gain = gain_series.get(name)
        if gain is None:
            masked[name] = phase
            continue
        masked[name] = mask_phase_by_gain_threshold(phase, gain, threshold_db)
    return masked
