from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import signal

from .crossover_kaiser import kaiser_crossover_taps
from .crossover_models import CrossoverFilterSpec


def clip_crossover_frequency(frequency_hz: float, sample_rate: float) -> float:
    nyquist = float(sample_rate) / 2.0
    return float(min(max(float(frequency_hz), 1.0), nyquist - 1.0))


def center_or_crop_crossover_fir(fir: np.ndarray, target_taps: int) -> np.ndarray:
    values = np.asarray(fir, dtype=np.float64).reshape(-1)
    target = max(1, int(target_taps))
    if values.size == target:
        return values.copy()
    if values.size < target:
        total = target - values.size
        left = total // 2
        return np.pad(values, (left, total - left))
    start = (values.size - target) // 2
    return values[start : start + target].copy()


def centered_delta(taps: int) -> np.ndarray:
    values = np.zeros(max(1, int(taps)), dtype=np.float64)
    values[values.size // 2] = 1.0
    return values


def design_kaiser_crossover_fir(
    sample_rate: int,
    mode: str,
    frequency_hz: float,
    spec: CrossoverFilterSpec,
    *,
    tap_reference_frequency_hz: float | None = None,
) -> np.ndarray:
    if spec.engine != "kaiser_fir":
        raise ValueError("Kaiser FIR design requires engine='kaiser_fir'")
    if mode not in {"hp", "lp"}:
        raise ValueError("crossover FIR mode must be hp or lp")
    spec.validate()
    tap_reference = frequency_hz if tap_reference_frequency_hz is None else tap_reference_frequency_hz
    taps = kaiser_crossover_taps(sample_rate, tap_reference, spec.cycles)
    cutoff = clip_crossover_frequency(frequency_hz, sample_rate)
    return signal.firwin(
        taps,
        cutoff,
        window=("kaiser", float(spec.beta)),
        pass_zero=(mode == "lp"),
        scale=True,
        fs=float(sample_rate),
    ).astype(np.float64, copy=False)


def design_direct_low_pass(
    sample_rate: int,
    frequency_hz: float,
    overlap_hz: float,
    spec: CrossoverFilterSpec,
) -> np.ndarray:
    return design_kaiser_crossover_fir(
        sample_rate,
        "lp",
        frequency_hz + overlap_hz,
        spec,
        tap_reference_frequency_hz=frequency_hz,
    )


def design_direct_high_pass(
    sample_rate: int,
    frequency_hz: float,
    overlap_hz: float,
    spec: CrossoverFilterSpec,
) -> np.ndarray:
    return design_kaiser_crossover_fir(
        sample_rate,
        "hp",
        frequency_hz - overlap_hz,
        spec,
        tap_reference_frequency_hz=frequency_hz,
    )


def design_complement_band(
    sample_rate: int,
    lower_frequency_hz: float,
    upper_frequency_hz: float,
    lower_overlap_hz: float,
    upper_overlap_hz: float,
    high_pass_spec: CrossoverFilterSpec,
    low_pass_spec: CrossoverFilterSpec,
) -> np.ndarray:
    """Generate an interior band as delta minus its two outer complements."""
    lower_complement = design_kaiser_crossover_fir(
        sample_rate,
        "lp",
        lower_frequency_hz - lower_overlap_hz,
        high_pass_spec,
        tap_reference_frequency_hz=lower_frequency_hz,
    )
    upper_complement = design_kaiser_crossover_fir(
        sample_rate,
        "hp",
        upper_frequency_hz + upper_overlap_hz,
        low_pass_spec,
        tap_reference_frequency_hz=upper_frequency_hz,
    )
    target = max(lower_complement.size, upper_complement.size)
    lower_complement = center_or_crop_crossover_fir(lower_complement, target)
    upper_complement = center_or_crop_crossover_fir(upper_complement, target)
    return centered_delta(target) - lower_complement - upper_complement


def align_crossover_firs(
    firs: Mapping[str, np.ndarray | None],
    *,
    target_taps: int | None = None,
) -> dict[str, np.ndarray]:
    """Center every FIR on one common length; None is an all-pass delta."""
    longest = max((np.asarray(value).size for value in firs.values() if value is not None), default=1)
    target = max(1, int(target_taps if target_taps is not None else longest))
    return {
        way_id: center_or_crop_crossover_fir(
            centered_delta(1) if fir is None else np.asarray(fir, dtype=np.float64),
            target,
        )
        for way_id, fir in firs.items()
    }
