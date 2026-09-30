from __future__ import annotations

from dataclasses import dataclass
import math


MIN_KAISER_ATTENUATION_DB = 21.0
_LINEAR_BRANCH_BETA = 0.1102 * (50.0 - 8.7)


def kaiser_beta_from_attenuation_db(attenuation_db: float) -> float:
    """Convert the Kaiser design attenuation estimate to beta."""
    attenuation = float(attenuation_db)
    if not math.isfinite(attenuation) or attenuation < MIN_KAISER_ATTENUATION_DB:
        raise ValueError("Kaiser attenuation must be finite and at least 21 dB")
    if attenuation > 50.0:
        return 0.1102 * (attenuation - 8.7)
    delta = attenuation - 21.0
    return 0.5842 * delta**0.4 + 0.07886 * delta


def kaiser_attenuation_db_from_beta(beta: float) -> float:
    """Convert beta back to the Kaiser design attenuation estimate."""
    value = float(beta)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Kaiser beta must be finite and non-negative")
    if value == 0.0:
        return MIN_KAISER_ATTENUATION_DB
    if value >= _LINEAR_BRANCH_BETA:
        return value / 0.1102 + 8.7

    low_db, high_db = MIN_KAISER_ATTENUATION_DB, 50.0
    for _ in range(60):
        midpoint = (low_db + high_db) / 2.0
        if kaiser_beta_from_attenuation_db(midpoint) < value:
            low_db = midpoint
        else:
            high_db = midpoint
    return (low_db + high_db) / 2.0


def legacy_kaiser_beta(attenuation_db: float, *, decimals: int = 1) -> float:
    """Return a legacy beta rounded for about +/-0.5 dB compatibility above 50 dB."""
    return round(kaiser_beta_from_attenuation_db(attenuation_db), int(decimals))


def kaiser_crossover_taps(sample_rate: float, frequency_hz: float, cycles: float) -> int:
    estimate = int(
        math.floor(float(sample_rate) / max(float(frequency_hz), 1.0) * float(cycles))
    )
    taps = estimate + 1 + (estimate % 2)
    return max(3, taps)


@dataclass(frozen=True)
class KaiserCycleMatch:
    """Odd-tap equivalent of one reference crossover at another frequency."""

    reference_frequency_hz: float
    target_frequency_hz: float
    reference_cycles: float
    reference_taps: int
    target_taps: int
    target_cycles: float
    target_cycles_min: float
    target_cycles_max: float


def equivalent_kaiser_cycles(
    sample_rate: float,
    reference_frequency_hz: float,
    target_frequency_hz: float,
    reference_cycles: float,
) -> KaiserCycleMatch:
    """Match ``(N - 1) * fc / fs`` while retaining an odd FIR length.

    A Kaiser-windowed sinc has approximately the same transition shape on a
    relative-frequency axis when its effective cycle count is unchanged. The
    nearest realizable target uses an even filter order and therefore odd taps.
    """
    rate = float(sample_rate)
    reference_frequency = float(reference_frequency_hz)
    target_frequency = float(target_frequency_hz)
    cycles = float(reference_cycles)
    values = (rate, reference_frequency, target_frequency, cycles)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("sample rate, frequencies, and cycles must be finite and positive")
    if reference_frequency >= rate / 2.0 or target_frequency >= rate / 2.0:
        raise ValueError("reference and target frequencies must be below Nyquist")

    reference_taps = kaiser_crossover_taps(rate, reference_frequency, cycles)
    reference_order = reference_taps - 1
    ideal_target_order = reference_order * reference_frequency / target_frequency
    target_order = max(2, int(math.floor(ideal_target_order / 2.0 + 0.5)) * 2)
    target_taps = target_order + 1
    target_cycles = target_order * target_frequency / rate
    return KaiserCycleMatch(
        reference_frequency_hz=reference_frequency,
        target_frequency_hz=target_frequency,
        reference_cycles=cycles,
        reference_taps=reference_taps,
        target_taps=target_taps,
        target_cycles=target_cycles,
        target_cycles_min=max(0.0, (target_taps - 2) * target_frequency / rate),
        target_cycles_max=target_taps * target_frequency / rate,
    )


@dataclass(frozen=True)
class KaiserCrossoverPreset:
    id: str
    cycles: float
    attenuation_db: float

    @property
    def beta(self) -> float:
        return kaiser_beta_from_attenuation_db(self.attenuation_db)


KAISER_CROSSOVER_PRESETS = {
    "compact": KaiserCrossoverPreset(
        "compact",
        cycles=3.0,
        attenuation_db=kaiser_attenuation_db_from_beta(6.0),
    ),
    "standard": KaiserCrossoverPreset(
        "standard",
        cycles=4.0,
        attenuation_db=kaiser_attenuation_db_from_beta(8.0),
    ),
    "steep": KaiserCrossoverPreset(
        "steep",
        cycles=5.0,
        attenuation_db=kaiser_attenuation_db_from_beta(10.0),
    ),
}


def kaiser_crossover_preset(preset_id: str) -> KaiserCrossoverPreset:
    try:
        return KAISER_CROSSOVER_PRESETS[str(preset_id)]
    except KeyError as exc:
        raise ValueError(f"unsupported Kaiser crossover preset: {preset_id}") from exc
