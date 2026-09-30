from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from .crossover_kaiser import (
    kaiser_attenuation_db_from_beta,
    kaiser_beta_from_attenuation_db,
    kaiser_crossover_taps,
)


CrossoverCycleRole = Literal["main", "sub"]
MAIN_MIN_CYCLES = 3.0
SUB_MIN_CYCLES = 2.4
SUB_REDUCED_CYCLES_MIN_BETA = 6.0
SUB_REDUCED_CYCLES_MIN_ATTENUATION_DB = kaiser_attenuation_db_from_beta(
    SUB_REDUCED_CYCLES_MIN_BETA
)


@dataclass(frozen=True)
class CrossoverCycleAdjustment:
    role: CrossoverCycleRole
    requested_cycles: float
    effective_cycles: float
    minimum_cycles: float
    frequency_hz: float
    sample_rate: float
    available_taps: int
    required_taps: int
    effective_taps: int
    adjusted: bool
    fits_tap_budget: bool
    attenuation_db: float
    beta: float
    safe: bool
    reason: str


def minimum_crossover_cycles(role: CrossoverCycleRole) -> float:
    if role == "main":
        return MAIN_MIN_CYCLES
    if role == "sub":
        return SUB_MIN_CYCLES
    raise ValueError(f"unsupported crossover cycle role: {role}")


def adjust_kaiser_cycles_for_tap_budget(
    sample_rate: float,
    frequency_hz: float,
    requested_cycles: float,
    available_taps: int,
    *,
    role: CrossoverCycleRole = "main",
    attenuation_db: float = kaiser_attenuation_db_from_beta(8.0),
    decimals: int = 3,
) -> CrossoverCycleAdjustment:
    """Reduce Cycles only when the natural Kaiser FIR exceeds a tap budget.

    The returned Cycles value is the largest value on the requested decimal
    grid that fits. If even the role-specific minimum does not fit, the minimum
    is returned with ``fits_tap_budget=False`` so the caller can choose crop,
    more taps, or a higher crossover frequency explicitly.
    """

    rate = float(sample_rate)
    frequency = float(frequency_hz)
    requested = float(requested_cycles)
    taps_budget = int(available_taps)
    attenuation = float(attenuation_db)
    digits = int(decimals)
    minimum = minimum_crossover_cycles(role)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("sample rate must be finite and positive")
    if not math.isfinite(frequency) or frequency <= 0.0 or frequency >= rate / 2.0:
        raise ValueError("crossover frequency must be finite, positive, and below Nyquist")
    if not math.isfinite(requested) or requested < minimum:
        raise ValueError(f"requested Cycles must be at least {minimum:.1f} for {role}")
    if taps_budget < 3:
        raise ValueError("available taps must be at least 3")
    if digits < 0 or digits > 9:
        raise ValueError("decimals must be between 0 and 9")
    beta = kaiser_beta_from_attenuation_db(attenuation)
    required = kaiser_crossover_taps(rate, frequency, requested)
    if required <= taps_budget:
        effective = requested
        effective_taps = required
        adjusted = False
        fits = True
        reason = "Natural FIR fits the tap budget; Cycles unchanged"
    else:
        minimum_taps = kaiser_crossover_taps(rate, frequency, minimum)
        if minimum_taps > taps_budget:
            effective = minimum
            effective_taps = minimum_taps
            adjusted = not math.isclose(effective, requested)
            fits = False
            reason = "Tap budget is below the role-specific minimum Cycles requirement"
        else:
            step = 10.0 ** (-digits)
            upper_exclusive = taps_budget * frequency / rate
            effective = math.floor(math.nextafter(upper_exclusive, 0.0) / step) * step
            effective = min(requested, max(minimum, effective))
            while effective > minimum and kaiser_crossover_taps(rate, frequency, effective) > taps_budget:
                effective = max(minimum, effective - step)
            effective = round(effective, digits)
            effective_taps = kaiser_crossover_taps(rate, frequency, effective)
            adjusted = not math.isclose(effective, requested, rel_tol=0.0, abs_tol=step / 2.0)
            fits = effective_taps <= taps_budget
            reason = "Cycles reduced to the largest value that fits the tap budget"

    safe = True
    if role == "sub" and effective < MAIN_MIN_CYCLES - 1e-12 and beta < SUB_REDUCED_CYCLES_MIN_BETA:
        safe = False
        reason += "; SUB below 3.0 Cycles requires beta 6.0 or higher"
    return CrossoverCycleAdjustment(
        role=role,
        requested_cycles=requested,
        effective_cycles=effective,
        minimum_cycles=minimum,
        frequency_hz=frequency,
        sample_rate=rate,
        available_taps=taps_budget,
        required_taps=required,
        effective_taps=effective_taps,
        adjusted=adjusted,
        fits_tap_budget=fits,
        attenuation_db=attenuation,
        beta=beta,
        safe=safe,
        reason=reason,
    )


def adjust_linked_kaiser_cycles_for_tap_budget(
    sample_rate: float,
    frequency_hz: float,
    requested_cycles: float,
    *,
    lower_available_taps: int | None,
    upper_available_taps: int | None,
    attenuation_db: float = kaiser_attenuation_db_from_beta(8.0),
    decimals: int = 3,
) -> CrossoverCycleAdjustment:
    """Choose one Cycles value shared by the LP and HP of one boundary.

    The stricter of the two Way tap budgets controls the result.  Applying the
    returned value to both sides keeps the linked LP/HP transition symmetric;
    a larger-budget Way never retains a different Cycles value.
    """

    budgets = [
        int(value)
        for value in (lower_available_taps, upper_available_taps)
        if value is not None
    ]
    if not budgets:
        raise ValueError("at least one linked Way tap budget is required")
    return adjust_kaiser_cycles_for_tap_budget(
        sample_rate,
        frequency_hz,
        requested_cycles,
        min(budgets),
        role="main",
        attenuation_db=attenuation_db,
        decimals=decimals,
    )
