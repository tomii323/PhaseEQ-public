"""Shared numerical contract for user-configurable Kaiser FIR parameters."""

from __future__ import annotations


KAISER_BETA_MIN = 7.0
KAISER_BETA_MAX = 16.0
KAISER_CYCLES_MIN = 2.7
KAISER_CYCLES_MAX = 15.0


def kaiser_beta_in_range(value: float) -> bool:
    """Return whether a beta value satisfies the shared product contract."""
    return KAISER_BETA_MIN <= float(value) <= KAISER_BETA_MAX


def clamp_kaiser_beta(value: float) -> float:
    """Normalize a persisted beta value to the shared supported range."""
    return min(max(float(value), KAISER_BETA_MIN), KAISER_BETA_MAX)


def kaiser_cycles_in_range(value: float) -> bool:
    """Return whether a cycle count satisfies the shared product contract."""
    return KAISER_CYCLES_MIN <= float(value) <= KAISER_CYCLES_MAX


def clamp_kaiser_cycles(value: float) -> float:
    """Normalize a persisted cycle count to the shared supported range."""
    return min(max(float(value), KAISER_CYCLES_MIN), KAISER_CYCLES_MAX)
