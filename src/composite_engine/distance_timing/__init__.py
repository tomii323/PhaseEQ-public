from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


NOMINAL_SOUND_SPEED_M_S = 343.42


@dataclass(frozen=True)
class ExternalDistanceTiming:
    distances_m: dict[str, float]
    arrival_samples: dict[str, float]
    delay_samples: dict[str, float]
    speed_of_sound_m_s: float
    reference_channel: str


def external_distance_timing(
    distances_m: Mapping[str, float],
    *,
    sample_rate_hz: int,
) -> ExternalDistanceTiming:
    """Convert common-reference distances with one nominal sound speed."""
    sample_rate = int(sample_rate_hz)
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    values = {str(channel): float(value) for channel, value in distances_m.items()}
    if not values:
        raise ValueError("At least one external distance is required")
    if not np.isfinite(tuple(values.values())).all() or any(
        value <= 0.0 for value in values.values()
    ):
        raise ValueError("Every Channel requires a finite distance greater than 0 m")
    speed = NOMINAL_SOUND_SPEED_M_S
    arrivals = {
        channel: distance / speed * sample_rate
        for channel, distance in values.items()
    }
    latest = max(arrivals.values())
    delays = {
        channel: float(np.floor((latest - arrival) + 0.5))
        for channel, arrival in arrivals.items()
    }
    reference = max(values, key=values.get)
    return ExternalDistanceTiming(
        distances_m=values,
        arrival_samples=arrivals,
        delay_samples=delays,
        speed_of_sound_m_s=speed,
        reference_channel=reference,
    )


__all__ = [
    "ExternalDistanceTiming",
    "NOMINAL_SOUND_SPEED_M_S",
    "external_distance_timing",
]
