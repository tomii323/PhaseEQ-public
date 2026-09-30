from __future__ import annotations

from dataclasses import dataclass
import math

from .models import DSPInput, DSPRoute, DSPWay


@dataclass(frozen=True)
class GainPathTrace:
    input_gain_db: float
    route_gain_db: float
    way_gain_db: float
    total_gain_db: float
    input_polarity_invert: bool
    way_polarity_invert: bool
    total_polarity_invert: bool
    muted: bool


def accumulate_gain_path(input_item: DSPInput, route: DSPRoute, way: DSPWay) -> GainPathTrace:
    """Resolve every scalar level and polarity operation on one output path."""

    values = (float(input_item.gain_db), float(route.gain_db), float(way.gain_db))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("gain path contains a non-finite value")
    total_polarity = (
        bool(input_item.polarity_invert)
        ^ bool(way.polarity_invert)
    )
    return GainPathTrace(
        input_gain_db=values[0],
        route_gain_db=values[1],
        way_gain_db=values[2],
        total_gain_db=sum(values),
        input_polarity_invert=bool(input_item.polarity_invert),
        way_polarity_invert=bool(way.polarity_invert),
        total_polarity_invert=total_polarity,
        muted=bool(input_item.muted or route.muted or way.muted or not route.enabled),
    )
