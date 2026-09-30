from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping


@dataclass(frozen=True)
class OutputTiming:
    tap_count: int | None
    additional_delay_samples: float


def resolve_output_timing(
    assigned_taps: Mapping[str, int | None], *, align_output_taps: bool,
) -> dict[str, OutputTiming]:
    """Plan final FIR frames and external delays from resolved assignments.

    An FIR-disabled channel cannot be padded, so it needs an external delay
    even when the FIR-enabled channels use a common frame.
    """
    if any(taps is not None and taps < 1 for taps in assigned_taps.values()):
        raise ValueError("assigned FIR tap counts must be positive or None")
    reference = max((taps or 1 for taps in assigned_taps.values()), default=1)
    result = {}
    for band, taps in assigned_taps.items():
        output_taps = reference if align_output_taps and taps is not None else taps
        padding_delay = (reference - taps) // 2 if align_output_taps and taps is not None else 0
        result[band] = OutputTiming(
            output_taps, (reference - (taps or 1)) / 2.0 - padding_delay,
        )
    return result


def fir_length_difference_delay_samples(
    tap_count: int,
    reference_tap_count: int,
    *,
    align_output_taps: bool,
) -> float:
    """Return only the external DSP delay needed to align FIR lengths.

    The intrinsic linear-phase latency ``(N - 1) / 2`` remains in the FIR
    coefficients and must not be added here a second time.
    """
    taps = int(tap_count)
    reference = int(reference_tap_count)
    if taps < 1 or reference < 1:
        raise ValueError("tap counts must be positive")
    if taps > reference:
        raise ValueError("tap count must not exceed the reference tap count")
    if align_output_taps:
        return 0.0
    return (reference - taps) / 2.0
