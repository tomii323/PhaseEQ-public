"""Cubic LR4 length estimate, qualified against the realized boundary FIR."""
from functools import lru_cache

from .filters import _linear_phase_lr4_fir, kaiser_overlap_edges_hz, odd_number
from .lr2_taps import LR2TapPlan, _response_error, _response_errors

MAX_TAPS = 131071


def estimate_lr4_taps(sample_rate, crossover_hz):
    # Normalized frequency preserves the 96 kHz study's cubic coordinates.
    x = crossover_hz / sample_rate * 96.0
    correction = max(1.0, ((0.003079*x - 0.023119)*x + 0.198385)*x + 1.0)
    return max(3, odd_number(sample_rate / crossover_hz * 9.3 * correction))


def lr4_boundary_error(sample_rate, lowpass_hz, highpass_hz, taps, *, refine=False):
    low = _linear_phase_lr4_fir(sample_rate, "lp", lowpass_hz, taps)
    if lowpass_hz == highpass_hz:
        return max(_response_errors(
            low,
            sample_rate,
            (
                (lowpass_hz, "lp", 4, -100.0),
                (highpass_hz, "hp", 4, -100.0),
            ),
            refine=refine,
        ))
    high = _linear_phase_lr4_fir(sample_rate, "lp", highpass_hz, taps)
    return max(_response_error(low, sample_rate, lowpass_hz, "lp", refine=refine,
                               order=4, target_floor_db=-100.0),
               _response_error(high, sample_rate, highpass_hz, "hp", refine=refine,
                               order=4, target_floor_db=-100.0))


@lru_cache(maxsize=128)
def automatic_lr4_taps(sample_rate: int, crossover_hz: float, overlap_oct: float = 0.0) -> LR2TapPlan:
    lp, hp = kaiser_overlap_edges_hz(crossover_hz, overlap_oct)
    if sample_rate <= 10 or not 0 < min(lp, hp) <= max(lp, hp) < sample_rate/2:
        raise ValueError("LR4自動タップ算出: クロス周波数と重ね合わせをNyquist未満に設定してください。")
    taps = min(MAX_TAPS, max(estimate_lr4_taps(sample_rate, lp), estimate_lr4_taps(sample_rate, hp)))
    while True:
        error = lr4_boundary_error(sample_rate, lp, hp, taps, refine=True)
        if error <= 0.05:
            return LR2TapPlan(taps, error, 5.0, sample_rate/2)
        if taps == MAX_TAPS:
            raise ValueError(f"LR4: {MAX_TAPS:,} taps以内で−100 dB以上・誤差0.05 dB以内を満たせません。")
        # A cubic is an estimate, especially outside the measured rate/range.
        # Never silently accept an unqualified length or assume monotonic error.
        taps = min(MAX_TAPS, odd_number(taps * 1.1))
