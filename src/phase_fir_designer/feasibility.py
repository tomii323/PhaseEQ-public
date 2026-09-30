from __future__ import annotations


def fir_duration_ms(taps: int, sample_rate: int) -> float:
    if int(sample_rate) <= 0:
        return 0.0
    return float(taps) * 1000.0 / float(sample_rate)
