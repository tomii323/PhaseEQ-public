"""Unity EQ generation is separate from basic FIR generation and output."""
import numpy as np


def unity_fir_eq(taps: int) -> np.ndarray:
    """An odd, center-referenced unity EQ with an explicit causal delay."""
    count = int(taps)
    if count < 1 or count % 2 == 0:
        raise ValueError("Unity FIR EQ requires a positive odd tap count")
    result = np.zeros(count, dtype=float)
    result[count // 2] = 1.0
    return result
