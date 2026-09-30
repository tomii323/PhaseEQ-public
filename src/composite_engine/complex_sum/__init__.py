from __future__ import annotations

import numpy as np

from ..alignment import apply_delay


def transform_response(response: np.ndarray, frequency_hz: np.ndarray, sample_rate_hz: int, *, gain_db: float = 0.0, polarity: int = 1, delay_samples: float = 0.0) -> np.ndarray:
    transformed = np.asarray(response, dtype=complex) * (10.0 ** (float(gain_db) / 20.0)) * int(polarity)
    return apply_delay(transformed, frequency_hz, sample_rate_hz, delay_samples)


def complex_sum(responses: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
    if not responses:
        raise ValueError("at least one complex response is required")
    arrays = [np.asarray(item, dtype=complex) for item in responses]
    if len({item.shape for item in arrays}) != 1:
        raise ValueError("all complex responses must share one frequency axis")
    return np.sum(arrays, axis=0, dtype=np.complex128)
