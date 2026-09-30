from __future__ import annotations

import numpy as np
from scipy import signal


def make_window(
    name: str,
    taps: int,
    kaiser_beta: float = 12.0,
    chebyshev_attenuation_db: float = 100.0,
    tukey_alpha: float = 0.5,
) -> np.ndarray:
    if name == "rectangular":
        return np.ones(taps, dtype=float)
    if name == "kaiser":
        return signal.windows.kaiser(taps, beta=kaiser_beta, sym=True)
    if name == "dolph_chebyshev":
        return signal.windows.chebwin(taps, at=max(float(chebyshev_attenuation_db), 1.0), sym=True)
    if name == "tukey":
        return signal.windows.tukey(taps, alpha=tukey_alpha, sym=True)
    if name == "cosine_tapered":
        return cosine_tapered_window(taps, taper_ratio=tukey_alpha)
    if name == "hann":
        return signal.windows.hann(taps, sym=True)
    if name == "hamming":
        return signal.windows.hamming(taps, sym=True)
    if name == "blackman":
        return signal.windows.blackman(taps, sym=True)
    if name == "blackmanharris":
        return signal.windows.blackmanharris(taps, sym=True)
    raise ValueError(f"Unsupported window: {name}")


def cosine_tapered_window(taps: int, taper_ratio: float = 0.5) -> np.ndarray:
    if taps <= 0:
        return np.array([], dtype=float)
    taper_ratio = min(max(float(taper_ratio), 0.0), 1.0)
    if taper_ratio <= 0:
        return np.ones(taps, dtype=float)
    if taper_ratio >= 1:
        return signal.windows.hann(taps, sym=True)

    window = np.ones(taps, dtype=float)
    taper_samples = int(np.floor(taper_ratio * (taps - 1) / 2.0))
    if taper_samples <= 0:
        return window

    n = np.arange(taper_samples, dtype=float)
    fade_in = 0.5 - 0.5 * np.cos(np.pi * (n + 1.0) / (taper_samples + 1.0))
    window[:taper_samples] = fade_in
    window[-taper_samples:] = fade_in[::-1]
    return window
