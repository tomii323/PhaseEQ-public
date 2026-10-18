"""NTM4 analog prototype and prewarped SOS, independent of application state."""
from functools import lru_cache

import numpy as np
from scipy import signal
from scipy.optimize import brentq

DEFAULT_LOBE_DB = -36.0
MIN_LOBE_DB = -80.0
MAX_LOBE_DB = -20.0


def validate_lobe_db(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not MIN_LOBE_DB <= value <= MAX_LOBE_DB:
        raise ValueError("NTM4 Lobe Height must be within -80..-20 dB")
    return value


@lru_cache(maxsize=601)
def k_squared(lobe_db: float = DEFAULT_LOBE_DB) -> float:
    """Solve the post-notch LP peak (HP is reciprocal in analog frequency)."""
    amplitude = 10.0 ** (validate_lobe_db(lobe_db) / 20.0)

    def peak(t):
        d = np.sqrt(1.0 - t * t)
        # Stable form of (1-d)/(2*d), including small requested lobes.
        return t * t / (2.0 * d * (1.0 + d))

    return float(brentq(lambda t: peak(t) - amplitude, 1e-9, 0.99, xtol=1e-15))


def allpass_q(lobe_db: float = DEFAULT_LOBE_DB) -> float:
    return float(1.0 / np.sqrt(2.0 * (1.0 - k_squared(lobe_db))))


def ntm4_sos(kind: str, frequency_hz: float, sample_rate_hz: int,
             lobe_db: float = DEFAULT_LOBE_DB) -> np.ndarray:
    """Return two LP/HP Biquads or one sum-AP using the same prototype."""
    fc, fs = float(frequency_hz), float(sample_rate_hz)
    if not np.isfinite(fs) or not fs > 0 or not np.isfinite(fc) or not 0 < fc < fs / 2:
        raise ValueError("NTM4 frequency must be within 0..Nyquist")
    t = k_squared(lobe_db)
    if kind == "allpass":
        from .alignment import AllPassSection, allpass_sos
        return allpass_sos(AllPassSection(fc, allpass_q(lobe_db)), sample_rate_hz)
    alpha = np.sqrt(2.0 * (1.0 - t))
    # Transform each second-order factor separately to avoid fourth-order
    # polynomial root conditioning at low Fc / high sample rates.
    warp = 2.0 * fs * np.tan(np.pi * fc / fs)
    denominator = [1.0, alpha * warp, warp * warp]
    numerators = {
        "lowpass": ([t, 0.0, warp * warp], [0.0, 0.0, warp * warp]),
        "highpass": ([1.0, 0.0, t * warp * warp], [1.0, 0.0, 0.0]),
    }
    if kind not in numerators:
        raise ValueError("NTM4 kind must be lowpass, highpass or allpass")
    rows = []
    for numerator in numerators[kind]:
        b, a = signal.bilinear(numerator, denominator, fs=fs)
        rows.append(np.r_[np.pad(b, (0, 3-len(b))), a])
    return np.asarray(rows)
