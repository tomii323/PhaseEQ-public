"""Final-output taper shared by PhaseEQ and Composite; no gain normalization."""
import numpy as np
from scipy.signal.windows import tukey

COSINE_TAPER_ALPHA = 0.10


def remove_nyquist_component(fir: np.ndarray, strength: float = 1.0) -> tuple[np.ndarray, float]:
    """Project out the alternating Nyquist basis before the final window."""
    coefficients = np.asarray(fir, dtype=float)
    if coefficients.size == 0:
        return coefficients.copy(), 0.0
    strength = min(max(float(strength), 0.0), 1.0)
    basis = np.where(np.arange(coefficients.size) % 2 == 0, 1.0, -1.0)
    amplitude = float(np.dot(coefficients, basis) / np.dot(basis, basis))
    cleaned = coefficients - strength * amplitude * basis
    return cleaned, amplitude


def output_window(taps: int, enabled: bool = False) -> np.ndarray:
    if taps < 1:
        raise ValueError("FIR tap count must be positive")
    # A two-tap symmetric Tukey window would erase the entire filter.
    if taps <= 2:
        return np.ones(taps)
    return tukey(taps, alpha=COSINE_TAPER_ALPHA, sym=True) if enabled else np.ones(taps)


def apply_output_window(fir: np.ndarray, enabled: bool = False) -> np.ndarray:
    values = np.asarray(fir, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("FIR coefficients must be a finite one-dimensional array")
    return values * output_window(values.size, enabled)
