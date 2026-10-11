"""Final-output taper shared by PhaseEQ and Composite; no gain normalization."""
import numpy as np
from scipy.signal.windows import tukey

COSINE_TAPER_ALPHA = 0.10


def output_window(
    taps: int,
    enabled: bool = False,
    *,
    protected_center_taps: int | None = None,
) -> np.ndarray:
    if taps < 1:
        raise ValueError("FIR tap count must be positive")
    if protected_center_taps is not None:
        protected = int(protected_center_taps)
        if not 0 < protected <= taps:
            raise ValueError("protected center taps must fit in the FIR frame")
        padding = taps - protected
        if padding % 2:
            raise ValueError("protected center frame must preserve tap-center parity")
        if not enabled or padding == 0:
            return np.ones(taps)
        side = padding // 2
        # The added taps are a guard region, not a longer Kaiser design.  End
        # each cosine ramp immediately before the frozen center coefficients
        # so the one generated Kaiser FIR is multiplied by exactly 1.0.
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, side + 1)) ** 2
        window = np.ones(taps)
        window[:side] = ramp[:-1]
        window[-side:] = ramp[:-1][::-1]
        return window
    # A two-tap symmetric Tukey window would erase the entire filter.
    if taps <= 2:
        return np.ones(taps)
    return tukey(taps, alpha=COSINE_TAPER_ALPHA, sym=True) if enabled else np.ones(taps)


def apply_output_window(
    fir: np.ndarray,
    enabled: bool = False,
    *,
    protected_center_taps: int | None = None,
) -> np.ndarray:
    values = np.asarray(fir, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("FIR coefficients must be a finite one-dimensional array")
    return values * output_window(
        values.size,
        enabled,
        protected_center_taps=protected_center_taps,
    )
