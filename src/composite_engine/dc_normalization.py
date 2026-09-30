"""Apply PhaseEQ's DC policy to the correction FIR, before band splitting."""
import numpy as np
from fir_design_common import normalize_dc_gain


def normalize_correction_fir(coefficients, *, enabled, target_dc_abs):
    values = np.asarray(coefficients, dtype=float)
    if not enabled or target_dc_abs is None:
        return values
    target = float(target_dc_abs)
    if not np.isfinite(target) or target < 0:
        raise ValueError('FIR target DC magnitude must be finite and nonnegative')
    return normalize_dc_gain(values, target)
