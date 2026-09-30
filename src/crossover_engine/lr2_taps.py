"""Numerically qualify LR2 boundary lengths against the shared target response."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import freqz

ERROR_DB = 0.05
TARGET_FLOOR_DB = -80.0
MAX_TAPS = 32767
ALGORITHM_VERSION = "multiway-v2-lr2-auto"


@dataclass(frozen=True)
class LR2TapPlan:
    taps: int
    max_error_db: float
    low_hz: float
    high_hz: float


def _response_error(coefficients, sample_rate, cutoff, side, *, refine=False, order=2, target_floor_db=TARGET_FLOOR_DB):
    # Resolve narrow truncation ripples independently of the design FFT mesh.
    size = 1 << max(17, (len(coefficients) * 32 - 1).bit_length())
    frequency = np.fft.rfftfreq(size, 1.0 / sample_rate)
    delay = (len(coefficients) - 1) / 2
    response = np.fft.rfft(coefficients, size) * np.exp(2j*np.pi*frequency*delay/sample_rate)
    target = 1.0 / (1.0 + (frequency / cutoff)**order)
    floor = 10.0**(target_floor_db / 20)
    lo, hi = 5.0, sample_rate / 2
    if side == "hp":
        response, target = 1-response, 1-target
        lo = max(lo, cutoff*(floor/(1-floor))**(1/order))
    else:
        hi = min(hi, cutoff*(1/floor-1)**(1/order))
    if lo >= hi:
        return 0.0
    mask = (frequency >= lo) & (frequency <= hi)
    grid = frequency[mask]
    errors = np.abs(20*np.log10(np.maximum(np.abs(response[mask]), 1e-20)/target[mask]))

    def error_at(value):
        _, h = freqz(coefficients, worN=np.asarray([value]), fs=sample_rate)
        actual = h[0]*np.exp(2j*np.pi*value*delay/sample_rate)
        aim = 1/(1+(value/cutoff)**order)
        if side == "hp":
            actual, aim = 1-actual, 1-aim
        return float(abs(20*np.log10(max(abs(actual), 1e-20)/aim)))

    maximum = max(float(np.max(errors, initial=0)), error_at(lo), error_at(hi))
    if refine and len(errors) > 2:
        peaks = np.flatnonzero((errors[1:-1] >= errors[:-2]) & (errors[1:-1] >= errors[2:]))+1
        for index in sorted(peaks, key=lambda i: errors[i], reverse=True)[:12]:
            result = minimize_scalar(lambda f: -error_at(f),
                                     bounds=(grid[index-1], grid[index+1]), method="bounded")
            maximum = max(maximum, -float(result.fun))
    return maximum


def lr2_boundary_error(sample_rate, lowpass_hz, highpass_hz, taps, *, refine=False):
    from .filters import _linear_phase_lr2_fir
    low = _linear_phase_lr2_fir(sample_rate, "lp", lowpass_hz, taps)
    high_base = low if lowpass_hz == highpass_hz else _linear_phase_lr2_fir(sample_rate, "lp", highpass_hz, taps)
    return max(_response_error(low, sample_rate, lowpass_hz, "lp", refine=refine),
               _response_error(high_base, sample_rate, highpass_hz, "hp", refine=refine))


@lru_cache(maxsize=128)
def automatic_lr2_taps(sample_rate: int, crossover_hz: float, overlap_oct: float = 0.0) -> LR2TapPlan:
    from .filters import kaiser_overlap_edges_hz
    sample_rate = int(sample_rate)
    lp, hp = kaiser_overlap_edges_hz(crossover_hz, overlap_oct)
    if sample_rate <= 10 or not 0 < min(lp, hp) <= max(lp, hp) < sample_rate/2:
        raise ValueError("LR2自動タップ算出: クロス周波数と重ね合わせをNyquist未満に設定してください。")
    # Empirical 5%-reserve envelope, qualified only for this sample rate and
    # zero overlap. Always validate the realized FIR: reserve is not a proof.
    if sample_rate == 96000 and overlap_oct == 0 and 20 <= crossover_hz <= 20000:
        from .filters import odd_number
        x = crossover_hz / 1000
        correction = max(1.0, ((-0.002812*x + 0.088255)*x - 0.266863)*x + 1)
        taps = odd_number(sample_rate / crossover_hz * 5.3 * correction)
        verified = lr2_boundary_error(sample_rate, lp, hp, taps, refine=True)
        if verified <= ERROR_DB:
            return LR2TapPlan(taps, verified, 5.0, sample_rate/2)
    # Grow the search bracket first, then examine every odd length in it.
    # Ripple error need not be monotonic; we promise a qualified length, not a
    # mathematical global minimum over every possible FIR design.
    previous, candidate = 1, 3
    while True:
        error = lr2_boundary_error(sample_rate, lp, hp, candidate)
        if error <= ERROR_DB:
            for taps in range(previous+2, candidate+1, 2):
                if lr2_boundary_error(sample_rate, lp, hp, taps) <= ERROR_DB:
                    verified = lr2_boundary_error(sample_rate, lp, hp, taps, refine=True)
                    if verified <= ERROR_DB:
                        return LR2TapPlan(taps, verified, 5.0, sample_rate/2)
        if candidate == MAX_TAPS:
            raise ValueError(f"LR2自動タップ算出: {MAX_TAPS:,} taps以内で最大誤差0.05 dBを満たせません。")
        previous, candidate = candidate, min(MAX_TAPS, 2*candidate+1)
