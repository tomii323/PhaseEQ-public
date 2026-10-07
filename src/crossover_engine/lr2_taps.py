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


def _zero_phase_evaluator(coefficients, sample_rate):
    """Return a scalar-frequency evaluator for a centered FIR response.

    Natural-length crossover FIRs are odd and symmetric.  Evaluating their
    real zero-phase cosine series avoids scipy.freqz's general polynomial
    path for every peak-refinement sample.  Keep the general path as a
    fallback so this verifier remains correct for other coefficient shapes.
    """
    coefficients = np.asarray(coefficients)
    delay = (len(coefficients) - 1) / 2
    if len(coefficients) % 2 == 1 and np.array_equal(coefficients, coefficients[::-1]):
        center = len(coefficients) // 2
        offsets = np.arange(1, center + 1)
        weights = 2 * coefficients[center - offsets]

        def evaluate(frequency_hz):
            angle = 2 * np.pi * float(frequency_hz) / sample_rate
            return float(coefficients[center] + np.dot(weights, np.cos(angle * offsets)))

        return evaluate

    def evaluate(frequency_hz):
        _, response = freqz(coefficients, worN=np.asarray([frequency_hz]), fs=sample_rate)
        return response[0] * np.exp(2j * np.pi * frequency_hz * delay / sample_rate)

    return evaluate


def _response_errors(coefficients, sample_rate, specifications, *, refine=False):
    """Evaluate multiple targets against one realized FIR spectrum."""
    # Resolve narrow truncation ripples independently of the design FFT mesh.
    size = 1 << max(17, (len(coefficients) * 32 - 1).bit_length())
    frequency = np.fft.rfftfreq(size, 1.0 / sample_rate)
    delay = (len(coefficients) - 1) / 2
    response = np.fft.rfft(coefficients, size) * np.exp(2j*np.pi*frequency*delay/sample_rate)
    zero_phase_at = _zero_phase_evaluator(coefficients, sample_rate)
    results = []
    for cutoff, side, order, target_floor_db in specifications:
        target = 1.0 / (1.0 + (frequency / cutoff)**order)
        floor = 10.0**(target_floor_db / 20)
        lo, hi = 5.0, sample_rate / 2
        if side == "hp":
            realized, target = 1-response, 1-target
            lo = max(lo, cutoff*(floor/(1-floor))**(1/order))
        else:
            realized = response
            hi = min(hi, cutoff*(1/floor-1)**(1/order))
        if lo >= hi:
            results.append(0.0)
            continue
        mask = (frequency >= lo) & (frequency <= hi)
        grid = frequency[mask]
        errors = np.abs(20*np.log10(np.maximum(np.abs(realized[mask]), 1e-20)/target[mask]))

        def error_at(value):
            actual = zero_phase_at(value)
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
        results.append(maximum)
    return tuple(results)


def _response_error(coefficients, sample_rate, cutoff, side, *, refine=False, order=2, target_floor_db=TARGET_FLOOR_DB):
    return _response_errors(
        coefficients,
        sample_rate,
        ((cutoff, side, order, target_floor_db),),
        refine=refine,
    )[0]


def lr2_boundary_error(sample_rate, lowpass_hz, highpass_hz, taps, *, refine=False):
    from .filters import _linear_phase_lr2_fir
    low = _linear_phase_lr2_fir(sample_rate, "lp", lowpass_hz, taps)
    if lowpass_hz == highpass_hz:
        return max(_response_errors(
            low,
            sample_rate,
            (
                (lowpass_hz, "lp", 2, TARGET_FLOOR_DB),
                (highpass_hz, "hp", 2, TARGET_FLOOR_DB),
            ),
            refine=refine,
        ))
    high_base = _linear_phase_lr2_fir(sample_rate, "lp", highpass_hz, taps)
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
