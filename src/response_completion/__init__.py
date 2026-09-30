"""Shared Speaker boundary estimation; independent of FFT, UI and design."""
from dataclasses import dataclass, replace

import numpy as np

VERSION = "speaker-boundary-v1"


@dataclass(frozen=True)
class CompletionInfo:
    input_min_hz: float
    input_max_hz: float
    hf_near_slope: float | None
    hf_tail_slope: float | None
    hf_fallback: bool = False


def _regression(x, y, width):
    mask = x >= x[-1] - width
    if np.count_nonzero(mask) < 3:
        return None
    t, y = x[mask] - x[-1], y[mask]
    base = np.gradient(t)
    base /= base.sum()
    weights = base.copy()
    for _ in range(4):
        center, level = np.average(t, weights=weights), np.average(y, weights=weights)
        dx = t - center
        value = np.sum(weights * dx * (y - level)) / np.sum(weights * dx * dx)
        residual = y - (level + value * dx)
        scale = max(1.4826 * np.median(abs(residual - np.median(residual))), 1e-9)
        weights = base * np.minimum(1., 1.345 * scale / np.maximum(abs(residual), 1e-12))
    return float(value)


def _edge_slope(x, y, low):
    x, y = (x[:8], y[:8]) if low else (x[-8:], y[-8:])
    if len(x) < 3:
        return None
    dx = x - x.mean()
    denominator = np.dot(dx, dx)
    return float(np.dot(dx, y-y.mean()) / denominator) if denominator > 0 else None


def complete_response(frequency, gain_db, phase_deg, target_frequency, *, gain_axis="linear", phase_is_continuous=False):
    """Preserve measured knots; low Hermite amplitude, high integrated slope.

    Missing phase remains None. Sparse HF fits hold amplitude (explicit info),
    rather than throwing away a valid sparse measurement or inventing resonance.
    """
    x, gain, f = np.asarray(frequency, float), np.asarray(gain_db, float), np.asarray(target_frequency, float)
    if (x.ndim != 1 or len(x) < 2 or gain.shape != x.shape or not np.isfinite(x+gain).all()
            or np.any(x < 0) or np.any(np.diff(x) <= 0)
            or f.ndim != 1 or not np.isfinite(f).all() or np.any(f < 0)):
        raise ValueError("補完には有限な昇順の正周波数・Gainが2点以上必要です。")
    phase = None if phase_deg is None else np.asarray(phase_deg, float)
    if phase is not None and (phase.shape != x.shape or not np.isfinite(phase).all()):
        raise ValueError("補完の位相は周波数と同じ点数の有限値、または位相なしである必要があります。")
    from response_math import interpolate_values
    out = interpolate_values(x, gain, f, axis=gain_axis)
    low, high = f < x[0], f > x[-1]
    if np.any(low):
        amplitude = 10 ** (gain[:8]/20)
        slope = _edge_slope(x, 10 ** (gain/20), True)
        a0 = amplitude[0]
        d = 2. if slope is None or a0 <= 1e-12 else float(np.clip(x[0]*slope/a0, 0, 3))
        u = f[low]/x[0]
        a = a0*((3-d)*u*u+(d-2)*u*u*u)
        with np.errstate(divide="ignore"):
            out[low] = 20*np.log10(a)
    s0 = s1 = None
    fallback = False
    if np.any(high):
        positive = x > 0
        logx = np.log2(x[positive])
        s0, s1 = _regression(logx, gain[positive], 1/6), _regression(logx, gain[positive], .5)
        fallback = s0 is None or s1 is None
        if fallback:
            s0 = s1 = 0.
        s1 = float(np.clip(s1, -24, 0))
        z = np.log2(f[high]/x[-1])
        length = 1/3
        u = np.clip(z/length, 0, 1)
        integral = length*(u**6-3*u**5+2.5*u**4)+np.maximum(z-length, 0)
        out[high] = gain[-1]+s0*z+(s1-s0)*integral
    output_phase = None
    if phase is not None:
        p = phase if phase_is_continuous else np.rad2deg(np.unwrap(np.deg2rad(phase)))
        output_phase = np.interp(f, x, p)
        for is_low, mask, edge in ((True, low, 0), (False, high, -1)):
            if mask.any():
                slope = _edge_slope(x, p, is_low)
                if slope is None:
                    slope = (p[-1]-p[0])/(x[-1]-x[0])
                output_phase[mask] = p[edge]+slope*(f[mask]-x[edge])
        # P0 receives continuous phase. Unwrap changes only the 360-degree
        # representation, not the complex values or the time origin.
    if not np.isfinite(out[f > 0]).all():
        raise ValueError("補完結果が有限値ではありません。")
    return out, output_phase, CompletionInfo(float(x[0]), float(x[-1]), s0, s1, fallback)


def complete_speaker(speaker, *, sample_rate, minimum_hz=2.):
    """P0 adapter: append positive samples only, keeping FRD/DC contracts apart."""
    if speaker is None:
        return None
    x = np.asarray(speaker.frequency, float)
    if np.any(x < 0):
        raise ValueError("Speaker補完の周波数は0以上が必要です。")
    if np.any(x == 0):
        positive = x > 0
        speaker = replace(speaker, frequency=x[positive].tolist(),
            gain_db=np.asarray(speaker.gain_db)[positive].tolist(),
            phase_deg=None if speaker.phase_deg is None else np.asarray(speaker.phase_deg)[positive].tolist())
        x = x[positive]
    if len(x) < 2:
        raise ValueError("Speaker補完には測定点が2点以上必要です。")
    if sample_rate <= 0 or minimum_hz <= 0:
        raise ValueError("補完のサンプリングレートと最低周波数は正数が必要です。")
    # App input adapter owns exclusion above Nyquist. Do not discard source
    # knots here: malformed direct calls must be visible to their caller.
    if np.max(x) > sample_rate/2:
        raise ValueError("Speaker補完の入力にNyquistを超える点があります。")
    added = []
    for start, end in ((minimum_hz, x[0]), (x[-1], sample_rate/2)):
        if start < end:
            count = max(2, int(np.ceil(np.log2(end/start)*192))+1)
            added.extend(np.geomspace(start, end, count))
    axis = np.unique(np.r_[x, added])
    gain, phase, _ = complete_response(x, speaker.gain_db, speaker.phase_deg, axis)
    return replace(speaker, frequency=axis.tolist(), gain_db=gain.tolist(),
                   phase_deg=None if phase is None else phase.tolist())
