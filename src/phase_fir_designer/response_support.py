"""Runtime response support, independent of persisted spectral samples.

The annotation is deliberately outside dataclass fields: settings keep their
existing schema. Derived responses explicitly inherit support at processing
boundaries; deepcopy and the process-local content cache retain the annotation.
"""
from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class ResponseSupport:
    kind: str
    intervals: tuple[tuple[float, float], ...]
    resolution_hz: float | None = None

    def validity(self, frequency):
        axis = np.asarray(frequency, float)
        mask = np.zeros(axis.shape, bool)
        for lo, hi in self.intervals:
            mask |= (axis >= lo) & (axis <= hi)
        return mask & np.isfinite(axis) & (axis > 0)

    @property
    def signature(self):
        return hashlib.sha256(json.dumps(
            [self.kind, self.intervals, self.resolution_hz], allow_nan=False,
        ).encode()).hexdigest()


def support_for(response):
    annotated = getattr(response, '_evaluation_support', None)
    if annotated is not None:
        return annotated
    f, g = np.asarray(response.frequency, float), np.asarray(response.gain_db, float)
    usable = np.isfinite(f) & (f > 0)
    f, g = f[usable], g[usable]
    if len(f) < 2 or np.any(np.diff(f) <= 0):
        return ResponseSupport('discrete', ())
    hz_steps = np.diff(f)
    resolution = float(np.median(hz_steps))
    # A regular Hz grid describes FFT sampling, not logarithmic display rows.
    # Missing bins remain gaps; one DC-adjacent placeholder is not a bin.
    regular = np.isclose(hz_steps, resolution, rtol=1e-6, atol=1e-9)
    linear = len(f) >= 8 and np.mean(regular) >= .9
    if linear:
        accepted = regular.copy()
        if f[0] <= 2.0 and len(accepted) > 1 and accepted[1]:
            accepted[0] = True
    else:
        steps = np.diff(np.log2(f))
        accepted = steps <= max(1/6, 4 * float(np.median(steps)))
    accepted &= np.isfinite(g[:-1]) & np.isfinite(g[1:])
    intervals = []
    for i in np.flatnonzero(accepted):
        lo, hi = float(f[i]), float(f[i+1])
        if intervals and intervals[-1][1] == lo:
            intervals[-1] = (intervals[-1][0], hi)
        else:
            intervals.append((lo, hi))
    return ResponseSupport('linear_fft' if linear else 'discrete', tuple(intervals),
                           resolution if linear else None)


def with_support(response, support):
    if response is None:
        return None
    result = replace(response)
    object.__setattr__(result, '_evaluation_support', support)
    return result


def inherit_support(response, source, *, extend=False):
    if response is None or source is None:
        return response
    if response is source and not extend:
        return response
    support = support_for(source)
    if extend and support.intervals:
        f = np.asarray(response.frequency, float)
        f = f[np.isfinite(f) & (f > 0)]
        intervals = list(support.intervals)
        if f.size:
            intervals[0] = (min(float(f[0]), intervals[0][0]), intervals[0][1])
            intervals[-1] = (intervals[-1][0], max(float(f[-1]), intervals[-1][1]))
            support = replace(support, intervals=tuple(intervals))
    return response if support_for(response) == support else with_support(response, support)


def generated_support(response):
    f = np.asarray(response.frequency, float)
    f = f[np.isfinite(f) & (f > 0)]
    return with_support(response, ResponseSupport('generated',
        ((float(f[0]), float(f[-1])),) if f.size >= 2 else ()))


def evaluate_response(response, frequency):
    from .iir import _response_gain_on_log_axis
    gain = _response_gain_on_log_axis(response, frequency)
    native = support_for(replace(response)).validity(frequency)
    return gain, support_for(response).validity(frequency) & native & np.isfinite(gain)
