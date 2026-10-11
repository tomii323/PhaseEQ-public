"""Display-only Multiway sampling; no smoothing, DSP or persistence transforms."""
import hashlib

import numpy as np
from .cache import content_cached


# PhaseEQ and Multiway evaluate Gain/Phase on the same arithmetic grid before
# reducing it for a logarithmic display.  These are display-only constants;
# FIR generation, Analysis FFT, and exported coefficients do not depend on them.
DISPLAY_FREQUENCY_RESOLUTION_HZ = 2.0
DISPLAY_GRAPH_POINTS = 2048


def array_content_revision(values) -> str:
    """Return a stable revision for browser-side redraw of numeric arrays."""
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def display_fft_length(sample_rate, *, minimum_length=0):
    """Return an odd FFT length with spacing no coarser than the display policy."""
    fs = int(sample_rate)
    required = int(minimum_length)
    if fs <= 0:
        raise ValueError("sample rate must be positive")
    if required < 0:
        raise ValueError("minimum length must not be negative")
    length = max(
        int(np.ceil(fs / DISPLAY_FREQUENCY_RESOLUTION_HZ)),
        required,
        3,
    )
    return length if length % 2 else length + 1


def display_frequency_axis(sample_rate, *, minimum_length=0):
    """Return the shared linear analysis grid used before logarithmic sampling."""
    length = display_fft_length(sample_rate, minimum_length=minimum_length)
    return np.fft.rfftfreq(length, 1.0 / int(sample_rate))


@content_cached(revision="multiway-indices-v1", max_entries=32, max_bytes=4*1024*1024)
def log_spaced_sample_indices(x_values, max_points):
    x = np.asarray(x_values, dtype=float).reshape(-1)
    limit = max(2, int(max_points))
    if x.size <= limit:
        return np.arange(x.size, dtype=int)
    positive = np.flatnonzero(np.isfinite(x) & (x > 0))
    if positive.size < 2 or np.any(np.diff(x[positive]) < 0):
        return np.linspace(0, x.size-1, limit, dtype=int)
    first, last = int(positive[0]), int(positive[-1])
    targets = np.geomspace(x[first], x[last], limit)
    insertion = np.clip(np.searchsorted(x, targets), first, last)
    previous = np.maximum(insertion-1, first)
    selected = np.where(abs(x[previous]-targets) <= abs(x[insertion]-targets), previous, insertion)
    selected = np.unique(np.concatenate(([0], selected, [x.size-1]))).astype(int)
    if selected.size < limit:
        remaining = np.setdiff1d(np.arange(x.size), selected, assume_unique=True)
        needed = min(limit-selected.size, remaining.size)
        selected = np.unique(np.concatenate((selected, remaining[np.linspace(0,remaining.size-1,needed,dtype=int)])))
    if selected.size > limit:
        selected = selected[np.linspace(0,selected.size-1,limit,dtype=int)]
    return selected


@content_cached(revision="display-series-v1", max_entries=32, max_bytes=16*1024*1024)
def sample_series(x, y, max_points, *, wrapped_phase=False):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("display x/y must be matching one-dimensional arrays")
    indices = log_spaced_sample_indices(x, max_points)
    out = y[indices].copy()
    valid = np.isfinite(x) & np.isfinite(y)
    boundaries = ~valid
    if wrapped_phase and len(y)>1:
        displayed = (y+180)%360-180
        boundaries = boundaries.copy()
        boundaries[1:] |= (np.abs(np.diff(displayed)) > 180) | (np.abs(np.diff(y)) >= 180)
    cumulative = np.cumsum(boundaries)
    if indices.size>1:
        # A dropped mask/wrap boundary must not become a connecting line.
        cut = cumulative[indices[1:]] != cumulative[indices[:-1]]
        out[1:][cut] = np.nan
    return x[indices].copy(), out


def phase_segment_ids(phases):
    phases = np.asarray(phases, dtype=float)
    valid = np.isfinite(phases)
    safe = np.where(valid, phases, 0)
    display = (safe+180)%360-180
    breaks = ~valid
    if len(phases)>1:
        breaks[1:] |= (~valid[:-1]) | (np.abs(np.diff(display))>180) | (np.abs(np.diff(safe))>=180)
    return np.cumsum(breaks), display
