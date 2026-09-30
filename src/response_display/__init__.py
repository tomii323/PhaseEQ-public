"""Display-only Multiway sampling; no smoothing, DSP or persistence transforms."""
import numpy as np
from .cache import content_cached


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
