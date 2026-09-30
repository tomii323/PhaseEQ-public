"""Optional source-edge extrapolation shared by Target previews and processing."""
import re
import numpy as np
from response_completion import complete_response
from .config import SpeakerResponse


def is_wav_target_source(source_name):
    """Recognize both uploaded names and persisted WAV provenance labels."""
    return bool(re.search(r"\.wav(?:$| \(Impulse WAV mono, [^)]+\)$)", str(source_name).strip(), re.IGNORECASE))


def extend_target_response(source, sample_rate, *, lf_enabled=False, hf_enabled=False):
    if source is None or not (lf_enabled or hf_enabled):
        return source
    f = np.asarray(source.frequency, dtype=float)
    g = np.asarray(source.gain_db, dtype=float)
    p = None if source.phase_deg is None else np.asarray(source.phase_deg, dtype=float)
    valid = np.isfinite(f) & np.isfinite(g) & (f >= 0) & (f <= sample_rate / 2)
    if p is not None:
        valid &= np.isfinite(p)
    f, g = f[valid], g[valid]
    if p is not None:
        p = p[valid]
    order = np.argsort(f)
    f, g = f[order], g[order]
    if p is not None:
        p = p[order]
    f, unique = np.unique(f, return_index=True)
    g = g[unique]
    if p is not None:
        p = p[unique]
    positive = np.flatnonzero(f > 0)
    if positive.size < 2:
        return source

    low_f = np.array([], dtype=float)
    low_g = np.array([], dtype=float)
    low_phase = None
    if lf_enabled and f[0] > 2.0:
        n = max(2, int(np.ceil(np.log2(f[0]/2.0)*192))+1)
        low_f = np.geomspace(2.0, f[0], n)[:-1]
        low_g, low_phase, _ = complete_response(f, g, p, low_f)
    high_f = np.array([], dtype=float)
    high_g = np.array([], dtype=float)
    high_phase = None
    if hf_enabled and f[-1] < sample_rate/2:
        n = max(2, int(np.ceil(np.log2(sample_rate/2/f[-1])*192))+1)
        high_f = np.geomspace(f[-1], sample_rate/2, n)[1:]
        # Use the same Hi completion as Response Processing; only append its
        # out-of-band samples so the source and LF extension remain unchanged.
        high_g, high_phase, _ = complete_response(f, g, p, high_f)
    phase = None
    if p is not None:
        if high_phase is None:
            high_phase = np.array([], dtype=float)
        phase = np.r_[[] if low_phase is None else low_phase, p, high_phase].tolist()
    return SpeakerResponse(np.r_[low_f, f, high_f].tolist(), np.r_[low_g, g, high_g].tolist(), phase)
