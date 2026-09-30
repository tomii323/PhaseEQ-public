"""Fit-only magnitude preparation; never modifies measurement or output gain."""
from dataclasses import dataclass, asdict
import hashlib
import json

import numpy as np
from scipy.signal import find_peaks


SHAPING_VERSION = 2


@dataclass(frozen=True)
class InputShapingSettings:
    # API/legacy callers retain the old fitter unless explicitly opted in.
    dip_enabled: bool = False
    smoothing_enabled: bool = False
    depth_db: float = 3.0
    ratio: float = 9.0
    knee_db: float = 3.0
    max_width_oct: float = 1 / 3
    low_boundary_hz: float = 200.0
    high_boundary_hz: float = 2000.0
    low_oct: float = 1 / 12
    mid_oct: float = 1 / 6
    high_oct: float = 1 / 3

    @property
    def enabled(self):
        return self.dip_enabled or self.smoothing_enabled

    def validate(self):
        values = [v for k, v in asdict(self).items() if not k.endswith('enabled')]
        if (not np.all(np.isfinite(values)) or not 3 <= self.depth_db <= 18
                or not 1 <= self.ratio <= 20 or not 0 <= self.knee_db <= min(12, 2*self.depth_db)
                or not 1/12 <= self.max_width_oct <= .5
                or not 0 < self.low_boundary_hz < self.high_boundary_hz
                or any(not 0 <= v <= .5 for v in (self.low_oct, self.mid_oct, self.high_oct))):
            raise ValueError('Invalid Auto EQ input shaping settings')


SHAPING_UI_DEFAULTS = {'auto_iir_shape_' + k: v for k, v in asdict(
    InputShapingSettings(dip_enabled=True, smoothing_enabled=True)).items()}


def input_signature(speaker, target, sample_rate, manual):
    digest = hashlib.sha256()
    for response in (speaker, target):
        for values in (response.frequency, response.gain_db):
            data = np.asarray(values, dtype='<f8')
            digest.update(str(data.shape).encode())
            digest.update(data.tobytes())
    digest.update(json.dumps([sample_rate, [asdict(f) for f in manual]], sort_keys=True).encode())
    return digest.hexdigest()


def runs(mask):
    ids = np.flatnonzero(mask)
    return np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1) if ids.size else []


def smooth_segment(values, x, width):
    """Gaussian FWHM in octaves, renormalized at data ends (no extrapolation)."""
    if width == 0 or len(x) < 3:
        return values.copy()
    sigma = width / 2.355
    result = np.empty_like(values)
    for i, center in enumerate(x):
        ids = np.flatnonzero(abs(x - center) <= 3 * sigma)
        weights = np.exp(-.5 * ((x[ids] - center) / sigma)**2)
        result[i] = np.dot(values[ids], weights) / weights.sum()
    return result


def compressed_dip_depth(depth, threshold, ratio, knee):
    """Static soft-knee compressor on positive dip depth (dB domain).

    Unity slope before the knee, 1/ratio after it, continuous value and first
    derivative at both joins. No attack/release: this is a frequency trace.
    """
    depth = np.maximum(np.asarray(depth, dtype=float), 0)
    if knee <= 0:
        return np.where(depth <= threshold, depth, threshold+(depth-threshold)/ratio)
    lower, upper = threshold-knee/2, threshold+knee/2
    middle = depth+(1/ratio-1)*np.square(depth-lower)/(2*knee)
    return np.where(depth < lower, depth,
                    np.where(depth > upper, threshold+(depth-threshold)/ratio, middle))


def _limit_lift(lift, available):
    """Smoothly approach Target instead of introducing a second hard corner."""
    cap = np.maximum(available, 0)
    width = np.minimum(1., cap)
    join = cap-width
    excess = np.maximum(lift-join, 0)
    return np.where(lift > join, cap-width*np.exp(-excess/np.maximum(width, 1e-15)), lift)


def source_validity(response, frequency):
    """Do not bridge missing samples or unusually large source log gaps."""
    f, g = np.asarray(response.frequency, float), np.asarray(response.gain_db, float)
    valid = np.zeros(len(frequency), bool)
    usable = np.isfinite(f) & (f > 0)
    f, g = f[usable], g[usable]
    if len(f) < 2 or np.any(np.diff(f) <= 0):
        return valid
    steps = np.diff(np.log2(f))
    max_step = max(1/6, 4 * float(np.median(steps)))
    for i in range(len(f)-1):
        if np.isfinite(g[i:i+2]).all() and steps[i] <= max_step:
            valid |= (frequency >= f[i]) & (frequency <= f[i+1])
    return valid


def shape_error(frequency, error, valid, settings, *, speaker_trace=None):
    """Return fixed shaped error, lift and detected dips on a uniform log grid."""
    settings.validate()
    x = np.log2(frequency)
    shaped = np.array(error, dtype=float, copy=True)
    trace = np.asarray(error if speaker_trace is None else speaker_trace, dtype=float)
    if trace.shape != shaped.shape:
        raise ValueError('Input shaping trace must match the frequency axis')
    lift = np.zeros_like(shaped)
    dips = []
    for ids in runs(valid & np.isfinite(error) & np.isfinite(trace)):
        if len(ids) < 3:
            continue
        xx, yy = x[ids], np.asarray(error)[ids]
        measured = trace[ids]
        local_lift = np.zeros_like(yy)
        if settings.dip_enabled:
            for peak in find_peaks(-measured)[0]:
                # Far shoulders resist a single adjacent peak. The complete
                # window is required, so monotonic ends are never filled.
                left = np.flatnonzero((xx >= xx[peak]-.5) & (xx <= xx[peak]-.25))
                right = np.flatnonzero((xx >= xx[peak]+.25) & (xx <= xx[peak]+.5))
                if (len(left) < 3 or len(right) < 3
                        or np.ptp(xx[left]) < 1/12 or np.ptp(xx[right]) < 1/12):
                    continue
                baseline = np.interp(xx, [np.median(xx[left]), np.median(xx[right])],
                                     [np.median(measured[left]), np.median(measured[right])])
                depth = baseline - measured
                if depth[peak] <= settings.depth_db-settings.knee_db/2:
                    continue
                half = next((r for r in runs(depth >= depth[peak]/2) if peak in r), None)
                if half is None or half[0] == 0 or half[-1] == len(xx)-1:
                    continue
                first, last = half[0], half[-1]
                crossing = depth[peak]/2
                left_cross = np.interp(crossing, depth[[first-1, first]], xx[[first-1, first]])
                right_cross = np.interp(crossing, depth[[last+1, last]], xx[[last+1, last]])
                half_width = float(right_cross-left_cross)
                if half_width > settings.max_width_oct:
                    continue
                # Require recovery toward the local trend on both sides.
                support = next(r for r in runs(depth > 0.5) if peak in r)
                if support[0] == 0 or support[-1] == len(xx)-1:
                    continue
                soft = depth[support] - compressed_dip_depth(depth[support],
                    settings.depth_db, settings.ratio, settings.knee_db)
                # Shaping removes boost demand; it must never manufacture a
                # cut when the original input already meets/exceeds Target.
                soft = _limit_lift(soft, -yy[support])
                if not np.any(soft > 0):
                    continue
                local_lift[support] = np.maximum(local_lift[support], soft)
                dips.append({'frequency_hz': float(frequency[ids[peak]]),
                             'depth_db': float(depth[peak]),
                             'input_gain_db': float(measured[peak]),
                             'reference_gain_db': float(baseline[peak]),
                             'lift_db': float(local_lift[peak]),
                             'width_oct': half_width})
        lift[ids] = local_lift
        adjusted = yy + local_lift
        if settings.smoothing_enabled:
            low = np.clip((xx-np.log2(settings.low_boundary_hz)+1/6)/(1/3), 0, 1)
            high = np.clip((xx-np.log2(settings.high_boundary_hz)+1/6)/(1/3), 0, 1)
            low, high = low*low*(3-2*low), high*high*(3-2*high)
            weights = (1-low, low*(1-high), low*high)
            adjusted = sum(w*smooth_segment(adjusted, xx, width) for w, width in
                           zip(weights, (settings.low_oct, settings.mid_oct, settings.high_oct)))
        shaped[ids] = adjusted
    return shaped, lift, dips


def prepare_speaker_input(speaker, target, sample_rate, shaping, *, f_min=0.,
                          f_max=float("inf"), reliable_f_min_hz=None,
                          reliable_f_max_hz=None, require_complete=True):
    """Generate a fit-only SpeakerResponse; no filter design or output edits.

    Dips are detected on Speaker itself. Target is used only as a ceiling for
    lifting and a reference for smoothing, preserving intended crossover slopes.
    """
    from .config import SpeakerResponse
    from .iir import AutoIIRSettings, _auto_reliable_range, _response_gain_on_log_axis

    if not shaping.enabled:
        return speaker, []
    shaping.validate()
    common = AutoIIRSettings(reliable_f_min_hz=reliable_f_min_hz,
                            reliable_f_max_hz=reliable_f_max_hz)
    lo, hi = _auto_reliable_range(speaker, target, sample_rate, common)
    if hi <= lo or min(hi, f_max) <= max(lo, f_min):
        raise ValueError("No common Input/Target data in the requested correction band")
    # Twice the original shaping resolution; fitting still uses the unchanged
    # legacy grid and optimizer. Only this preparation step receives more detail.
    frequency = np.geomspace(lo, hi, max(3, min(8192, int(np.ceil(np.log2(hi/lo)*192))+1)))
    source = _response_gain_on_log_axis(speaker, frequency)
    target_gain = _response_gain_on_log_axis(target, frequency)
    valid = source_validity(speaker, frequency) & source_validity(target, frequency)
    requested = (frequency >= f_min) & (frequency <= f_max)
    if require_complete and (not np.all(valid[requested]) or np.count_nonzero(valid & requested) < 3):
        # The existing fitter interpolates through gaps. Do not silently fill
        # missing measurements merely to feed it a complete prepared response.
        raise ValueError('Insufficient common Input/Target data for input shaping')
    error, _lift, dips = shape_error(frequency, source-target_gain, valid, shaping,
                                   speaker_trace=source)
    # Transfer only the adjustment back to the original sampling. Regridding
    # the entire measured curve would change even an already matching input.
    original_f = np.asarray(speaker.frequency, float)
    original_gain = np.asarray(speaker.gain_db, float).copy()
    in_context = np.isfinite(original_f) & (original_f >= lo) & (original_f <= hi)
    original_gain[in_context] += np.interp(np.log2(original_f[in_context]), np.log2(frequency),
                                         error-(source-target_gain))
    prepared = SpeakerResponse(list(speaker.frequency), original_gain.tolist(),
                               None if speaker.phase_deg is None else list(speaker.phase_deg))
    return prepared, dips
