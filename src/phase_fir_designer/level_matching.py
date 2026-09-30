"""Match measurement level to a fixed target on a shared, reliable passband."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

TARGET_LEVEL_MODE = "Targetに自動で合わせる"


@dataclass(frozen=True)
class LevelMatch:
    shift_db: float
    low_hz: float
    high_hz: float
    residual_db: float
    message: str = ""
    method: str = "plateau"

    @property
    def valid(self):
        return not self.message


def _stable_reference_band(f, input_db, target_db, input_peak):
    """Find a high measured band with a stable relative level, allowing ripple.

    A rolling slope test is less brittle than requiring every derivative to
    be flat. Absolute level and slope gates still exclude noise/stopbands;
    the upper-level preference keeps broad dips from defining the offset.
    """
    x = np.log2(f)
    source = median_filter(input_db, size=9, mode="nearest")
    target = median_filter(target_db, size=9, mode="nearest")
    high = source >= input_peak - 6.0
    if f[-1] >= 80.0 * 2 ** (1.0 / 3.0):
        high &= f >= 80.0
    if not np.any(high):
        return None
    target_peak = float(np.percentile(target[high], 98))
    best = None
    for width in (1.0, 0.5, 1.0 / 3.0):
        for start in range(len(f)):
            end = int(np.searchsorted(x, x[start] + width))
            if end >= len(f):
                break
            sl = slice(start, end + 1)
            a, b = source[sl], target[sl]
            if (np.mean(high[sl]) < 0.9
                    or np.mean(b >= target_peak - 6.0) < 0.9):
                continue
            axis = x[sl] - np.mean(x[sl])
            denominator = float(np.dot(axis, axis))
            sa = float(np.dot(axis, a - np.mean(a)) / denominator)
            sb = float(np.dot(axis, b - np.mean(b)) / denominator)
            if max(abs(sa), abs(sb)) > 6.0:
                continue
            difference = b - a
            spread = float(np.percentile(difference, 90) - np.percentile(difference, 10))
            if spread > 1.5:
                continue
            score = (spread + 0.25 * abs(sa - sb)
                     + 0.5 * max(input_peak - float(np.median(a)), 0.0)
                     + 0.05 / width)
            candidate = (score, start, end)
            if best is None or candidate < best:
                best = candidate
    return None if best is None else best[1:]


def estimate_input_target_match(speaker, target, *, measured_range=None, reference_range=None,
                                trim_db=0.0, plateau_window_db=3.0,
                                require_speaker_plateau=True, allow_automatic_fallback=False):
    if speaker is None or target is None:
        return LevelMatch(0, 0, 0, 0, "スピーカー入力とTargetを適用してください。")
    sf, tf = np.asarray(speaker.frequency), np.asarray(target.frequency)
    lo = max(5.0, float(min(sf)), float(min(tf)))
    hi = min(float(max(sf)), float(max(tf)))
    if measured_range is not None:
        lo, hi = max(lo, measured_range[0]), min(hi, measured_range[1])
    if reference_range is not None:
        lo, hi = max(lo, reference_range[0]), min(hi, reference_range[1])
    if hi <= lo or hi/lo < 2**(1/3):
        return LevelMatch(0, lo, hi, 0, "共通の測定帯域が不足しています。基準帯域を確認してください。")
    f = np.geomspace(lo, hi, max(33, int(np.log2(hi/lo)*96)+1))
    target_db = np.interp(f, tf, target.gain_db)
    input_db = np.interp(f, sf, speaker.gain_db)
    # At 96 samples/oct this rejects narrow spikes without erasing a plateau.
    smooth = median_filter(target_db, size=9, mode="nearest")
    method = "manual" if reference_range is not None else "plateau"
    if reference_range is None:
        # With no speaker gate (the reverse Target Auto path), retain the
        # global reference. Input matching instead uses the speaker passband:
        # a bass shelf outside a tweeter's useful range must not disqualify it.
        all_f = np.geomspace(max(5.0, float(min(tf))), float(max(tf)), 2049)
        peak = np.percentile(np.interp(all_f, tf, target.gain_db), 98)
        slope = np.gradient(smooth, np.log2(f))
        candidate = (smooth >= peak-float(plateau_window_db)) & (np.abs(slope) <= 3.0)
        if require_speaker_plateau:
            # A flat Target alone says nothing about the measurement's useful
            # band. Reject its low-level noise floor even when that is wider.
            input_lo, input_hi = max(5.0, float(min(sf))), float(max(sf))
            if measured_range is not None:
                input_lo = max(input_lo, measured_range[0])
                input_hi = min(input_hi, measured_range[1])
            input_f = np.geomspace(input_lo, input_hi, 2049)
            input_peak = np.percentile(np.interp(input_f, sf, speaker.gain_db), 98)
            input_smooth = median_filter(input_db, size=9, mode="nearest")
            input_slope = np.gradient(input_smooth, np.log2(f))
            speaker_candidate = ((input_smooth >= input_peak-float(plateau_window_db))
                                 & (np.abs(input_slope) <= 3.0))
            if np.any(speaker_candidate):
                peak = np.percentile(smooth[speaker_candidate], 98)
            # The slope gate still rejects crossover stopbands; a local
            # level reference alone would accept an attenuated roll-off.
            candidate = (speaker_candidate
                         & (smooth >= peak-float(plateau_window_db))
                         & (np.abs(slope) <= 3.0))
        starts = np.flatnonzero(candidate & ~np.r_[False, candidate[:-1]])
        ends = np.flatnonzero(candidate & ~np.r_[candidate[1:], False])
        runs = [(a, b) for a, b in zip(starts, ends) if f[b]/f[a] >= 2**(1/3)]
        fallback = None
        if not runs and allow_automatic_fallback:
            if require_speaker_plateau:
                reference_f = input_f
                reference_db = np.interp(reference_f, sf, speaker.gain_db)
            else:
                reference_f = all_f
                reference_db = np.interp(reference_f, tf, target.gain_db)
            reliable = reference_f >= 80.0 if reference_f[-1] >= 80.0 * 2 ** (1.0 / 3.0) else np.ones(reference_f.size, dtype=bool)
            fallback_peak = float(np.percentile(reference_db[reliable], 98))
            fallback = _stable_reference_band(
                f, input_db if require_speaker_plateau else target_db,
                target_db if require_speaker_plateau else input_db, fallback_peak,
            )
        if not runs and fallback is None:
            message = ("スピーカー測定とTargetに共通する高く平坦な帯域を検出できません。"
                       if require_speaker_plateau else "Targetの高く平坦な帯域を検出できません。")
            return LevelMatch(0, lo, hi, 0, message + "基準帯域を手動指定してください。")
        if runs:
            a, b = max(runs, key=lambda pair: np.log2(f[pair[1]]/f[pair[0]]))
        else:
            a, b = fallback
            method = "stable_band"
        f, target_db, input_db = f[a:b+1], target_db[a:b+1], input_db[a:b+1]
    offset = float(np.median(target_db-input_db)) + float(trim_db)
    if not np.isfinite(offset) or abs(offset) > 120:
        return LevelMatch(0, float(f[0]), float(f[-1]), 0, "調整量が±120 dBを超えます。データの単位を確認してください。")
    residual = float(np.median(input_db+offset-target_db))
    return LevelMatch(offset, float(f[0]), float(f[-1]), residual, method=method)
