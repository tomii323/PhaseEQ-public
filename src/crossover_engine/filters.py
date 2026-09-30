from __future__ import annotations

import numpy as np
import math
from scipy.signal import firwin, convolve, fftconvolve, firwin2, freqz
from .fractional_centering import auto_center_fractional
from .slope import make_baffle_step_fir as _bsc_make_baffle_step_fir


def make_baffle_step_fir(
    fs: float,
    *,
    f0: float = 170.0,
    f1: float = 1000.0,
    max_atten_db: float = 6.0,
    taps: int | None = None,
    window=("kaiser", 12.0),
) -> np.ndarray:
    f0 = float(max(f0, 1.0))
    f1 = float(max(f1, f0 + 1.0))
    max_atten_db = float(max(max_atten_db, 0.0))

    slope_db_per_oct = max_atten_db / max(math.log2(f1 / f0), 1e-9)

    if taps is None or taps <= 0:
        est = int(fs / f1 * 5.0)
        taps = est + (1 - est % 2)

    h = _bsc_make_baffle_step_fir(
        fs=fs,
        taps=taps,
        slope_db_per_oct=slope_db_per_oct,
        gain_max_db=max_atten_db,
        f_pivot=f1,
        baffle_width_m=None,
        f_lo=max(20.0, f0 * 0.5),
        f_hi=f1,
        window=window,
        soft_knee=None,
        auto_soft_knee_flag=True,
        grid_mul=2,
        ensure_odd_taps=True,
    )

    w, H = freqz(h, worN=8192, fs=fs)
    gmax = float(np.max(np.abs(H)))
    if gmax > 0:
        h = h / gmax

    return h


def even_number(n):
    n = int(np.floor(n))
    return n + (n % 2)


def odd_number(n):
    n = int(np.floor(n))
    return n + 1 + (n % 2)


def _clip_cutoff(fc, fs):
    nyq = fs / 2.0
    return float(min(max(fc, 1.0), nyq - 1.0))


def kaiser_overlap_edges_hz(fc, overlap_oct):
    """Return logarithmically symmetric LP/HP Kaiser cutoff edges."""
    frequency = float(fc)
    overlap = float(overlap_oct)
    if frequency <= 0.0 or not np.isfinite(frequency):
        raise ValueError("crossover frequency must be positive")
    if not np.isfinite(overlap) or not -1.0 <= overlap <= 1.0:
        raise ValueError("Kaiser overlap must be within -1..+1 oct")
    return frequency * 2.0 ** (overlap / 2.0), frequency * 2.0 ** (-overlap / 2.0)


def _linear_phase_lr2_fir(fs, mode, frequency_hz, taps):
    return _linear_phase_lr_fir(fs, mode, frequency_hz, taps, order=2)


def _linear_phase_lr4_fir(fs, mode, frequency_hz, taps):
    return _linear_phase_lr_fir(fs, mode, frequency_hz, taps, order=4)


def _linear_phase_lr_fir(fs, mode, frequency_hz, taps, *, order):
    grid_intervals = 1
    while grid_intervals < max(int(taps), 16_384):
        grid_intervals *= 2
    grid_frequency = np.linspace(0.0, float(fs) / 2.0, grid_intervals + 1)
    target_lp = 1.0 / (1.0 + (grid_frequency / float(frequency_hz)) ** order)
    low_pass = firwin2(
        int(taps), grid_frequency, target_lp, nfreqs=grid_intervals + 1,
        window=None, fs=float(fs),
    )
    low_pass = 0.5 * (low_pass + low_pass[::-1])
    if mode == "lp":
        return low_pass
    delta = np.zeros(int(taps), dtype=float)
    delta[int(taps) // 2] = 1.0
    return delta - low_pass


def generate_exclusive_kaiser_firs(
    fs, ordered_ways, crossover_frequencies_hz, methods, overlap_oct, cycles, beta,
    *, lr2_auto_taps=True,
):
    """Build only the Kaiser boundaries selected in an exclusive mixed layout."""
    ways = tuple(ordered_ways)
    frequencies = tuple(float(value) for value in crossover_frequencies_hz)
    methods = tuple(str(value) for value in methods)
    overlaps = tuple(float(value) for value in overlap_oct)
    cycles_values = tuple(float(value) for value in cycles)
    beta_values = tuple(float(value) for value in beta)
    count = len(ways) - 1
    if not all(len(values) == count for values in (frequencies, methods, overlaps, cycles_values, beta_values)):
        raise ValueError("exclusive crossover boundary count does not match Ways")
    if any(value not in {"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR", "LR2", "LR4", "Through"} for value in methods):
        raise ValueError("unsupported exclusive crossover method")
    result = {}
    for way_index, way in enumerate(ways):
        response = np.asarray([1.0], dtype=float)
        for boundary_index in (way_index - 1, way_index):
            if boundary_index < 0 or boundary_index >= count or methods[boundary_index] not in {"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR"}:
                continue
            lp_edge, hp_edge = kaiser_overlap_edges_hz(frequencies[boundary_index], overlaps[boundary_index])
            taps = odd_number(fs / max(frequencies[boundary_index], 1.0) * cycles_values[boundary_index])
            if methods[boundary_index] == "Linear-phase LR2 FIR" and lr2_auto_taps:
                from .lr2_taps import automatic_lr2_taps
                taps = automatic_lr2_taps(int(fs), frequencies[boundary_index], overlaps[boundary_index]).taps
            if methods[boundary_index] == "Linear-phase LR4 FIR":
                from .lr4_taps import automatic_lr4_taps
                taps = automatic_lr4_taps(int(fs), frequencies[boundary_index], overlaps[boundary_index]).taps
            is_upper_way = way_index == boundary_index + 1
            cutoff = hp_edge if is_upper_way else lp_edge
            section = (
                _linear_phase_lr_fir(fs, "hp" if is_upper_way else "lp", cutoff, taps,
                                     order=4 if methods[boundary_index] == "Linear-phase LR4 FIR" else 2)
                if methods[boundary_index] in {"Linear-phase LR2 FIR", "Linear-phase LR4 FIR"}
                else firwin(
                    int(taps), _clip_cutoff(cutoff, fs) / (fs / 2.0),
                    window=("kaiser", beta_values[boundary_index]), pass_zero=not is_upper_way,
                )
            )
            response = np.convolve(response, section)
        result[str(way)] = auto_center_fractional(
            response, resolution=4, mode="groupdelay", smart_threshold=0.10,
        ) if response.size > 1 else response
    return result


def _orig_generate_2way_filters(fs, fc, cycles=5, beta=12.0, overlap_hz=0.0, overlap_oct=None):
    """
    2Way FIR分割フィルタ生成（Kaiser固定・Octaveアルゴリズム）
    overlap_hz を与えると、fc を中心として Low:+overlap/2, High:-overlap/2 に広げる。
    Return:
        dict {"Low": fir_low, "High": fir_high}
    """
    if overlap_oct is None:
        overlap_hz = max(float(overlap_hz), 0.0)
        fc_low, fc_high = fc + overlap_hz / 2.0, fc - overlap_hz / 2.0
    else:
        fc_low, fc_high = kaiser_overlap_edges_hz(fc, overlap_oct)
    fc_low, fc_high = _clip_cutoff(fc_low, fs), _clip_cutoff(fc_high, fs)

    taps = odd_number(fs / max(fc, 1.0) * cycles)
    fir_low = firwin(int(taps), fc_low/(fs/2), window=('kaiser', beta), pass_zero=True)
    fir_high = firwin(int(taps), fc_high/(fs/2), window=('kaiser', beta), pass_zero=False)
    return {"Low": fir_low, "High": fir_high}


def _orig_generate_3way_filters(
    fs,
    fc1,
    fc2,
    cycles=5,
    beta=12.0,
    overlap1_hz=0.0,
    overlap2_hz=0.0,
    cycles_low=None,
    beta_low=None,
    cycles_high=None,
    beta_high=None,
    overlap1_oct=None,
    overlap2_oct=None,
):
    """
    3Way FIR分割フィルタ生成（Kaiser固定・Octave模倣）
    overlap1_hz: Low/Mid クロスでの重ね代
    overlap2_hz: Mid/High クロスでの重ね代
    Return:
        dict {"Low": fir_low, "Mid": fir_mid, "High": fir_high}
    """
    cycles_low = float(cycles if cycles_low is None else cycles_low)
    beta_low = float(beta if beta_low is None else beta_low)
    cycles_high = float(cycles if cycles_high is None else cycles_high)
    beta_high = float(beta if beta_high is None else beta_high)

    overlap1_hz = max(float(overlap1_hz), 0.0)
    overlap2_hz = max(float(overlap2_hz), 0.0)

    low_fc, low2_fc = (
        kaiser_overlap_edges_hz(fc1, overlap1_oct)
        if overlap1_oct is not None else (fc1 + overlap1_hz, fc1 - overlap1_hz)
    )
    high2_fc, high_fc = (
        kaiser_overlap_edges_hz(fc2, overlap2_oct)
        if overlap2_oct is not None else (fc2 + overlap2_hz, fc2 - overlap2_hz)
    )
    low_fc, high_fc = _clip_cutoff(low_fc, fs), _clip_cutoff(high_fc, fs)
    low2_fc, high2_fc = _clip_cutoff(low2_fc, fs), _clip_cutoff(high2_fc, fs)

    t1 = odd_number(fs / max(fc1, 1.0) * cycles_low)
    t2 = odd_number(fs / max(fc2, 1.0) * cycles_high)

    fir_low = firwin(int(t1), low_fc/(fs/2), window=('kaiser', beta_low), pass_zero=True)
    fir_high = firwin(int(t2), high_fc/(fs/2), window=('kaiser', beta_high), pass_zero=False)

    # Mid は出力用 Low/High とは別に、逆方向へ動かした Low2/High2 から作る。
    # これにより Mid 側の重なりも厚くしつつ、既存の残差生成の考え方を保つ。
    fir_low2 = firwin(int(t1), low2_fc/(fs/2), window=('kaiser', beta_low), pass_zero=True)
    fir_high2 = firwin(int(t2), high2_fc/(fs/2), window=('kaiser', beta_high), pass_zero=False)
    fir_low2, fir_high2 = center_fir_lengths([fir_low2, fir_high2])
    delta = np.zeros_like(fir_low2)
    delta[len(delta) // 2] = 1.0
    fir_mid = delta - fir_low2 - fir_high2

    return {
        "Low": fir_low,
        "Mid": fir_mid,
        "High": fir_high
    }


def _orig_generate_4way_filters(
    fs,
    fc1,
    fc2,
    fc3,
    cycles=5,
    beta=12.0,
    overlap1_hz=0.0,
    overlap2_hz=0.0,
    overlap3_hz=0.0,
    cycles_sub_low=None,
    beta_sub_low=None,
    cycles_low_mid=None,
    beta_low_mid=None,
    cycles_mid_high=None,
    beta_mid_high=None,
    overlap1_oct=None,
    overlap2_oct=None,
    overlap3_oct=None,
):
    """
    4Way FIR分割フィルタ生成。

    SUBとHighは直接生成し、LowとMidは3WayのMidと同じ残差方式で生成する。
    overlap1/2/3_hz は各クロスで指定値をそのまま足し引きする。
    Return:
        dict {"SUB": fir_sub, "Low": fir_low, "Mid": fir_mid, "High": fir_high}
    """
    cycles_sub_low = float(cycles if cycles_sub_low is None else cycles_sub_low)
    beta_sub_low = float(beta if beta_sub_low is None else beta_sub_low)
    cycles_low_mid = float(cycles if cycles_low_mid is None else cycles_low_mid)
    beta_low_mid = float(beta if beta_low_mid is None else beta_low_mid)
    cycles_mid_high = float(cycles if cycles_mid_high is None else cycles_mid_high)
    beta_mid_high = float(beta if beta_mid_high is None else beta_mid_high)

    overlap1_hz = max(float(overlap1_hz), 0.0)
    overlap2_hz = max(float(overlap2_hz), 0.0)
    overlap3_hz = max(float(overlap3_hz), 0.0)

    t1 = odd_number(fs / max(fc1, 1.0) * cycles_sub_low)
    t2 = odd_number(fs / max(fc2, 1.0) * cycles_low_mid)
    t3 = odd_number(fs / max(fc3, 1.0) * cycles_mid_high)

    edge1_lp, edge1_hp = (
        kaiser_overlap_edges_hz(fc1, overlap1_oct)
        if overlap1_oct is not None else (fc1 + overlap1_hz, fc1 - overlap1_hz)
    )
    edge2_lp, edge2_hp = (
        kaiser_overlap_edges_hz(fc2, overlap2_oct)
        if overlap2_oct is not None else (fc2 + overlap2_hz, fc2 - overlap2_hz)
    )
    edge3_lp, edge3_hp = (
        kaiser_overlap_edges_hz(fc3, overlap3_oct)
        if overlap3_oct is not None else (fc3 + overlap3_hz, fc3 - overlap3_hz)
    )

    fir_sub = firwin(
        int(t1),
        _clip_cutoff(edge1_lp, fs) / (fs / 2),
        window=("kaiser", beta_sub_low),
        pass_zero=True,
    )
    fir_high = firwin(
        int(t3),
        _clip_cutoff(edge3_hp, fs) / (fs / 2),
        window=("kaiser", beta_mid_high),
        pass_zero=False,
    )

    # Low = delta - SUB側ローパス - Low/Mid側ハイパス
    fir_sub2 = firwin(
        int(t1),
        _clip_cutoff(edge1_hp, fs) / (fs / 2),
        window=("kaiser", beta_sub_low),
        pass_zero=True,
    )
    fir_low_high = firwin(
        int(t2),
        _clip_cutoff(edge2_lp, fs) / (fs / 2),
        window=("kaiser", beta_low_mid),
        pass_zero=False,
    )
    fir_sub2, fir_low_high = center_fir_lengths([fir_sub2, fir_low_high])
    delta_low = np.zeros_like(fir_sub2)
    delta_low[len(delta_low) // 2] = 1.0
    fir_low = delta_low - fir_sub2 - fir_low_high

    # Mid = delta - Low/Mid側ローパス - Mid/High側ハイパス
    fir_mid_low = firwin(
        int(t2),
        _clip_cutoff(edge2_hp, fs) / (fs / 2),
        window=("kaiser", beta_low_mid),
        pass_zero=True,
    )
    fir_mid_high = firwin(
        int(t3),
        _clip_cutoff(edge3_lp, fs) / (fs / 2),
        window=("kaiser", beta_mid_high),
        pass_zero=False,
    )
    fir_mid_low, fir_mid_high = center_fir_lengths([fir_mid_low, fir_mid_high])
    delta_mid = np.zeros_like(fir_mid_low)
    delta_mid[len(delta_mid) // 2] = 1.0
    fir_mid = delta_mid - fir_mid_low - fir_mid_high

    return {
        "SUB": fir_sub,
        "Low": fir_low,
        "Mid": fir_mid,
        "High": fir_high,
    }


def generate_residual_multiband_filters(
    fs,
    ways,
    crossover_frequencies_hz,
    cycles,
    beta,
    overlap_oct=None,
):
    """Generate an ordered 2..5 Way Kaiser bank with residual inner bands."""
    ordered = tuple(str(way) for way in ways)
    frequencies = tuple(float(value) for value in crossover_frequencies_hz)
    cycles_values = tuple(float(value) for value in cycles)
    beta_values = tuple(float(value) for value in beta)
    overlaps = tuple(float(value) for value in (overlap_oct or (0.0,) * len(frequencies)))
    boundary_count = len(ordered) - 1
    if not 2 <= len(ordered) <= 5:
        raise ValueError("residual multiband generation supports 2..5 Ways")
    if not all(
        len(values) == boundary_count
        for values in (frequencies, cycles_values, beta_values, overlaps)
    ):
        raise ValueError("multiband boundary count does not match Ways")
    if any(left >= right for left, right in zip(frequencies, frequencies[1:])):
        raise ValueError("crossover frequencies must be strictly increasing")

    boundary_sections = []
    for frequency, boundary_cycles, boundary_beta, overlap in zip(
        frequencies, cycles_values, beta_values, overlaps, strict=True,
    ):
        lp_edge, hp_edge = kaiser_overlap_edges_hz(frequency, overlap)
        taps = odd_number(fs / max(frequency, 1.0) * boundary_cycles)
        low_pass = firwin(
            int(taps), _clip_cutoff(lp_edge, fs) / (fs / 2.0),
            window=("kaiser", boundary_beta), pass_zero=True,
        )
        high_pass = firwin(
            int(taps), _clip_cutoff(hp_edge, fs) / (fs / 2.0),
            window=("kaiser", boundary_beta), pass_zero=False,
        )
        boundary_sections.append((low_pass, high_pass))

    result = {ordered[0]: boundary_sections[0][0]}
    for index, way in enumerate(ordered[1:-1], start=1):
        lower_low_pass = boundary_sections[index - 1][0]
        upper_high_pass = boundary_sections[index][1]
        lower_low_pass, upper_high_pass = center_fir_lengths(
            [lower_low_pass, upper_high_pass]
        )
        delta = np.zeros_like(lower_low_pass)
        delta[len(delta) // 2] = 1.0
        result[way] = delta - lower_low_pass - upper_high_pass
    result[ordered[-1]] = boundary_sections[-1][1]
    return result


def center_fir_lengths(fir_list):
    lengths = [len(h) for h in fir_list]
    max_len = max(lengths)
    return [
        np.pad(h, ((max_len - len(h)) // 2, (max_len - len(h) + 1) // 2))
        for h in fir_list
    ]


def match_fir_length(fir, target_length):
    current = len(fir)
    if current < target_length:
        pad = (target_length - current) // 2
        return np.pad(fir, (pad, target_length - current - pad))
    elif current > target_length:
        start = (current - target_length) // 2
        return fir[start:start + target_length]
    return fir


def combine_fir(base, eq, mode='full'):
    return fftconvolve(base, eq, mode)


def combine_and_crop_fir_conv(split_fir, eq_fir=None, crop_len=0, mode='full'):
    if eq_fir is not None:
        fir = combine_fir(split_fir, eq_fir, mode)
    else:
        fir = split_fir.copy()
    if crop_len and crop_len > 0:
        fir = match_fir_length(fir, crop_len)
    return fir


def normalize_fir_group_response(
    firs_by_band: dict[str, np.ndarray],
    target_max_db: float = -0.3,
    response_points: int = 65536,
) -> tuple[dict[str, np.ndarray], float, str | None, dict[str, float]]:
    """各帯域の最大周波数応答を共通ゲインで目標値へ合わせる。"""
    converted = {
        band: np.asarray(fir, dtype=np.float64)
        for band, fir in firs_by_band.items()
    }
    response_points = max(int(response_points), 1024)
    response_peaks = {
        band: (
            float(np.max(np.abs(freqz(fir, worN=response_points, include_nyquist=True)[1])))
            if fir.size else 0.0
        )
        for band, fir in converted.items()
    }
    reference = max(response_peaks, key=response_peaks.get) if response_peaks else None
    maximum_response = response_peaks.get(reference, 0.0)
    if maximum_response <= 0.0:
        return (
            {band: fir.copy() for band, fir in converted.items()},
            1.0,
            reference,
            response_peaks,
        )

    target_linear = 10.0 ** (float(target_max_db) / 20.0)
    gain = target_linear / maximum_response
    normalized = {
        band: fir * gain
        for band, fir in converted.items()
    }
    return normalized, float(gain), reference, response_peaks


def measure_fir_group_response_peaks(
    firs_by_band: dict[str, np.ndarray],
    response_points: int = 65536,
) -> tuple[dict[str, float], float]:
    """各帯域と、中央揃えした合成FIRの最大周波数応答を返す。"""
    converted = {
        band: np.asarray(fir, dtype=np.float64)
        for band, fir in firs_by_band.items()
    }
    response_points = max(int(response_points), 1024)
    band_peaks = {
        band: (
            float(np.max(np.abs(freqz(fir, worN=response_points, include_nyquist=True)[1])))
            if fir.size else 0.0
        )
        for band, fir in converted.items()
    }
    if not converted:
        return band_peaks, 0.0

    combined = np.sum(center_fir_lengths(list(converted.values())), axis=0)
    sum_peak = float(np.max(np.abs(
        freqz(combined, worN=response_points, include_nyquist=True)[1]
    )))
    return band_peaks, sum_peak


def kaiser_beta_from_atten(atten_db):
    A = float(atten_db)
    if A > 50:
        beta = 0.1102 * (A - 8.7)
    elif 21 < A <= 50:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21)
    else:
        beta = 0.0
    return beta


def _orig_make_baffle_step_fir(
    fs, f0=170, f1=1000, max_atten_db=6.0, atten_freq_ratio=0.999, min_taps=31
):
    if fs <= 0 or f0 <= 0 or f1 <= f0 or max_atten_db <= 0 or min_taps < 3:
        raise ValueError("引数が不正です")
    nyq = fs / 2
    taps = int(np.ceil(fs / f0 / 2))
    taps = taps if taps % 2 == 1 else taps + 1
    taps = max(min_taps, min(taps, 1023))
    atten_f = nyq * atten_freq_ratio
    freq = [0, f0, f1, atten_f, nyq]
    gain = [1.0, 1.0, 10**(-max_atten_db/20), 10**(-max_atten_db/20), 0.0]
    freq = [f / nyq for f in freq]
    beta = kaiser_beta_from_atten(max_atten_db + 25)
    fir = firwin2(taps, freq, gain, window=('kaiser', beta))
    w, h = freqz(fir, worN=4096, fs=fs)
    idx_pass = np.where((w >= 20) & (w <= f0))[0]
    max_gain = np.abs(h[idx_pass]).max()
    fir = fir / max_gain
    return fir


def generate_2way_filters(*args, **kwargs):
    out = _orig_generate_2way_filters(*args, **kwargs)
    if not isinstance(out, dict):
        raise TypeError("generate_2way_filters: _orig_generate_2way_filters must return dict.")
    key_low = "Low" if "Low" in out else "low"
    key_high = "High" if "High" in out else "high"
    low = auto_center_fractional(out[key_low], resolution=4, mode="groupdelay", smart_threshold=0.10)
    high = auto_center_fractional(out[key_high], resolution=4, mode="groupdelay", smart_threshold=0.10)
    out[key_low] = low
    out[key_high] = high
    return out


def generate_3way_filters(*args, **kwargs):
    out = _orig_generate_3way_filters(*args, **kwargs)
    if not isinstance(out, dict):
        raise TypeError("generate_3way_filters: _orig_generate_3way_filters must return dict.")
    key_low = "Low" if "Low" in out else "low"
    key_mid = "Mid" if "Mid" in out else "mid"
    key_high = "High" if "High" in out else "high"
    out[key_low] = auto_center_fractional(out[key_low], resolution=4, mode="groupdelay", smart_threshold=0.10)
    out[key_mid] = auto_center_fractional(out[key_mid], resolution=4, mode="groupdelay", smart_threshold=0.10)
    out[key_high] = auto_center_fractional(out[key_high], resolution=4, mode="groupdelay", smart_threshold=0.10)
    return out


def generate_4way_filters(*args, **kwargs):
    out = _orig_generate_4way_filters(*args, **kwargs)
    if not isinstance(out, dict):
        raise TypeError("generate_4way_filters: _orig_generate_4way_filters must return dict.")
    for band in ("SUB", "Low", "Mid", "High"):
        key = band if band in out else band.lower()
        out[key] = auto_center_fractional(
            out[key], resolution=4, mode="groupdelay", smart_threshold=0.10
        )
    return out
