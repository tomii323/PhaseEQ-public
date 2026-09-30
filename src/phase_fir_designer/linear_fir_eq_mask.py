from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.signal import kaiser_beta

from octave_boundary_smoothing import OneSidedBoundarySpec, apply_one_sided_boundary

from .config import (
    DesignConfig,
    LINEAR_FIR_BETA_MAX,
    LINEAR_FIR_BETA_MIN,
    LINEAR_FIR_EQ_MASK_SMOOTHING_DEFAULT,
)
from .iir import iir_frequency_response
from .linear_fir import band_split_iir_filters, design_linear_fir
from .phase_eq_mask_adaptive import apply_adaptive_phase_eq_mask_low

DEFAULT_EQ_MASK_SMOOTHING_OCT = LINEAR_FIR_EQ_MASK_SMOOTHING_DEFAULT
EQ_MASK_PRACTICAL_FLOOR_DB = -80.0
LR_EQ_MASK_THRESHOLD_DB = -40.0
LR2_HP_EQ_MASK_THRESHOLD_DB = -20.0
LR2_HP_PHASE_EQ_MASK_INNER_GUARD_DB = -20.0
PHASE_EQ_MASK_INNER_GUARD_DB = -40.0
GAIN_OUTSIDE_FADE_MAX_DB_PER_OCT = 6.0
OUTSIDE_FADE_EDGE_SMOOTH_OCT = 1.0 / 6.0


@dataclass(frozen=True)
class LinearFIREQMask:
    lo_hz: float | None
    hi_hz: float | None
    threshold_db: float
    smoothing_oct: float = DEFAULT_EQ_MASK_SMOOTHING_OCT
    phase_lo_smoothing_oct: float | None = None
    phase_hi_smoothing_oct: float | None = None
    phase_inner_guard_db: float = PHASE_EQ_MASK_INNER_GUARD_DB
    lo_threshold_db: float | None = None
    hi_threshold_db: float | None = None
    phase_lo_inner_guard_db: float | None = None
    phase_hi_inner_guard_db: float | None = None

    @property
    def enabled(self) -> bool:
        return self.lo_hz is not None or self.hi_hz is not None


def linear_fir_eq_mask(config: DesignConfig, frequency: np.ndarray) -> LinearFIREQMask | None:
    linear_filters = [item for item in config.linear_fir_filters if item.enabled and not item.acoustic_target and item.response != "through"]
    if config.band_split_recipe is not None:
        from .config import LinearFIRFilter
        from crossover_engine.recipe import adjacent_filter_parameters
        linear_filters = [LinearFIRFilter(**item)
                          for item in adjacent_filter_parameters(config.band_split_recipe)
                          if not item.get("acoustic_target", False) and (config.band_split_recipe.get("fir_enabled", True)
                          or item["response"] in ("iir_lr2", "iir_lr4"))]
    if not linear_filters:
        return None
    freq = np.asarray(frequency, dtype=float)
    if freq.size < 3:
        return None
    valid_indices = np.flatnonzero(np.isfinite(freq) & (freq > 0.0))
    if valid_indices.size == 0:
        return None
    first_valid = int(valid_indices[0])
    last_valid = int(valid_indices[-1])
    by_mode = {item.mode: item for item in linear_filters}
    hp_item = by_mode.get("hp")
    lp_item = by_mode.get("lp")
    hp_gain_db = _boundary_gain_db(config, hp_item, freq) if hp_item is not None else None
    lp_gain_db = _boundary_gain_db(config, lp_item, freq) if lp_item is not None else None
    lo_threshold_db = _boundary_threshold_db(hp_item) if hp_item is not None else None
    hi_threshold_db = _boundary_threshold_db(lp_item) if lp_item is not None else None
    phase_lo_guard_db = _boundary_phase_guard_db(hp_item) if hp_item is not None else None
    phase_hi_guard_db = _boundary_phase_guard_db(lp_item) if lp_item is not None else None
    lo_hz = _low_boundary_hz(freq, hp_gain_db, lo_threshold_db, first_valid)
    hi_hz = _high_boundary_hz(freq, lp_gain_db, hi_threshold_db, last_valid)
    if lo_hz is None and hi_hz is None:
        return None
    smoothing_oct = float(config.linear_fir_eq_mask_smoothing_oct)
    phase_lo_passband = _passband_indices(freq, hp_gain_db, phase_lo_guard_db)
    phase_hi_passband = _passband_indices(freq, lp_gain_db, phase_hi_guard_db)
    phase_lo_smoothing_oct = _phase_low_smoothing_oct(
        freq,
        lo_hz,
        phase_lo_passband,
        smoothing_oct,
    )
    phase_hi_smoothing_oct = _phase_high_smoothing_oct(
        freq,
        hi_hz,
        phase_hi_passband,
        smoothing_oct,
    )
    if lo_hz is not None and hi_hz is not None:
        split_hz = float(np.sqrt(lo_hz * hi_hz))
        phase_lo_smoothing_oct = min(
            phase_lo_smoothing_oct,
            max(0.0, float(np.log2(split_hz / lo_hz))),
        )
        phase_hi_smoothing_oct = min(
            phase_hi_smoothing_oct,
            max(0.0, float(np.log2(hi_hz / split_hz))),
        )
    mask = LinearFIREQMask(
        lo_hz=lo_hz,
        hi_hz=hi_hz,
        threshold_db=float(lo_threshold_db if lo_threshold_db is not None else hi_threshold_db),
        smoothing_oct=smoothing_oct,
        phase_lo_smoothing_oct=phase_lo_smoothing_oct,
        phase_hi_smoothing_oct=phase_hi_smoothing_oct,
        phase_inner_guard_db=float(
            phase_lo_guard_db if phase_lo_guard_db is not None else phase_hi_guard_db
        ),
        lo_threshold_db=lo_threshold_db,
        hi_threshold_db=hi_threshold_db,
        phase_lo_inner_guard_db=phase_lo_guard_db,
        phase_hi_inner_guard_db=phase_hi_guard_db,
    )
    return mask if mask.enabled else None


def eq_mask_threshold_db(linear_filters) -> float:
    thresholds = [_boundary_threshold_db(item) for item in linear_filters]
    return float(min(thresholds)) if thresholds else float(EQ_MASK_PRACTICAL_FLOOR_DB)


def _normalized_method(item) -> str:
    method = str(getattr(item, "response", "kaiser"))
    return "linear_phase_lr2" if method == "lr2" else method


def _boundary_threshold_db(item) -> float:
    method = _normalized_method(item)
    if method == "kaiser":
        attenuation = kaiser_beta_to_attenuation_db(float(getattr(item, "beta", LINEAR_FIR_BETA_MIN)))
        return -min(float(attenuation), abs(EQ_MASK_PRACTICAL_FLOOR_DB))
    if method in {"linear_phase_lr2", "iir_lr2"} and str(getattr(item, "mode", "lp")) == "hp":
        return float(LR2_HP_EQ_MASK_THRESHOLD_DB)
    return float(LR_EQ_MASK_THRESHOLD_DB)


def _boundary_phase_guard_db(item) -> float:
    method = _normalized_method(item)
    if method in {"linear_phase_lr2", "iir_lr2"} and str(getattr(item, "mode", "lp")) == "hp":
        return float(LR2_HP_PHASE_EQ_MASK_INNER_GUARD_DB)
    return float(PHASE_EQ_MASK_INNER_GUARD_DB)


def _boundary_gain_db(config: DesignConfig, item, frequency: np.ndarray) -> np.ndarray:
    if _normalized_method(item) in {"iir_lr2", "iir_lr4"}:
        boundary_config = DesignConfig(
            sample_rate=config.sample_rate,
            taps=config.taps,
            analysis_fft_size=config.analysis_fft_size,
            linear_fir_filters=[item],
        )
        response = iir_frequency_response(
            band_split_iir_filters(boundary_config),
            config.sample_rate,
            frequency,
        )
    else:
        coefficients = design_linear_fir(config, item)
        fft_size = max(2, (frequency.size - 1) * 2)
        response = np.fft.rfft(np.asarray(coefficients, dtype=float), n=fft_size)
        if response.shape != frequency.shape:
            response = np.interp(
                frequency,
                np.fft.rfftfreq(fft_size, d=1.0 / config.sample_rate),
                response.real,
            ).astype(complex)
    return 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))


def _passband_indices(frequency, gain_db, threshold_db) -> np.ndarray:
    if gain_db is None or threshold_db is None:
        return np.array([], dtype=int)
    return np.flatnonzero(
        np.isfinite(frequency) & np.isfinite(gain_db) & (frequency > 0.0) & (gain_db >= threshold_db)
    )


def _low_boundary_hz(frequency, gain_db, threshold_db, first_valid: int) -> float | None:
    indices = _passband_indices(frequency, gain_db, threshold_db)
    if indices.size == 0:
        return None
    first = int(indices[0])
    return float(frequency[first]) if first > first_valid else None


def _high_boundary_hz(frequency, gain_db, threshold_db, last_valid: int) -> float | None:
    indices = _passband_indices(frequency, gain_db, threshold_db)
    if indices.size == 0:
        return None
    last = int(indices[-1])
    return float(frequency[last]) if last < last_valid else None


def kaiser_beta_to_attenuation_db(beta: float) -> float:
    beta = min(max(float(beta), LINEAR_FIR_BETA_MIN), LINEAR_FIR_BETA_MAX)
    if beta <= 0.0:
        return 21.0
    return float(brentq(lambda attenuation: kaiser_beta(attenuation) - beta, 21.0, 200.0))


def apply_linear_fir_eq_mask(
    config: DesignConfig,
    frequency: np.ndarray,
    values: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    mask = linear_fir_eq_mask(config, frequency)
    if mask is None:
        return np.asarray(values, dtype=float)
    return apply_eq_mask_to_values(frequency, values, mask, mode=mode)


def apply_eq_mask_to_values(
    frequency: np.ndarray,
    values: np.ndarray,
    mask: LinearFIREQMask,
    *,
    mode: str,
) -> np.ndarray:
    freq = np.asarray(frequency, dtype=float)
    source = np.asarray(values, dtype=float)
    if mode == "phase":
        source = _unwrap_phase_degrees(source)
    output = source.copy()
    if freq.size == 0 or source.size != freq.size:
        return output
    split_hz = None
    if mask.lo_hz is not None and mask.hi_hz is not None:
        split_hz = float(np.sqrt(float(mask.lo_hz) * float(mask.hi_hz)))
    if mask.lo_hz is not None:
        low_smoothing_oct = (
            mask.phase_lo_smoothing_oct
            if mode == "phase" and mask.phase_lo_smoothing_oct is not None
            else mask.smoothing_oct
        )
        low_upper_limit_hz = split_hz
        if mode == "phase":
            phase_limit_hz = float(mask.lo_hz) * (2.0 ** float(low_smoothing_oct))
            low_upper_limit_hz = (
                phase_limit_hz
                if low_upper_limit_hz is None
                else min(float(low_upper_limit_hz), phase_limit_hz)
            )
        output = _apply_low_side_mask(
            freq,
            output,
            float(mask.lo_hz),
            float(low_smoothing_oct),
            mode,
            upper_limit_hz=low_upper_limit_hz,
        )
    if mask.hi_hz is not None:
        high_smoothing_oct = (
            mask.phase_hi_smoothing_oct
            if mode == "phase" and mask.phase_hi_smoothing_oct is not None
            else mask.smoothing_oct
        )
        high_lower_limit_hz = split_hz
        if mode == "phase":
            phase_limit_hz = float(mask.hi_hz) / (2.0 ** float(high_smoothing_oct))
            high_lower_limit_hz = (
                phase_limit_hz
                if high_lower_limit_hz is None
                else max(float(high_lower_limit_hz), phase_limit_hz)
            )
        output = _apply_high_side_mask(
            freq,
            output,
            float(mask.hi_hz),
            float(high_smoothing_oct),
            mode,
            lower_limit_hz=high_lower_limit_hz,
        )
    return output


def _apply_low_side_mask(
    frequency: np.ndarray,
    values: np.ndarray,
    boundary_hz: float,
    smoothing_oct: float,
    mode: str,
    *,
    upper_limit_hz: float | None = None,
) -> np.ndarray:
    if mode == "phase":
        return apply_adaptive_phase_eq_mask_low(
            frequency,
            values,
            boundary_hz=boundary_hz,
            smoothing_oct=smoothing_oct,
            upper_limit_hz=upper_limit_hz,
        ).phase_deg

    output = values.copy()
    hold = _boundary_hold_value(frequency, values, boundary_hz, mode)
    outside = frequency < boundary_hz
    if np.any(outside):
        output[outside] = _fade_boundary_value_to_zero_by_octave_slope(
            frequency[outside],
            boundary_hz,
            hold,
            GAIN_OUTSIDE_FADE_MAX_DB_PER_OCT,
            side="low",
        )
    if smoothing_oct <= 0:
        return output
    outer_slope = _signed_outside_slope(hold, GAIN_OUTSIDE_FADE_MAX_DB_PER_OCT, side="low")
    connection = apply_one_sided_boundary(
        frequency,
        values,
        OneSidedBoundarySpec(
            fb_hz=boundary_hz,
            delta_oct=smoothing_oct,
            side="lower",
            axis="log",
            outer_value=hold,
            outer_slope=outer_slope,
            reference_oct=1.0 / 6.0,
            upper_limit_hz=upper_limit_hz,
            name=f"Linear FIR EQ mask {mode} Lo",
        ),
    )
    if connection.item.applied:
        transition = (frequency >= float(connection.item.f_start_hz)) & (
            frequency <= float(connection.item.f_end_hz)
        )
        output[transition] = connection.values[transition]
    return output


def _apply_high_side_mask(
    frequency: np.ndarray,
    values: np.ndarray,
    boundary_hz: float,
    smoothing_oct: float,
    mode: str,
    *,
    lower_limit_hz: float | None = None,
) -> np.ndarray:
    output = values.copy()
    hold = _boundary_hold_value(frequency, values, boundary_hz, mode)
    outside = frequency > boundary_hz
    if np.any(outside):
        if mode == "gain":
            output[outside] = _fade_boundary_value_to_zero_by_octave_slope(
                frequency[outside],
                boundary_hz,
                hold,
                GAIN_OUTSIDE_FADE_MAX_DB_PER_OCT,
                side="high",
            )
        else:
            output[outside] = hold
    if smoothing_oct <= 0:
        return output
    if mode == "gain":
        outer_slope = _signed_outside_slope(hold, GAIN_OUTSIDE_FADE_MAX_DB_PER_OCT, side="high")
        axis = "log"
        reference_oct = 1.0 / 6.0
    else:
        outer_slope = 0.0
        axis = "linear"
        reference_oct = 1.0 / 24.0
    connection = apply_one_sided_boundary(
        frequency,
        values,
        OneSidedBoundarySpec(
            fb_hz=boundary_hz,
            delta_oct=smoothing_oct,
            side="upper",
            axis=axis,
            outer_value=hold,
            outer_slope=outer_slope,
            reference_oct=reference_oct,
            lower_limit_hz=lower_limit_hz,
            name=f"Linear FIR EQ mask {mode} Hi",
        ),
    )
    if connection.item.applied:
        transition = (frequency >= float(connection.item.f_start_hz)) & (
            frequency <= float(connection.item.f_end_hz)
        )
        output[transition] = connection.values[transition]
    return output


def _signed_outside_slope(boundary_value: float, slope_per_oct: float, *, side: str) -> float:
    if boundary_value == 0.0:
        return 0.0
    direction = 1.0 if boundary_value > 0.0 else -1.0
    return direction * abs(float(slope_per_oct)) * (1.0 if side == "low" else -1.0)


def _phase_low_smoothing_oct(
    frequency: np.ndarray,
    boundary_hz: float | None,
    phase_passband: np.ndarray,
    requested_oct: float,
) -> float | None:
    if boundary_hz is None:
        return None
    if phase_passband.size == 0:
        return 0.0
    inner_hz = float(frequency[int(phase_passband[0])])
    available_oct = max(0.0, float(np.log2(inner_hz / float(boundary_hz))))
    return min(float(requested_oct), available_oct)


def _phase_high_smoothing_oct(
    frequency: np.ndarray,
    boundary_hz: float | None,
    phase_passband: np.ndarray,
    requested_oct: float,
) -> float | None:
    if boundary_hz is None:
        return None
    if phase_passband.size == 0:
        return 0.0
    inner_hz = float(frequency[int(phase_passband[-1])])
    available_oct = max(0.0, float(np.log2(float(boundary_hz) / inner_hz)))
    return min(float(requested_oct), available_oct)


def _boundary_hold_value(frequency: np.ndarray, values: np.ndarray, boundary_hz: float, mode: str) -> float:
    return float(np.interp(float(boundary_hz), frequency, values))


def _fade_boundary_value_to_zero_by_octave_slope(
    frequency: np.ndarray,
    boundary_hz: float,
    boundary_value: float,
    slope_per_oct: float,
    *,
    side: str,
) -> np.ndarray:
    value = float(boundary_value)
    slope = abs(float(slope_per_oct))
    if slope <= 0.0 or not np.isfinite(value) or value == 0.0:
        return np.full_like(np.asarray(frequency, dtype=float), value, dtype=float)

    freq = np.maximum(np.asarray(frequency, dtype=float), 1e-9)
    boundary = max(float(boundary_hz), 1e-9)
    if side == "low":
        distance_oct = np.maximum(np.log2(boundary / freq), 0.0)
    else:
        distance_oct = np.maximum(np.log2(freq / boundary), 0.0)

    magnitude = abs(value)
    sign = 1.0 if value >= 0.0 else -1.0
    total_oct = magnitude / slope
    if total_oct <= 0.0:
        return np.zeros_like(freq, dtype=float)

    linear_magnitude = np.maximum(magnitude - slope * distance_oct, 0.0)
    result = sign * linear_magnitude
    edge_oct = min(OUTSIDE_FADE_EDGE_SMOOTH_OCT, total_oct * 0.5)
    if edge_oct <= 0.0:
        return result

    end_start = max(total_oct - edge_oct, 0.0)
    end_mask = (distance_oct > end_start) & (distance_oct < total_oct)
    if np.any(end_mask):
        t = (distance_oct[end_mask] - end_start) / edge_oct
        y0 = sign * slope * edge_oct
        result[end_mask] = _cubic_hermite_values(t, y0, 0.0, -sign * slope, 0.0, edge_oct)

    result[distance_oct >= total_oct] = 0.0
    return result


def _cubic_hermite_values(
    t: np.ndarray,
    y0: float,
    y1: float,
    m0: float,
    m1: float,
    width: float,
) -> np.ndarray:
    values = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    h00 = 2.0 * values**3 - 3.0 * values**2 + 1.0
    h10 = values**3 - 2.0 * values**2 + values
    h01 = -2.0 * values**3 + 3.0 * values**2
    h11 = values**3 - values**2
    return h00 * y0 + h10 * width * m0 + h01 * y1 + h11 * width * m1


def _unwrap_phase_degrees(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    output = source.copy()
    finite = np.isfinite(output)
    if np.count_nonzero(finite) < 2:
        return output
    output[finite] = np.rad2deg(np.unwrap(np.deg2rad(output[finite])))
    return output
